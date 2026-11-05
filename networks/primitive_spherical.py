import math
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# Short-lived chart cache
# -----------------------------------------------------------------------------
# The head normally calls VirtualSphereProjector and GraphTriangleConstructor
# consecutively on the same tensors. To keep the public API unchanged, the
# projector stores the chart needed by the constructor in this small cache, and
# the constructor reuses it when the exact same tensors are passed in. On cache
# miss, the constructor falls back to the original recomputation path.
_CHART_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_CHART_CACHE_MAX_SIZE = 8


def _tensor_cache_id(t: torch.Tensor) -> tuple:
    # id(t) keeps reuse exact for the normal head path and also prevents stale
    # data_ptr reuse across later iterations from being accidentally matched.
    return (
        id(t),
        str(t.device),
        str(t.dtype),
        tuple(t.shape),
        tuple(t.stride()),
        int(t.storage_offset()),
        int(t.data_ptr()) if t.numel() > 0 else 0,
    )


def _chart_cache_key(
    nb_xyz: torch.Tensor,
    pos_q: torch.Tensor,
    knn_idx: torch.Tensor,
    knn_mask: torch.Tensor,
) -> tuple:
    return (
        _tensor_cache_id(nb_xyz),
        _tensor_cache_id(pos_q),
        _tensor_cache_id(knn_idx),
        _tensor_cache_id(knn_mask),
    )


def _chart_cache_put(
    key: tuple,
    u: torch.Tensor,
    v: torch.Tensor,
    chart_dbg: Dict[str, torch.Tensor],
    pt_valid: torch.Tensor,
) -> None:
    _CHART_CACHE[key] = (u, v, chart_dbg, pt_valid)
    _CHART_CACHE.move_to_end(key)
    while len(_CHART_CACHE) > _CHART_CACHE_MAX_SIZE:
        _CHART_CACHE.popitem(last=False)


def _chart_cache_get(key: tuple):
    item = _CHART_CACHE.get(key, None)
    if item is not None:
        _CHART_CACHE.move_to_end(key)
    return item


def _point_valid_from_graph_impl(
    nb_xyz: torch.Tensor,
    knn_idx: torch.Tensor,
    knn_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Recover point validity from the KNN graph. This function is shared by the
    projector-side cache preparation and the constructor, so the cached chart is
    computed with the same mask that the constructor would have used originally.
    """
    device = nb_xyz.device
    N, K, _ = nb_xyz.shape
    src_valid = knn_mask.bool().any(dim=-1)  # [N,K]
    tgt_valid = torch.zeros((N, K), device=device, dtype=torch.bool)

    if knn_idx.numel() > 0 and knn_mask.any():
        n = torch.arange(N, device=device)[:, None, None].expand_as(knn_idx)
        valid = knn_mask.bool()
        tgt_valid[n[valid], knn_idx.long()[valid]] = True

    pt_valid = src_valid | tgt_valid

    empty_row = pt_valid.sum(dim=1) == 0
    if empty_row.any():
        finite_valid = torch.isfinite(nb_xyz.float()).all(dim=-1)
        pt_valid = torch.where(empty_row[:, None], finite_valid, pt_valid)

    return pt_valid.bool()


def _safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    n = torch.linalg.norm(x, dim=dim, keepdim=True).clamp_min(eps)
    return x / n


def _masked_median(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    x: [N,K] float, mask: [N,K] bool -> [N] float
    """
    xf = x.float()
    m = mask.bool()
    inf = torch.tensor(float("inf"), device=xf.device, dtype=torch.float32)
    xs = xf.masked_fill(~m, inf)
    xs, _ = torch.sort(xs, dim=1)
    cnt = m.sum(dim=1).clamp_min(1)
    mid = ((cnt - 1) // 2).long()
    out = xs.gather(1, mid[:, None]).squeeze(1)
    out = torch.where(m.any(dim=1), out, torch.zeros_like(out))
    return out.float()


def _masked_quantile_last(x: torch.Tensor, mask: torch.Tensor, q: float) -> torch.Tensor:
    xf = x.float()
    m = mask.bool()
    xs = xf.masked_fill(~m, float("inf"))
    xs, _ = torch.sort(xs, dim=-1)
    cnt = m.sum(dim=-1).clamp_min(1)
    idx = torch.round((cnt.float() - 1.0) * float(q)).long()
    out = xs.gather(dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)
    out = torch.where(m.sum(dim=-1) > 0, out, torch.zeros_like(out))
    return out.float()


def _centroid_pca(x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    x: [N,K,3], mask: [N,K]
    Returns:
      mu:    [N,3]
      evals: [N,3] descending
      evecs: [N,3,3] columns, descending
    """
    xf = x.float()
    m = mask.float()
    cnt = mask.sum(dim=-1, keepdim=True).clamp_min(1).float()
    mu = (xf * m.unsqueeze(-1)).sum(dim=1) / cnt
    xc = (xf - mu[:, None, :]) * m.unsqueeze(-1)
    cov = torch.einsum("nki,nkj,nk->nij", xc, xc, m) / cnt[:, :, None]
    cov = 0.5 * (cov + cov.transpose(-1, -2))
    evals, evecs = torch.linalg.eigh(cov)
    order = torch.argsort(evals, dim=-1, descending=True)
    evals = torch.gather(evals, 1, order)
    evecs = torch.gather(evecs, 2, order[:, None, :].expand(-1, 3, -1))
    return mu.float(), evals.float(), evecs.float()


def _planar_uv(x: torch.Tensor, mask: torch.Tensor, mu: torch.Tensor, evecs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    t1 = _safe_normalize(evecs[:, :, 0], eps=1e-8).float()
    t2 = _safe_normalize(evecs[:, :, 1], eps=1e-8).float()
    xc = x.float() - mu[:, None, :].float()
    mf = mask.float()
    u = (xc * t1[:, None, :]).sum(dim=-1) * mf
    v = (xc * t2[:, None, :]).sum(dim=-1) * mf
    return u.float(), v.float()


def _pca_quantile_center(
    x: torch.Tensor,
    mask: torch.Tensor,
    mu: torch.Tensor,
    evecs: torch.Tensor,
    q_lo: float,
    q_hi: float,
) -> torch.Tensor:
    xf = x.float()
    mu = mu.float()
    evecs = evecs.float()
    m = mask.bool()
    xc = xf - mu[:, None, :]
    e0, e1, e2 = evecs[:, :, 0], evecs[:, :, 1], evecs[:, :, 2]
    p0 = (xc * e0[:, None, :]).sum(dim=-1)
    p1 = (xc * e1[:, None, :]).sum(dim=-1)
    p2 = (xc * e2[:, None, :]).sum(dim=-1)

    def qtl(p: torch.Tensor) -> torch.Tensor:
        lo = _masked_quantile_last(p, m, q=q_lo)
        hi = _masked_quantile_last(p, m, q=q_hi)
        return 0.5 * (lo + hi)

    c0, c1, c2 = qtl(p0), qtl(p1), qtl(p2)
    c = mu + c0[:, None] * e0 + c1[:, None] * e1 + c2[:, None] * e2
    return c.float()


def _tangent_basis(d0: torch.Tensor, axis_hint: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    d0 = _safe_normalize(d0.float(), eps=1e-8).float()
    ah = axis_hint.float()
    t1 = ah - (ah * d0).sum(dim=-1, keepdim=True) * d0
    n1 = torch.linalg.norm(t1, dim=-1, keepdim=True)
    bad = n1.squeeze(-1) < 1e-6
    if bad.any():
        axes = torch.eye(3, device=d0.device, dtype=d0.dtype)
        align = torch.abs(d0 @ axes.t())
        idx = torch.argmin(align, dim=-1)
        a = axes[idx]
        t1_alt = a - (a * d0).sum(dim=-1, keepdim=True) * d0
        t1 = torch.where(bad[:, None], t1_alt, t1)
    t1 = _safe_normalize(t1, eps=1e-8).float()
    t2 = _safe_normalize(torch.cross(d0, t1, dim=-1), eps=1e-8).float()
    return t1, t2


def _logmap_uv(
    dirs: torch.Tensor,
    mask: torch.Tensor,
    d0: torch.Tensor,
    axis_hint: torch.Tensor,
    anti_pole_eps: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    d = dirs.float()
    m = mask.bool()
    d0 = _safe_normalize(d0.float(), eps=1e-8).float()
    dot0 = (d * d0[:, None, :]).sum(dim=-1).clamp(-1.0, 1.0)
    anti = (dot0 < (-1.0 + float(anti_pole_eps))) & m
    ok = ~anti.any(dim=-1)
    theta = torch.acos(dot0)
    vtan = d - dot0[..., None] * d0[:, None, :]
    s = torch.linalg.norm(vtan, dim=-1).clamp_min(1e-8)
    z = vtan * (theta / s)[..., None]
    t1, t2 = _tangent_basis(d0, axis_hint)
    mf = m.float()
    u = (z * t1[:, None, :]).sum(dim=-1) * mf
    v = (z * t2[:, None, :]).sum(dim=-1) * mf
    return u.float(), v.float(), ok.bool()


def _simple_chart(
    nb_xyz: torch.Tensor,
    nb_mask: torch.Tensor,
    q_xyz: torch.Tensor,
    min_valid_points: int = 4,
    center_rmin_frac: float = 0.06,
    thick_min_frac: float = 0.02,
    bw_weight: float = 0.6,
    anti_pole_eps: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Minimal chart choice:
      - use PCA plane by default;
      - use spherical log-map only when the PCA-quantile center is clear and has points on both sides.
    """
    x = nb_xyz.float()
    m = nb_mask.bool()
    device = x.device
    N, K, _ = x.shape
    valid_n = m.sum(dim=-1)

    mu, evals, evecs = _centroid_pca(x, m)
    normal = _safe_normalize(evecs[:, :, 2], eps=1e-8).float()
    r_mu = torch.linalg.norm(x - mu[:, None, :], dim=-1)
    r_loc = _masked_quantile_last(r_mu, m, q=0.5).clamp_min(1e-4)

    c_q = _pca_quantile_center(x, m, mu, evecs, 0.10, 0.90)
    rx_c = x - c_q[:, None, :]
    rr_c = torch.linalg.norm(rx_c, dim=-1)
    rr_q10 = _masked_quantile_last(rr_c, m, q=0.10)

    signed = (rx_c * normal[:, None, :]).sum(dim=-1)
    band = (thick_min_frac * r_loc).clamp_min(1e-6)
    pos_n = ((signed > band[:, None]) & m).sum(dim=-1)
    neg_n = ((signed < -band[:, None]) & m).sum(dim=-1)
    min_side = torch.clamp(torch.ceil(valid_n.float() * 0.20).long(), min=1)
    two_side = (pos_n >= min_side) & (neg_n >= min_side)

    near_center = ((rr_c < (center_rmin_frac * r_loc)[:, None]) & m).float().sum(dim=-1)
    near_frac = near_center / valid_n.clamp_min(1).float()
    center_clear = rr_q10 > (center_rmin_frac * r_loc).clamp_min(1e-6)
    band_frac = ((signed.abs() < band[:, None]) & m).float().sum(dim=-1) / valid_n.clamp_min(1).float()
    band_sparse = band_frac < float(bw_weight)

    use_sphere = (valid_n >= int(min_valid_points)) & two_side & center_clear & band_sparse

    u, v = _planar_uv(x, m, mu, evecs)
    c_out = mu.clone()
    sphere_ok = torch.zeros((N,), device=device, dtype=torch.bool)

    if use_sphere.any():
        idx = torch.where(use_sphere)[0]
        xs = x.index_select(0, idx)
        ms = m.index_select(0, idx)
        cs = c_q.index_select(0, idx)
        evs = evecs.index_select(0, idx)
        q_s = q_xyz.float().index_select(0, idx)

        dirs = _safe_normalize(xs - cs[:, None, :], dim=-1, eps=1e-8).float()
        mean_dir_raw = (dirs * ms.float().unsqueeze(-1)).sum(dim=1)
        mean_norm = torch.linalg.norm(mean_dir_raw, dim=-1)
        d0_mean = _safe_normalize(mean_dir_raw, eps=1e-8).float()

        d_q = torch.linalg.norm(xs - q_s[:, None, :], dim=-1).masked_fill(~ms, float("inf"))
        nearest = d_q.argmin(dim=1)
        ar = torch.arange(idx.numel(), device=device, dtype=torch.long)
        d0_near = dirs[ar, nearest]
        d0 = torch.where((mean_norm > 1e-6)[:, None], d0_mean, d0_near)

        axis_hint = evs[:, :, 0]
        u_s, v_s, ok_s = _logmap_uv(dirs, ms, d0, axis_hint, anti_pole_eps=anti_pole_eps)

        u = u.clone()
        v = v.clone()
        c_out = c_out.clone()
        sphere_ok.index_copy_(0, idx, ok_s)

        ok_global = idx[ok_s]
        if ok_global.numel() > 0:
            u.index_copy_(0, ok_global, u_s[ok_s])
            v.index_copy_(0, ok_global, v_s[ok_s])
            c_out.index_copy_(0, ok_global, cs[ok_s])

    chart_id = torch.zeros((N, K), device=device, dtype=torch.long)
    dbg = {
        "use_sphere": use_sphere.float(),
        "thick": center_clear.float(),
        "one_sided": (~two_side).float(),
        "center_ok": center_clear.float(),
        "sph_ok": sphere_ok.float(),
        "single_chart": use_sphere.float(),
        "seam_frac_est": torch.zeros((N,), device=device, dtype=torch.float32),
        "chart_id": chart_id,
        "two_side": two_side.float(),
        "band_sparse": band_sparse.float(),
        "near_center_frac": near_frac.float(),
        "rr_q10": rr_q10.float(),
    }
    return u.float(), v.float(), c_out.float(), chart_id, dbg


class VirtualSphereProjector(nn.Module):
    """
    Drop-in minimal projector + fused KNN graph builder.

    The public parameters and forward output follow the provided implementation.
    Internally, the logic is simplified to:
      PCA quantile center -> PCA plane or spherical log-map -> simple UV KNN.
    """

    def __init__(
        self,
        min_valid_points: int = 4,

        # simple spherical decision: thickness only
        top_m: int = 3,
        thick_tau: float = 3.5,
        thick_min_frac: float = 0.02,

        # center selection
        center_rmin_frac: float = 0.06,
        bw_min: float = 0.06,
        bw_weight: float = 0.6,

        # base selection
        anti_pole_eps: float = 1e-3,

        # graph
        seam_tau: float = 0.20,
        intra_k: int = 4,
        cross_k: int = 2,

        # speed-up: single chart fast path
        single_chart_min_dot: float = 0.05,
        single_chart_max_seam_frac: float = 0.15,
    ):
        super().__init__()
        self.min_valid_points = int(min_valid_points)
        self.top_m = int(top_m)
        self.thick_tau = float(thick_tau)
        self.thick_min_frac = float(thick_min_frac)
        self.center_rmin_frac = float(center_rmin_frac)
        self.bw_min = float(bw_min)
        self.bw_weight = float(bw_weight)
        self.anti_pole_eps = float(anti_pole_eps)
        self.seam_tau = float(seam_tau)
        self.intra_k = int(intra_k)
        self.cross_k = int(cross_k)
        self.single_chart_min_dot = float(single_chart_min_dot)
        self.single_chart_max_seam_frac = float(single_chart_max_seam_frac)

    # Small tensor helpers used by the primitive construction path.
    @staticmethod
    def _masked_sort(x: torch.Tensor, mask: torch.Tensor, fill: float) -> torch.Tensor:
        xf = x.float()
        xs = xf.masked_fill(~mask.bool(), float(fill))
        xs, _ = torch.sort(xs, dim=-1)
        return xs

    @staticmethod
    def _masked_quantile_last(x: torch.Tensor, mask: torch.Tensor, q: float) -> torch.Tensor:
        return _masked_quantile_last(x, mask, q)

    @staticmethod
    def _centroid_pca(x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _centroid_pca(x, mask)

    @staticmethod
    def _planar_uv(x: torch.Tensor, mask: torch.Tensor, mu: torch.Tensor, evecs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return _planar_uv(x, mask, mu, evecs)

    @staticmethod
    def _pca_quantile_center(
        x: torch.Tensor, mask: torch.Tensor, mu: torch.Tensor, evecs: torch.Tensor, q_lo: float, q_hi: float
    ) -> torch.Tensor:
        return _pca_quantile_center(x, mask, mu, evecs, q_lo, q_hi)

    @staticmethod
    def _tangent_basis(d0: torch.Tensor, axis_hint: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return _tangent_basis(d0, axis_hint)

    def _logmap_uv(
        self,
        dirs: torch.Tensor,
        mask: torch.Tensor,
        d0: torch.Tensor,
        axis_hint: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _logmap_uv(dirs, mask, d0, axis_hint, self.anti_pole_eps)

    def _get_eye(self, K: int, device: torch.device) -> torch.Tensor:
        if not hasattr(self, "_eye_cache"):
            self._eye_cache = {}
        key = (int(K), str(device))
        eye = self._eye_cache.get(key, None)
        if eye is None:
            eye = torch.eye(K, device=device, dtype=torch.bool)[None, :, :]
            self._eye_cache[key] = eye
        return eye

    def _build_simple_knn(
        self, u: torch.Tensor, v: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = u.device
        N, K = u.shape
        deg = self.intra_k + self.cross_k
        knn_idx = torch.zeros((N, K, deg), device=device, dtype=torch.long)
        knn_mask = torch.zeros((N, K, deg), device=device, dtype=torch.bool)
        if deg <= 0 or K <= 1:
            return knn_idx, knn_mask

        du = u.float()[:, :, None] - u.float()[:, None, :]
        dv = v.float()[:, :, None] - v.float()[:, None, :]
        dist2 = du * du + dv * dv
        m = mask.bool()
        valid_pair = m[:, :, None] & m[:, None, :]
        eye = self._get_eye(K, device)
        valid_pair = valid_pair & (~eye)
        dist2 = dist2.masked_fill(~valid_pair, float("inf"))
        k_eff = min(deg, K - 1)
        vals, idx = torch.topk(dist2, k=k_eff, dim=-1, largest=False, sorted=True)
        mask_eff = vals < float("inf")
        knn_idx[:, :, :k_eff] = idx
        knn_mask[:, :, :k_eff] = mask_eff & m[:, :, None]
        return knn_idx, knn_mask

    @torch.no_grad()
    def forward(
        self, nb_xyz: torch.Tensor, nb_mask: torch.Tensor, q_xyz: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        out_dtype = nb_xyz.dtype
        u, v, c_out, chart_id, dbg = _simple_chart(
            nb_xyz=nb_xyz,
            nb_mask=nb_mask,
            q_xyz=q_xyz,
            min_valid_points=self.min_valid_points,
            center_rmin_frac=self.center_rmin_frac,
            thick_min_frac=self.thick_min_frac,
            bw_weight=self.bw_weight,
            anti_pole_eps=self.anti_pole_eps,
        )
        knn_idx, knn_mask = self._build_simple_knn(u, v, nb_mask)
        dbg["chart_id"] = chart_id

        # Prepare the exact chart that GraphTriangleConstructor.forward would
        # otherwise recompute from the graph-derived pt_valid mask. In the
        # common path pt_valid == nb_mask, so we reuse the projector chart. In
        # rare degenerate rows, we compute the constructor-equivalent chart here
        # and cache that instead, preserving the original execution logic.
        pt_valid_for_constructor = _point_valid_from_graph_impl(nb_xyz, knn_idx, knn_mask)
        if torch.equal(pt_valid_for_constructor, nb_mask.bool()):
            u_cache, v_cache, chart_dbg_cache = u, v, dbg
        else:
            u_cache, v_cache, _, _, chart_dbg_cache = _simple_chart(
                nb_xyz=nb_xyz,
                nb_mask=pt_valid_for_constructor,
                q_xyz=q_xyz,
                min_valid_points=4,
                center_rmin_frac=0.06,
                thick_min_frac=0.02,
                bw_weight=0.6,
                anti_pole_eps=1e-3,
            )
        _chart_cache_put(
            _chart_cache_key(nb_xyz, q_xyz, knn_idx, knn_mask),
            u_cache,
            v_cache,
            chart_dbg_cache,
            pt_valid_for_constructor,
        )
        return c_out.to(out_dtype), knn_idx, knn_mask, dbg



class GraphTriangleConstructor(nn.Module):
    """
    Drop-in pure geometry triangle constructor.

    Public parameters and forward signature follow the provided implementation.
    Fixed version for clean/noise=0 cases:
      1) Generate triangle candidates by enumerating local neighbor pairs instead of
         requiring a strict directed-edge CCW third point.
      2) Use CCW only to orient triangles, not as a hard existence gate.
      3) Fill missing primitive slots with repeated-point primitives [i,i,i], so the
         representation degenerates to point-level aggregation when triangles are insufficient.
    """

    def __init__(
        self,
        keep_k: int = 10,
        cand_M: int = 12,
        intra_k: int = 4,
        cross_k: int = 2,
        symmetric_graph: bool = True,

        w_qdist: float = 2.0,
        w_quality: float = 1.0,
        w_span: float = 1.0,
        w_compact: float = 0.5,
        w_cross: float = 0.25,

        max_faces_per_edge: int = 2,
        enforce_connected: bool = True,
        outward_bonus: float = 0.0,

        score_neg_inf: float = -1e9,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.keep_k = int(keep_k)
        self.cand_M = int(cand_M)
        self.intra_k = int(intra_k)
        self.cross_k = int(cross_k)
        self.symmetric_graph = bool(symmetric_graph)
        self.w_qdist = float(w_qdist)
        self.w_quality = float(w_quality)
        self.w_span = float(w_span)
        self.w_compact = float(w_compact)
        self.w_cross = float(w_cross)
        self.max_faces_per_edge = int(max_faces_per_edge)
        self.enforce_connected = bool(enforce_connected)
        self.outward_bonus = float(outward_bonus)
        self.score_neg_inf = float(score_neg_inf)
        self.eps = float(eps)

    def _point_valid_from_graph(
        self,
        nb_xyz: torch.Tensor,
        knn_idx: torch.Tensor,
        knn_mask: torch.Tensor,
    ) -> torch.Tensor:
        return _point_valid_from_graph_impl(nb_xyz, knn_idx, knn_mask)

    def _get_pair_index(self, S: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(self, "_pair_cache"):
            self._pair_cache = {}
        key = (int(S), str(device))
        cached = self._pair_cache.get(key, None)
        if cached is not None:
            return cached

        pair_a = []
        pair_b = []
        for a in range(S):
            for b in range(a + 1, S):
                pair_a.append(a)
                pair_b.append(b)
        pair_a = torch.tensor(pair_a, device=device, dtype=torch.long)
        pair_b = torch.tensor(pair_b, device=device, dtype=torch.long)
        self._pair_cache[key] = (pair_a, pair_b)
        return pair_a, pair_b

    def _neighbor_pair_candidates(
        self,
        nb_xyz: torch.Tensor,
        pos_q: torch.Tensor,
        pt_valid: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        knn_idx: torch.Tensor,
        knn_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build triangle candidates by enumerating neighbor pairs around each center point.

        For each valid center i and two valid local neighbors j,k, construct (i,j,k).
        The signed UV area is used only to orient triangles consistently. A triangle is
        valid if |area| is non-degenerate. This avoids the clean-sampling failure of
        strict directed-edge CCW gating.
        """
        device = nb_xyz.device
        N, K, _ = nb_xyz.shape
        S = knn_idx.shape[-1]

        if K < 3 or S < 2:
            tri_idx = torch.zeros((N, 0, 3), device=device, dtype=torch.long)
            tri_mask = torch.zeros((N, 0), device=device, dtype=torch.bool)
            tri_score = torch.empty((N, 0), device=device, dtype=torch.float32)
            return tri_idx, tri_mask, tri_score

        pair_a, pair_b = self._get_pair_index(S, device)
        P = pair_a.numel()
        M = K * P

        center = torch.arange(K, device=device, dtype=torch.long).view(1, K, 1).expand(N, K, P)
        j = knn_idx.long()[:, :, pair_a]
        k = knn_idx.long()[:, :, pair_b]
        pair_valid = knn_mask.bool()[:, :, pair_a] & knn_mask.bool()[:, :, pair_b]

        center_f = center.reshape(N, M)
        j_f = j.reshape(N, M)
        k_f = k.reshape(N, M)
        valid_f = pair_valid.reshape(N, M)

        arN = torch.arange(N, device=device, dtype=torch.long)[:, None]
        valid_f = (
            valid_f
            & pt_valid.gather(1, center_f)
            & pt_valid.gather(1, j_f)
            & pt_valid.gather(1, k_f)
            & (center_f != j_f)
            & (center_f != k_f)
            & (j_f != k_f)
        )

        uv = torch.stack([u.float(), v.float()], dim=-1)  # [N,K,2]
        ui = uv[arN, center_f]
        uj = uv[arN, j_f]
        uk = uv[arN, k_f]

        e1 = uj - ui
        e2 = uk - ui
        area2 = e1[..., 0] * e2[..., 1] - e1[..., 1] * e2[..., 0]  # signed 2D area * 2

        # Adaptive area threshold, small enough for clean surfaces but avoids exact collinearity.
        uv_r = torch.linalg.norm(uv - uv.mean(dim=1, keepdim=True), dim=-1)
        r_uv = _masked_median(uv_r, pt_valid).clamp_min(1e-6)
        area_eps = torch.clamp((1e-7 * r_uv * r_uv), min=self.eps).view(N, 1)
        nondeg = area2.abs() > area_eps

        valid_f = valid_f & nondeg

        # Orient as CCW by swapping j and k when the signed area is negative.
        j_ccw = torch.where(area2 < 0.0, k_f, j_f)
        k_ccw = torch.where(area2 < 0.0, j_f, k_f)
        tri_idx = torch.stack([center_f, j_ccw, k_ccw], dim=-1).long()

        # Suppress duplicate triangles, retaining the first occurrence in
        # candidate order with [N,M] memory.
        sorted_idx, _ = torch.sort(tri_idx, dim=-1)
        key = (sorted_idx[..., 0] * (K * K) + sorted_idx[..., 1] * K + sorted_idx[..., 2]).long()
        pos = torch.arange(M, device=device, dtype=torch.long).view(1, M)
        order = torch.argsort(key * M + pos, dim=1)
        key_sorted = torch.gather(key, 1, order)
        dup_sorted = torch.zeros((N, M), device=device, dtype=torch.bool)
        dup_sorted[:, 1:] = key_sorted[:, 1:] == key_sorted[:, :-1]
        dup = torch.zeros((N, M), device=device, dtype=torch.bool)
        dup.scatter_(1, order, dup_sorted)
        tri_mask = valid_f & (~dup)

        x = nb_xyz.float()
        q = pos_q.float()
        pi = x[arN, tri_idx[..., 0]]
        pj = x[arN, tri_idx[..., 1]]
        pk = x[arN, tri_idx[..., 2]]
        ctr = (pi + pj + pk) / 3.0

        # Nearest-first score. A small quality term is kept only as tie breaking.
        qdist = torch.linalg.norm(ctr - q[:, None, :], dim=-1)
        e01 = torch.linalg.norm(pj - pi, dim=-1)
        e12 = torch.linalg.norm(pk - pj, dim=-1)
        e20 = torch.linalg.norm(pi - pk, dim=-1)
        max_edge = torch.max(torch.stack([e01, e12, e20], dim=-1), dim=-1).values.clamp_min(self.eps)
        area3 = 0.5 * torch.linalg.norm(torch.cross(pj - pi, pk - pi, dim=-1), dim=-1)
        quality = area3 / (max_edge * max_edge + self.eps)
        tri_score = (-qdist + 0.05 * quality).float()
        tri_score = tri_score.masked_fill(~tri_mask, self.score_neg_inf)

        return tri_idx, tri_mask.bool(), tri_score.float()

    def _point_fallback(
        self,
        nb_xyz: torch.Tensor,
        pos_q: torch.Tensor,
        pt_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Repeated-point primitive fallback:
          tri_idx[n,t] = [i,i,i]
        If downstream averages three vertex features, this exactly becomes point-level aggregation.
        """
        device = nb_xyz.device
        N, K, _ = nb_xyz.shape
        keep_k = self.keep_k

        tri_idx = torch.zeros((N, keep_k, 3), device=device, dtype=torch.long)
        tri_mask = torch.zeros((N, keep_k), device=device, dtype=torch.bool)
        tri_ctr = torch.zeros((N, keep_k, 3), device=device, dtype=torch.float32)
        tri_score = torch.full((N, keep_k), self.score_neg_inf, device=device, dtype=torch.float32)

        if keep_k <= 0 or K <= 0:
            return tri_idx, tri_mask, tri_ctr, tri_score

        x = nb_xyz.float()
        q = pos_q.float()
        dist = torch.linalg.norm(x - q[:, None, :], dim=-1).masked_fill(~pt_valid.bool(), float("inf"))
        k_out = min(keep_k, K)
        top_dist, top_idx = torch.topk(dist, k=k_out, dim=1, largest=False, sorted=True)
        valid = top_dist < float("inf")

        tri_idx[:, :k_out, 0] = top_idx
        tri_idx[:, :k_out, 1] = top_idx
        tri_idx[:, :k_out, 2] = top_idx
        tri_mask[:, :k_out] = valid
        tri_score[:, :k_out] = (-top_dist).masked_fill(~valid, self.score_neg_inf).float()

        arN = torch.arange(N, device=device, dtype=torch.long)[:, None]
        pts = x[arN, top_idx]
        tri_ctr[:, :k_out, :] = (pts - q[:, None, :]) * valid.float().unsqueeze(-1)
        return tri_idx, tri_mask, tri_ctr, tri_score

    @torch.no_grad()
    def forward(
        self,
        nb_xyz: torch.Tensor,
        nb_feat: Optional[torch.Tensor],
        pos_q: torch.Tensor,
        knn_idx: torch.Tensor,
        knn_mask: torch.Tensor,
    ):
        device = nb_xyz.device
        N, K, _ = nb_xyz.shape
        assert knn_idx.shape[:2] == (N, K)
        assert knn_mask.shape == knn_idx.shape
        assert knn_mask.dtype == torch.bool
        assert knn_idx.dtype == torch.long

        cache_key = _chart_cache_key(nb_xyz, pos_q, knn_idx, knn_mask)
        cached_chart = _chart_cache_get(cache_key)
        if cached_chart is not None:
            u, v, chart_dbg, pt_valid = cached_chart
        else:
            pt_valid = self._point_valid_from_graph(nb_xyz, knn_idx, knn_mask)
            u, v, _, _, chart_dbg = _simple_chart(
                nb_xyz=nb_xyz,
                nb_mask=pt_valid,
                q_xyz=pos_q,
                min_valid_points=4,
                center_rmin_frac=0.06,
                thick_min_frac=0.02,
                bw_weight=0.6,
                anti_pole_eps=1e-3,
            )

        tri_all_idx, tri_all_mask, tri_all_score = self._neighbor_pair_candidates(
            nb_xyz=nb_xyz,
            pos_q=pos_q,
            pt_valid=pt_valid,
            u=u,
            v=v,
            knn_idx=knn_idx,
            knn_mask=knn_mask,
        )

        T = tri_all_idx.shape[1]
        sel_idx = torch.zeros((N, self.keep_k, 3), device=device, dtype=torch.long)
        sel_mask = torch.zeros((N, self.keep_k), device=device, dtype=torch.bool)
        sel_ctr = torch.zeros((N, self.keep_k, 3), device=device, dtype=torch.float32)
        sel_score = torch.full((N, self.keep_k), self.score_neg_inf, device=device, dtype=torch.float32)

        raw_tri_selected = torch.zeros((N,), device=device, dtype=torch.long)

        if T > 0 and self.keep_k > 0:
            k_out = min(self.keep_k, T)
            top_score, top_pos = torch.topk(tri_all_score, k=k_out, dim=1, largest=True, sorted=True)
            top_mask = top_score > (self.score_neg_inf / 10.0)
            top_idx = torch.gather(tri_all_idx, 1, top_pos.unsqueeze(-1).expand(-1, -1, 3))

            sel_idx[:, :k_out] = top_idx
            sel_mask[:, :k_out] = top_mask
            sel_score[:, :k_out] = top_score.masked_fill(~top_mask, self.score_neg_inf)

            arN = torch.arange(N, device=device, dtype=torch.long)[:, None]
            a, b, c = top_idx[..., 0], top_idx[..., 1], top_idx[..., 2]
            x = nb_xyz.float()
            q = pos_q.float()
            ctr = (x[arN, a] + x[arN, b] + x[arN, c]) / 3.0 - q[:, None, :]
            sel_ctr[:, :k_out] = ctr * top_mask.float().unsqueeze(-1)
            raw_tri_selected = sel_mask.sum(dim=1)

        # Fill missing slots with repeated-point primitives, preserving output format.
        p_idx, p_mask, p_ctr, p_score = self._point_fallback(nb_xyz, pos_q, pt_valid)
        need_fill = ~sel_mask
        fill_valid = need_fill & p_mask
        sel_idx = torch.where(fill_valid[:, :, None], p_idx, sel_idx)
        sel_ctr = torch.where(fill_valid[:, :, None], p_ctr, sel_ctr)
        sel_score = torch.where(fill_valid, p_score, sel_score)
        sel_mask = sel_mask | fill_valid

        debug = {
            "pt_valid_n": pt_valid.sum(dim=1),
            "tri_total": tri_all_mask.sum(dim=1) if T > 0 else torch.zeros((N,), device=device, dtype=torch.long),
            "tri_pool": torch.minimum(
                tri_all_mask.sum(dim=1) if T > 0 else torch.zeros((N,), device=device, dtype=torch.long),
                torch.full((N,), self.keep_k, device=device, dtype=torch.long),
            ),
            # triangle-only count before point fallback
            "tri_selected": raw_tri_selected,
            # number of repeated-point primitives used to fill invalid slots
            "point_fallback": fill_valid.sum(dim=1),
            # final valid primitive count after fallback
            "all_selected": sel_mask.sum(dim=1),
            "use_sphere": chart_dbg["use_sphere"],
        }
        return sel_idx, sel_mask, sel_ctr, sel_score, debug
