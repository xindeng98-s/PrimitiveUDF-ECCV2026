from __future__ import annotations

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- geometry-only primitive modules ----
from networks.primitive_spherical import VirtualSphereProjector, GraphTriangleConstructor

# ---- PyG / torch-cluster KNN ----
try:
    from torch_geometric.nn import knn
except Exception:
    try:
        from torch_cluster import knn  # type: ignore
    except Exception:
        knn = None




def offset2batch(offset: torch.Tensor) -> torch.Tensor:
    """
    offset: [B] cumulative sum, offset[-1] == N
    returns: [N] batch id per point
    """
    offset = offset.to(torch.long)
    counts = torch.diff(torch.cat([offset.new_zeros(1), offset], dim=0), dim=0)
    return torch.repeat_interleave(
        torch.arange(offset.numel(), device=offset.device, dtype=torch.long),
        counts,
    )

def _dense_knn_from_edges(y_idx: torch.Tensor, x_idx: torch.Tensor, M: int, k: int, device):
    """
    Pack sparse knn edges (y_idx,x_idx) into dense [M,k] with -1 padding.
    y_idx: query indices in [0, M)
    x_idx: support indices in [0, Ns)
    """
    y_idx = y_idx.to(torch.long)
    x_idx = x_idx.to(torch.long)

    perm = torch.argsort(y_idx, stable=True)
    y = y_idx[perm]
    x = x_idx[perm]

    cnt = torch.bincount(y, minlength=M)  # [M]
    start = torch.cumsum(cnt, dim=0) - cnt
    local = torch.arange(y.numel(), device=device) - start[y]  # [E]

    out = torch.full((M, k), -1, device=device, dtype=torch.long)
    m = local < k
    out[y[m], local[m]] = x[m]
    return out

def knn_query_dense(
    k: int,
    sup_xyz: torch.Tensor, sup_offset: torch.Tensor,
    qry_xyz: torch.Tensor, qry_offset: torch.Tensor,
):
    """
    support: sup_xyz [Ns,3], sup_offset [B]
    query:   qry_xyz [Nq,3], qry_offset [B]
    return:  idx [Nq,k] in global indices of sup_xyz, padded with -1

    Offsets are converted to batch ids before KNN so neighbors never cross
    batch boundaries. If a query has fewer than k valid in-batch neighbors,
    the remaining slots stay -1.
    """
    if knn is None:
        raise ImportError("Install torch_geometric or torch_cluster for KNN queries.")

    device = sup_xyz.device
    sup_offset = sup_offset.to(torch.long)
    qry_offset = qry_offset.to(torch.long)

    batch_x = offset2batch(sup_offset)
    batch_y = offset2batch(qry_offset)

    assign_index = knn(
        x=sup_xyz,
        y=qry_xyz,
        k=k,
        batch_x=batch_x,
        batch_y=batch_y,
    )
    y_idx, x_idx = assign_index[0], assign_index[1]  # y=query, x=support
    Nq = int(qry_xyz.shape[0])
    return _dense_knn_from_edges(y_idx, x_idx, Nq, k, device)

def masked_softmax_safe(logits: torch.Tensor, mask: torch.Tensor, dim: int):
    """
    logits: [N,K] or [N,K,H]
    mask:   [N,K] bool
    """
    if logits.ndim == 2:
        logits = logits.masked_fill(~mask, -1e9)
        w = torch.softmax(logits.float(), dim=dim)
        all_off = (~mask).all(dim=dim, keepdim=True)
        w = torch.where(all_off, torch.zeros_like(w), w)
        return w

    m = mask.unsqueeze(-1)
    logits = logits.masked_fill(~m, -1e9)
    w = torch.softmax(logits.float(), dim=dim)
    all_off = (~mask).all(dim=dim, keepdim=True).unsqueeze(-1)
    w = torch.where(all_off, torch.zeros_like(w), w)
    return w

def _pick_nh(d: int, prefer=(8, 6, 4, 3, 2, 1)) -> int:
    for h in prefer:
        if d % h == 0:
            return h
    return 1

def _masked_quantile_1d(x: torch.Tensor, mask: torch.Tensor, q: float) -> torch.Tensor:
    """
    x, mask: [N,K] -> [N]
    Quantile via sort with inf-fill on masked elements.
    """
    xf = x.float()
    m = mask.bool()
    inf = torch.tensor(float("inf"), device=xf.device, dtype=xf.dtype)
    xs = xf.masked_fill(~m, inf)
    xs, _ = torch.sort(xs, dim=-1)

    cnt = m.sum(dim=-1)  # [N]
    idx = torch.round((cnt.clamp_min(1) - 1).to(xf.dtype) * float(q)).long()
    out = xs.gather(dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)
    out = torch.where(cnt > 0, out, torch.zeros_like(out))
    return out.float()

def gather_global_zero(src: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    src: [G,C], idx: arbitrary shape of global indices.
    Invalid indices (-1 or >=G) gather as all-zero values.
    """
    idx = idx.long()
    G, C = src.shape
    out_shape = (*idx.shape, C)
    if G <= 0:
        return torch.zeros(out_shape, device=src.device, dtype=src.dtype)

    valid = (idx >= 0) & (idx < G)
    idx_safe = idx.clamp(0, G - 1)
    out = src[idx_safe].reshape(out_shape)
    return out * valid.unsqueeze(-1).to(out.dtype)

def gather_relative_xyz_zero(
    src_xyz: torch.Tensor,
    idx: torch.Tensor,
    query_xyz: torch.Tensor,
) -> torch.Tensor:
    """
    Gather neighbor-query relative vectors. Invalid indices return zero vectors.
    """
    idx = idx.long()
    G = int(src_xyz.shape[0])
    if G <= 0:
        return torch.zeros((*idx.shape, 3), device=query_xyz.device, dtype=query_xyz.dtype)

    valid = (idx >= 0) & (idx < G)
    idx_safe = idx.clamp(0, G - 1)
    rel = src_xyz[idx_safe].float() - query_xyz[:, None, :].float()
    rel = rel * valid.unsqueeze(-1).to(rel.dtype)
    return rel.to(src_xyz.dtype)

def gather_local_tri(src: torch.Tensor, tri_idx: torch.Tensor) -> torch.Tensor:
    """
    src:     [N,K,C]
    tri_idx: [N,T,3] local indices in 0..K-1
    return:  [N,T,3,C], with invalid local indices gathered as zero.
    """
    N, K, C = src.shape
    _, T, V = tri_idx.shape
    assert V == 3
    if K <= 0:
        return torch.zeros((N, T, 3, C), device=src.device, dtype=src.dtype)

    valid = (tri_idx >= 0) & (tri_idx < K)
    base = (torch.arange(N, device=src.device, dtype=torch.long) * K)[:, None, None]   # [N,1,1]
    flat_idx = (tri_idx.clamp(0, K - 1) + base).reshape(-1)                             # [N*T*3]
    flat_src = src.reshape(N * K, C)                                                    # [N*K,C]
    out = flat_src[flat_idx].reshape(N, T, 3, C)                                        # [N,T,3,C]
    return out * valid.unsqueeze(-1).to(out.dtype)

class Att_pooling_regressor_logitsbias(nn.Module):
    """
    Attention pooling regressor with relative-position bias added in logits space.
    """

    def __init__(
        self,
        d_prim: int,
        pos_hidden: int = 32,
        nh: int | None = None,
        pos_gate_init: float = -10.0,
        use_r_median_norm: bool = True,
    ):
        super().__init__()
        self.d_prim = int(d_prim)
        self.nh = int(_pick_nh(self.d_prim) if nh is None else nh)
        assert self.d_prim % self.nh == 0
        self.dh = self.d_prim // self.nh

        self.use_r_median_norm = bool(use_r_median_norm)

        # content path
        self.k_feat_proj = nn.Linear(self.d_prim, self.d_prim, bias=True)
        self.k_ln = nn.LayerNorm(self.d_prim)

        self.score_proj = nn.Sequential(
            nn.Linear(self.d_prim, self.nh * 2),
            nn.ReLU(True),
            nn.Linear(self.nh * 2, self.nh),
        )

        # pos -> per-head logits bias
        self.pos2logits = nn.Sequential(
            nn.Linear(4, pos_hidden),
            nn.ReLU(True),
            nn.Linear(pos_hidden, pos_hidden),
            nn.ReLU(True),
            nn.Linear(pos_hidden, self.nh),
        )

        self.pos_gate = nn.Parameter(torch.tensor(float(pos_gate_init)))

        # value path
        self.v_proj = nn.Linear(self.d_prim, self.nh * self.dh, bias=True)
        self.post = nn.Sequential(
            nn.Linear(self.nh * self.dh, self.d_prim),
            nn.ReLU(True),
        )

    def forward(
        self,
        prim_feat: torch.Tensor,      # [N,K,Dp]
        prim_mask: torch.Tensor,      # [N,K] bool
        prim_c_rel4: torch.Tensor,    # [N,K,4] = [dx,dy,dz,r]
        return_att: bool = False,
    ):
        N, K, Dp = prim_feat.shape
        assert Dp == self.d_prim
        assert prim_mask.shape == (N, K)
        assert prim_c_rel4.shape == (N, K, 4)
        assert prim_mask.dtype == torch.bool

        dtype = prim_feat.dtype

        # ---- rel encoding: (dx,dy,dz,r) -> (dir(3), logr(1)) ----
        rel = prim_c_rel4.float()
        dxyz = rel[..., :3]
        r = rel[..., 3:4].clamp_min(1e-6)

        if self.use_r_median_norm:
            r_med = _masked_quantile_1d(r.squeeze(-1), prim_mask, q=0.5).clamp_min(1e-6)
            r_hat = r / r_med.view(N, 1, 1)
        else:
            r_hat = r

        dir3 = dxyz / r
        logr = torch.log1p(r_hat)
        rel_n = torch.cat([dir3, logr], dim=-1)

        # ---- content logits ----
        k_feat = self.k_ln(self.k_feat_proj(prim_feat)).float()
        logits_content = self.score_proj(k_feat) / math.sqrt(self.dh)

        # ---- pos logits bias ----
        logits_pos = self.pos2logits(rel_n) / math.sqrt(self.dh)
        gamma = torch.sigmoid(self.pos_gate).float()

        logits = logits_content + gamma * logits_pos
        w = masked_softmax_safe(logits, prim_mask, dim=1)

        # ---- value aggregation ----
        v = self.v_proj(prim_feat).view(N, K, self.nh, self.dh)
        pooled = (w.to(v.dtype).unsqueeze(-1) * v).sum(dim=1).reshape(N, self.nh * self.dh)
        pooled = self.post(pooled.float()).to(dtype)

        if return_att:
            att = w.mean(dim=-1) * prim_mask.float()
            att = att / att.sum(dim=1, keepdim=True).clamp_min(1e-12)
            return pooled, att

        return pooled

class GeometryPrimitiveConstructor(nn.Module):
    """
    Only does geometry:
      1) local neighborhood packing
      2) projector -> local graph
      3) graph triangle construction

    No primitive feature is produced here.
    """

    def __init__(
        self,
        keep_prims: int = 8,
        detach_construct: bool = True,
    ):
        super().__init__()
        self.detach_construct = bool(detach_construct)

        self.projector = VirtualSphereProjector(
            min_valid_points=4,
            top_m=3,
            anti_pole_eps=1e-3,
            center_rmin_frac=0.06,
            bw_min=0.08,
            bw_weight=0.25,
            seam_tau=0.15,
            intra_k=4,
            cross_k=2,
            single_chart_min_dot=0.05,
            single_chart_max_seam_frac=0.15,
        )

        self.tri_builder = GraphTriangleConstructor(
            keep_k=keep_prims,
            cand_M=12,
            intra_k=4,
            cross_k=2,
            symmetric_graph=True,
            w_qdist=1.0,
            w_quality=1.0,
            w_span=1.0,
            w_compact=0.25,
            w_cross=0.35,
            max_faces_per_edge=2,
            enforce_connected=True,
            outward_bonus=0.0,
        )

    def forward(self, on_xyz, off_xyz, neigh_idx):
        """
        on_xyz:    [Ng,3]
        off_xyz:   [Nq,3]
        neigh_idx: [Nq,K] dense knn indices in global on_xyz, padded by -1

        returns:
          nb_xyz:   [Nq,K,3]
          nb_mask:  [Nq,K]
          tri_idx:  [Nq,T,3]  local vertex indices in 0..K-1
          tri_mask: [Nq,T]
          debug:    dict
        """
        Ng = int(on_xyz.shape[0])

        idx = neigh_idx.long()
        nb_mask = (idx >= 0) & (idx < Ng)
        nb_rel = gather_relative_xyz_zero(on_xyz, idx, off_xyz)
        nb_xyz = off_xyz[:, None, :].to(nb_rel.dtype) + nb_rel

        geom_nb_xyz = nb_xyz.detach() if self.detach_construct else nb_xyz
        geom_off_xyz = off_xyz.detach() if self.detach_construct else off_xyz

        c, graph_idx, graph_mask, dbg_proj = self.projector(
            geom_nb_xyz,
            nb_mask,
            geom_off_xyz,
        )

        tri_idx, tri_mask, _, tri_score, dbg_tri = self.tri_builder(
            nb_xyz=geom_nb_xyz,
            nb_feat=None,
            pos_q=geom_off_xyz,
            knn_idx=graph_idx,
            knn_mask=graph_mask,
        )

        debug = {}
        for k, v in dbg_proj.items():
            debug[f"proj_{k}"] = v
        for k, v in dbg_tri.items():
            debug[f"tri_{k}"] = v
        debug["proj_center"] = c
        debug["tri_score"] = tri_score

        return nb_xyz, nb_mask, tri_idx, tri_mask, debug

class TrianglePrimitiveFeatureEncoder(nn.Module):
    """
    For each triangle primitive:
      - gather 3 point features
      - gather 3 point positions relative to query
      - encode each point independently
      - mean pool over the 3 points

    Primitive center = centroid of the 3 points.
    """

    def __init__(
        self,
        point_feat_dim: int,
        prim_dim: int,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        self.point_feat_dim = int(point_feat_dim)
        self.prim_dim = int(prim_dim)
        hidden_dim = self.prim_dim if hidden_dim is None else int(hidden_dim)

        in_dim = self.point_feat_dim + 3

        self.point_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, self.prim_dim),
            nn.ReLU(True),
        )
        self.out_ln = nn.LayerNorm(self.prim_dim)

    def forward(
        self,
        nb_xyz: torch.Tensor,      # [N,K,3]
        nb_feat: torch.Tensor,     # [N,K,C]
        tri_idx: torch.Tensor,     # [N,T,3]
        tri_mask: torch.Tensor,    # [N,T]
        pos_q: torch.Tensor,       # [N,3]
    ):
        feat_dtype = nb_feat.dtype
        xyz_dtype = nb_xyz.dtype

        tri_xyz = gather_local_tri(nb_xyz, tri_idx)      # [N,T,3,3]
        tri_feat = gather_local_tri(nb_feat, tri_idx)    # [N,T,3,C]
        return self.forward_triangles(
            tri_xyz=tri_xyz,
            tri_feat=tri_feat,
            tri_mask=tri_mask,
            pos_q=pos_q,
            feat_dtype=feat_dtype,
            xyz_dtype=xyz_dtype,
        )

    def forward_triangles(
        self,
        tri_xyz: torch.Tensor,      # [N,T,3,3]
        tri_feat: torch.Tensor,     # [N,T,3,C]
        tri_mask: torch.Tensor,     # [N,T]
        pos_q: torch.Tensor,        # [N,3]
        feat_dtype=None,
        xyz_dtype=None,
    ):
        feat_dtype = tri_feat.dtype if feat_dtype is None else feat_dtype
        xyz_dtype = tri_xyz.dtype if xyz_dtype is None else xyz_dtype

        tri_rel_q = tri_xyz.float() - pos_q[:, None, None, :].float()  # [N,T,3,3]

        x = torch.cat([tri_feat.float(), tri_rel_q], dim=-1)            # [N,T,3,C+3]
        x = self.point_mlp(x)                                           # [N,T,3,Dp]

        prim_feat = x.mean(dim=2)                                       # [N,T,Dp]
        prim_feat = self.out_ln(prim_feat)
        prim_feat = prim_feat * tri_mask.unsqueeze(-1).float()

        prim_ctr_rel = tri_xyz.float().mean(dim=2) - pos_q[:, None, :].float()  # [N,T,3]
        prim_ctr_rel = prim_ctr_rel * tri_mask.unsqueeze(-1).float()

        return prim_feat.to(feat_dtype), prim_ctr_rel.to(xyz_dtype)
