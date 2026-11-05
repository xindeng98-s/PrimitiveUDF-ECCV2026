"""Per-query inference primitives: certified KNN, fused geometry, vertex reuse."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import math

import torch
import torch.nn.functional as F

from heads import primitive_geometry as geometry
from heads.primitive_distance import PrimitiveDistanceHead
from heads.primitive_layers import PrimitiveInterpolationLayers
from networks import primitive_spherical as spherical
from networks.primitive_fast_cuda import extension
from networks.primitive_point_cache import CertifiedPointCache


def _version(tensor):
    try:
        return tensor._version
    except RuntimeError:  # Inference-mode tensors have no version counter.
        return None


def pack_query_neighbors(support, query, neighbors):
    relative = geometry.gather_relative_xyz_zero(support, neighbors, query)
    return query[:, None].to(relative.dtype) + relative


class PrimitiveBankV3Interpolation(PrimitiveInterpolationLayers):
    def __init__(self, *args, voxel_size=.01, cache_points=48, cache_max_cells=2_000_000,
                 cache_tolerance=1e-6, geometry_backend='auto', chunk_size=50000,
                 cache_point_features=True, topology_voxel_size=0., topology_candidates=16,
                 topology_radius=1., **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry = geometry.GeometryPrimitiveConstructor(keep_prims=self.keep_prims)
        self.geometry_backend, self.chunk_size = geometry_backend, int(chunk_size)
        self.cache_point_features = bool(cache_point_features)
        if float(topology_voxel_size) != 0:
            raise ValueError('The final V3 release supports exact geometry only')
        if self.chunk_size <= 0:
            raise ValueError('primitive_query_chunk_size must be positive')
        self.point_cache = CertifiedPointCache(
            voxel_size, cache_points, self.bank_neighbor_k, cache_max_cells, cache_tolerance, geometry_backend)

    @torch.no_grad()
    def build_bank(self, on_xyz, support_offset):
        bank = dict(support_xyz=on_xyz, support_offset=support_offset.long(),
                    anchor_xyz=on_xyz, anchor_offset=support_offset.long())
        if not self.training:
            bank['point_cache'] = self.point_cache.build(on_xyz, support_offset)
        return bank

    def _feature_signature(self, feature, support):
        first = self.prim_encoder.point_mlp[0]
        return tuple((id(t), _version(t)) for t in (feature, support, first.weight, first.bias))

    def prepare_feature_cache(self, bank, feature):
        # Only an explicit no-grad build may retain detached neural results.
        # Training and ordinary grad-enabled eval keep feature/weight gradients.
        if (self.training or torch.is_grad_enabled() or not self.cache_point_features
                or bank['support_xyz'].dtype != torch.float32
                or (feature.is_cuda and torch.backends.cuda.matmul.allow_tf32)):
            return
        first = self.prim_encoder.point_mlp[0]
        with torch.autocast(device_type=feature.device.type, enabled=False):
            value = F.linear(torch.cat([feature.float(), bank['support_xyz'].float()], -1),
                             first.weight.float(), first.bias.float())
        bank['point_affine'] = (self._feature_signature(feature, bank['support_xyz']), value)

    def _direct_local(self, support, query, neighbors):
        _, _, indices, mask, _ = self.geometry(support, query, neighbors)
        return indices, mask

    def _direct(self, support, query, neighbors):
        indices, mask = self._direct_local(support, query, neighbors)
        return neighbors.gather(1, indices.reshape(len(query), -1)).reshape_as(indices), mask

    @torch.no_grad()
    def _construct_local(self, bank, query, neighbors):
        support = bank['support_xyz']
        geometry = self.geometry
        keep_prims = self.keep_prims
        def direct():
            _, _, indices, mask, _ = geometry(support, query, neighbors)
            return indices, mask
        ext = None if self.training else extension(query.device, self.geometry_backend)
        if ext is None or neighbors.shape[1] != 16 or keep_prims > 32:
            indices, mask = direct()
            return indices, mask, {'geometry_cuda_count': query.new_zeros((), dtype=torch.long),
                                   'geometry_fallback_count': query.new_tensor(len(query), dtype=torch.long)}
        # Preserve the original coordinate packing and chart. Complete KNN16
        # rows have the same graph-derived validity mask; all incomplete rows
        # are delegated to the original projector/constructor below.
        xyz = pack_query_neighbors(support, query, neighbors).float()
        point_mask = (neighbors >= 0) & (neighbors < len(support))
        projector = geometry.projector
        # The reference projector recomputes its constructor chart for the
        # whole batch if isolated valid points change graph-derived validity.
        # Preserve that rare batch-level behavior, rather than mixing charts.
        if bool((point_mask.sum(-1) < 2).any()):
            indices, mask = direct()
            return indices, mask, {'geometry_cuda_count': query.new_zeros((), dtype=torch.long),
                                   'geometry_fallback_count': query.new_tensor(len(query), dtype=torch.long)}
        u, v, _, _, chart = spherical._simple_chart(
            xyz, point_mask, query, min_valid_points=projector.min_valid_points,
            center_rmin_frac=projector.center_rmin_frac, thick_min_frac=projector.thick_min_frac,
            bw_weight=projector.bw_weight, anti_pole_eps=projector.anti_pole_eps)
        valid = point_mask
        uv = torch.stack([u, v], -1)
        radius = spherical._masked_median((uv - uv.mean(1, keepdim=True)).norm(dim=-1), valid).clamp_min(1e-6)
        area_eps = (1e-7 * radius.square()).clamp_min(geometry.tri_builder.eps)
        indices, mask, uncertain, _ = ext.select(
            xyz.float().contiguous(), query.float().contiguous(), uv.contiguous(),
            neighbors.new_empty(0, 16, 6), valid.new_empty(0, 16, 6),
            valid.contiguous(), area_eps.contiguous(), keep_prims, True)
        fallback = torch.where(uncertain | ~point_mask.all(-1))[0]
        if len(fallback):
            complete = point_mask[fallback].all(-1)
            stable_rows = fallback[complete]
            if len(stable_rows):
                # Keep the chart computed at the original batch size. Rerunning
                # PCA on a small fallback subset can change boundary decisions.
                xs, qs = xyz[stable_rows], query[stable_rows]
                us, vs, ms = u[stable_rows], v[stable_rows], valid[stable_rows]
                graph, graph_mask = projector._build_simple_knn(us, vs, ms)
                spherical._chart_cache_put(
                    spherical._chart_cache_key(xs, qs, graph, graph_mask), us, vs,
                    {'use_sphere': chart['use_sphere'][stable_rows]}, ms)
                local, local_mask, _, _, _ = geometry.tri_builder(xs, None, qs, graph, graph_mask)
                indices[stable_rows], mask[stable_rows] = local, local_mask
            incomplete_rows = fallback[~complete]
            if len(incomplete_rows):
                _, _, local, local_mask, _ = geometry(support, query[incomplete_rows], neighbors[incomplete_rows])
                indices[incomplete_rows], mask[incomplete_rows] = local, local_mask
        return indices, mask, {'geometry_cuda_count': query.new_tensor(len(query) - len(fallback), dtype=torch.long),
                               'geometry_fallback_count': query.new_tensor(len(fallback), dtype=torch.long)}

    @torch.no_grad()
    def _acquire(self, bank, query, offsets):
        n = len(query)
        if not n:
            return (torch.empty((0, self.bank_neighbor_k), dtype=torch.long, device=query.device),
                    torch.empty((0, self.keep_prims, 3), dtype=torch.long, device=query.device),
                    torch.empty((0, self.keep_prims), dtype=torch.bool, device=query.device), {})
        all_neighbors, all_indices, all_masks, debug = [], [], [], {}
        batch = geometry.offset2batch(offsets.long())
        if not self.training and 'point_cache' not in bank:
            bank['point_cache'] = self.point_cache.build(bank['support_xyz'], bank['support_offset'])
        for start in range(0, n, self.chunk_size):
            q = query[start:start + self.chunk_size]
            local_offsets = torch.bincount(batch[start:start + len(q)], minlength=len(offsets)).cumsum(0)
            if self.training:
                neighbors = geometry.knn_query_dense(
                    k=self.bank_neighbor_k, sup_xyz=bank['support_xyz'], sup_offset=bank['support_offset'],
                    qry_xyz=q, qry_offset=local_offsets)
                stats = {'direct_query_count': q.new_tensor(len(q), dtype=torch.long)}
            else:
                neighbors, stats = self.point_cache.query(bank, q, local_offsets)
            indices, mask, geometry_stats = self._construct_local(bank, q, neighbors)
            stats.update(geometry_stats)
            for key, value in stats.items():
                debug[key] = debug.get(key, 0) + value
            all_neighbors.append(neighbors)
            all_indices.append(indices)
            all_masks.append(mask)
        return torch.cat(all_neighbors), torch.cat(all_indices), torch.cat(all_masks), debug

    def _select_query_triangles(self, bank, off_xyz, off_offset):
        neighbors, local, mask, debug = self._acquire(bank, off_xyz, off_offset)
        indices = neighbors.gather(1, local.flatten(1)).reshape_as(local)
        return indices, mask, debug

    def _encode_neighbors(self, bank, feature, query, neighbors, local, mask):
        encoder = self.prim_encoder
        xyz = pack_query_neighbors(bank['support_xyz'], query, neighbors)
        nb_feature = geometry.gather_global_zero(feature, neighbors)
        cache = bank.get('point_affine')
        # Autocast changes the rounding of the split first affine. Keep the
        # original affine under AMP, while still encoding each neighbor once.
        autocast = torch.is_autocast_enabled() or torch.is_autocast_cpu_enabled()
        # Splitting the affine changes roundoff at rare ReLU zero crossings.
        # Value-only inference tolerates that noise; gradient-bearing calls
        # retain the original affine and still reuse all vertex encodings.
        use_cache = (not self.training and not torch.is_grad_enabled()
                     and cache is not None and not autocast
                     and not (query.is_cuda and torch.backends.cuda.matmul.allow_tf32)
                     and cache[0] == self._feature_signature(feature, bank['support_xyz'])
                     and bool((neighbors >= 0).all()))
        if use_cache:
            first = encoder.point_mlp[0]
            hidden = geometry.gather_global_zero(cache[1], neighbors)
            hidden = hidden - F.linear(query.float(), first.weight[:, -3:].float())[:, None]
            encoded = encoder.point_mlp[1:](hidden)
        else:
            encoded = encoder.point_mlp(torch.cat([nb_feature.float(), xyz.float() - query[:, None].float()], -1))
        primitive = encoder.out_ln(geometry.gather_local_tri(encoded, local).mean(2))
        center = geometry.gather_local_tri(xyz, local).float().mean(2) - query[:, None].float()
        primitive = primitive * mask[..., None].float()
        center = center * mask[..., None].float()
        return primitive.to(feature.dtype), center.to(bank['support_xyz'].dtype), use_cache

    def predict(self, bank, feature, off_xyz, off_offset):
        neighbors, local, mask, debug = self._acquire(bank, off_xyz, off_offset)
        primitive, center, cached = self._encode_neighbors(bank, feature, off_xyz, neighbors, local, mask)
        keep, _ = self._apply_primitive_dropout(mask)
        primitive = primitive * keep[..., None].to(primitive.dtype)
        center = center * keep[..., None].to(center.dtype)
        radius = center.float().norm(dim=-1, keepdim=True).clamp_min(1e-8)
        pooled = self.prim_pool(prim_feat=primitive, prim_mask=keep,
                                prim_c_rel4=torch.cat([center.float(), radius], -1))
        debug['point_affine_cached'] = off_xyz.new_tensor(int(cached), dtype=torch.long)
        return self.out_proj(pooled), debug


class PrimitiveBankV3Head(PrimitiveDistanceHead):
    def __init__(self, *, init_grid=.01, primitive_point_cache_size=48,
                 primitive_point_cache_max_cells=2_000_000, primitive_point_cache_tolerance=1e-6,
                 primitive_geometry_backend='auto', primitive_query_chunk_size=50000,
                 primitive_cache_point_features=True, primitive_topology_voxel_size=0.,
                 primitive_topology_candidates=16, primitive_topology_radius=1., **kwargs):
        self._query_settings = dict(voxel_size=init_grid, cache_points=primitive_point_cache_size,
                                   cache_max_cells=primitive_point_cache_max_cells,
                                   cache_tolerance=primitive_point_cache_tolerance,
                                   geometry_backend=primitive_geometry_backend, chunk_size=primitive_query_chunk_size,
                                   cache_point_features=primitive_cache_point_features,
                                   topology_voxel_size=primitive_topology_voxel_size,
                                   topology_candidates=primitive_topology_candidates,
                                   topology_radius=primitive_topology_radius)
        super().__init__(**kwargs)

    def _make_interpolator(self, **kwargs):
        return PrimitiveBankV3Interpolation(**self._query_settings, **kwargs)

    def build_bank(self, points):
        state = super().build_bank(points)
        self.interpolators.prepare_feature_cache(state[1], points[1])
        return state


def head_options(opt):
    return dict(in_channels=48, hidden_dim=32, out_dim=1, sdf_flag=bool(opt['is_sdf']),
                init_grid=float(opt['init_grid']),
                primitive_point_cache_size=int(opt.get('primitive_point_cache_size', 48)),
                primitive_point_cache_max_cells=int(opt.get('primitive_point_cache_max_cells', 2_000_000)),
                primitive_point_cache_tolerance=float(opt.get('primitive_point_cache_tolerance', 1e-6)),
                primitive_geometry_backend=str(opt.get('primitive_geometry_backend', 'auto')),
                primitive_query_chunk_size=int(opt.get('primitive_query_chunk_size', 50000)),
                primitive_cache_point_features=bool(opt.get('primitive_cache_point_features', True)),
                primitive_topology_voxel_size=float(opt.get('primitive_topology_voxel_size', 0.)),
                primitive_topology_candidates=int(opt.get('primitive_topology_candidates', 16)),
                primitive_topology_radius=float(opt.get('primitive_topology_radius', 1.)),
                primitive_dropout_p=float(opt['primitive_dropout_p']),
                primitive_dropout_eval=bool(opt['primitive_dropout_eval']),
                primitive_bank_neighbor_k=int(opt['primitive_bank_neighbor_k']),
                primitive_bank_keep_prims=int(opt['primitive_bank_keep_prims']))


def build_head(opt: Mapping[str, Any]) -> PrimitiveBankV3Head:
    return PrimitiveBankV3Head(**head_options(opt))
