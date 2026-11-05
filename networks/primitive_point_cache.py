"""Lazy voxel point supersets with a certified, exact-KNN fallback.

Caches belong to one build/predict state. Bounds cap the dense voxel lookup;
queries outside it (or in oversized scenes) still use the original KNN.
"""
import math

import torch

from heads import primitive_geometry as geometry
from networks.primitive_fast_cuda import extension


class CertifiedPointCache:
    def __init__(self, voxel_size=.01, points=48, k=16, max_cells=2_000_000,
                 tolerance=1e-6, backend='auto'):
        self.voxel_size, self.points, self.k = float(voxel_size), int(points), int(k)
        self.max_cells, self.tolerance, self.backend = int(max_cells), float(tolerance), backend
        if not math.isfinite(self.voxel_size) or self.voxel_size <= 0:
            raise ValueError('init_grid must be finite and positive')
        if self.k <= 0 or self.points < self.k or self.points > 96:
            raise ValueError('point cache requires 0 < K <= M <= 96')
        if self.max_cells < 0 or not math.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError('cache size and tolerance must be nonnegative')
        if backend not in {'auto', 'torch', 'cuda'}:
            raise ValueError('primitive_geometry_backend must be auto, cuda or torch')

    @torch.no_grad()
    def build(self, support, offsets):
        if not bool(torch.isfinite(support).all()):
            raise ValueError('Point-cache support coordinates must be finite')
        device = support.device
        start, used = 0, 0
        lower, sizes, bases, enabled = [], [], [0], []
        for end in offsets.tolist():
            if end > start:
                cells = torch.floor(support[start:end].float() / self.voxel_size).long()
                lo, hi = cells.amin(0) - 4, cells.amax(0) + 4
                shape = (hi - lo + 1).tolist()
                volume = math.prod(shape)
            else:
                lo, shape, volume = torch.zeros(3, device=device, dtype=torch.long), [1, 1, 1], 0
            active = 0 < volume <= self.max_cells - used
            lower.append(lo)
            sizes.append(shape if active else [1, 1, 1])
            used += volume if active else 0
            bases.append(used)
            enabled.append(active)
            start = end
        cache = dict(
            lower=torch.stack(lower), sizes=torch.tensor(sizes, device=device, dtype=torch.long),
            bases=torch.tensor(bases, device=device, dtype=torch.long),
            enabled=torch.tensor(enabled, device=device, dtype=torch.bool),
            lookup=torch.full((used,), -1, device=device, dtype=torch.long),
            indices=torch.empty((0, self.points + 1), device=device, dtype=torch.long),
            centers=torch.empty((0, 3), device=device, dtype=torch.float32),
        )
        if device.type == 'cuda' and self.backend != 'torch' and used:
            batch = geometry.offset2batch(offsets.long())
            active = torch.where(cache['enabled'][batch])[0]
            cells = torch.floor(support[active].float() / self.voxel_size).long() - cache['lower'][batch[active]]
            shape = cache['sizes'][batch[active]]
            code = ((cells[:, 0] * shape[:, 1] + cells[:, 1]) * shape[:, 2]
                    + cells[:, 2] + cache['bases'][batch[active]])
            order = code.argsort()
            unique, count = torch.unique_consecutive(code[order], return_counts=True)
            end = count.cumsum(0)
            ranges = torch.zeros((used, 2), device=device, dtype=torch.long)
            ranges[unique] = torch.stack([end - count, end], -1)
            cache.update(support_ids=active[order].contiguous(), ranges=ranges)
        return cache

    @torch.no_grad()
    def _rows(self, bank, query, batch):
        cache = bank['point_cache']
        finite = torch.isfinite(query).all(-1)
        cells = torch.floor(query.float() / self.voxel_size).long()
        relative = cells - cache['lower'][batch]
        sizes = cache['sizes'][batch]
        inside = finite & cache['enabled'][batch] & ((relative >= 0) & (relative < sizes)).all(-1)
        code = ((relative[:, 0] * sizes[:, 1] + relative[:, 1]) * sizes[:, 2]
                + relative[:, 2] + cache['bases'][batch])
        rows = torch.full_like(batch, -1)
        active = torch.where(inside)[0]
        if not len(active):
            return rows, 0
        rows[active] = cache['lookup'][code[active]]
        missing = active[rows[active] < 0]
        if not len(missing):
            return rows, 0
        unique = torch.unique(code[missing], sorted=True)
        cell_batch = torch.bucketize(unique, cache['bases'][1:], right=True)
        local = unique - cache['bases'][cell_batch]
        shape = cache['sizes'][cell_batch]
        cx = local // (shape[:, 1] * shape[:, 2])
        cy = (local // shape[:, 2]) % shape[:, 1]
        cz = local % shape[:, 2]
        cells = torch.stack([cx, cy, cz], -1) + cache['lower'][cell_batch]
        centers = ((cells.double() + .5) * self.voxel_size).float()
        offsets = torch.bincount(cell_batch, minlength=len(bank['support_offset'])).cumsum(0)
        ext = extension(query.device, self.backend)
        if ext is not None and 'ranges' in cache:
            indices, certified = ext.cache_centers(
                bank['support_xyz'].float().contiguous(), centers.contiguous(), unique.contiguous(),
                cell_batch.contiguous(), cache['sizes'], cache['bases'], cache['ranges'],
                cache['support_ids'], bank['support_offset'].contiguous(), self.points + 1, self.voxel_size)
            fallback = torch.where(~certified)[0]
            if len(fallback):
                fallback_offsets = torch.bincount(cell_batch[fallback], minlength=len(offsets)).cumsum(0)
                indices[fallback] = geometry.knn_query_dense(
                    k=self.points + 1, sup_xyz=bank['support_xyz'], sup_offset=bank['support_offset'],
                    qry_xyz=centers[fallback], qry_offset=fallback_offsets)
        else:
            indices = geometry.knn_query_dense(
                k=self.points + 1, sup_xyz=bank['support_xyz'], sup_offset=bank['support_offset'],
                qry_xyz=centers, qry_offset=offsets)
        # The certificate needs an actual (M+1)-nearest bound, independently of
        # the edge ordering exposed by the installed KNN backend.
        if ext is None and len(bank['support_xyz']):
            dist = (bank['support_xyz'][indices.clamp_min(0)].float() - centers[:, None]).square().sum(-1)
            order = dist.masked_fill(indices < 0, float('inf')).argsort(-1)
            indices = indices.gather(1, order)
        first = len(cache['centers'])
        cache['indices'] = torch.cat([cache['indices'], indices])
        cache['centers'] = torch.cat([cache['centers'], centers])
        cache['lookup'][unique] = torch.arange(first, first + len(unique), device=query.device)
        rows[missing] = cache['lookup'][code[missing]]
        return rows, len(unique)

    def _torch_select(self, bank, query, rows):
        cache = bank['point_cache']
        neighbors = torch.full((len(query), self.k), -1, device=query.device, dtype=torch.long)
        accepted = torch.zeros(len(query), device=query.device, dtype=torch.bool)
        active = torch.where(rows >= 0)[0]
        if not len(active):
            return neighbors, accepted
        ids = cache['indices'][rows[active]]
        support = bank['support_xyz'].float()
        xyz = support[ids.clamp_min(0)]
        distance = (xyz[:, :-1] - query[active, None].float()).square().sum(-1)
        distance = distance.masked_fill(ids[:, :-1] < 0, float('inf'))
        count = min(self.k + 1, self.points)
        values, order = distance.topk(count, largest=False, sorted=True)
        neighbors[active] = ids[:, :-1].gather(1, order[:, :self.k])
        complete = torch.isfinite(values[:, self.k - 1])
        centers = cache['centers'][rows[active]]
        bound = (xyz[:, -1] - centers).norm(dim=-1).masked_fill(ids[:, -1] < 0, float('inf'))
        radius = (query[active].float() - centers).norm(dim=-1)
        t = values[:, self.k - 1].sqrt()
        scale = centers.abs().amax(-1) + radius + t
        guard = self.tolerance + 32 * torch.finfo(torch.float32).eps * scale
        # Delegate ties / near ties, including inside the selected set, to the
        # original KNN backend so neighborhood ordering remains compatible.
        difference = values[:, 1:] - values[:, :-1]
        ties = (difference <= 8 * torch.finfo(torch.float32).eps * values[:, 1:].clamp_min(1e-8)).any(-1)
        accepted[active] = complete & ~ties & (t + radius + guard < bound)
        return neighbors, accepted

    @torch.no_grad()
    def query(self, bank, query, offsets):
        batch = geometry.offset2batch(offsets.long())
        rows, created = self._rows(bank, query, batch)
        ext = extension(query.device, self.backend)
        if ext is not None and self.k == 16:
            neighbors, accepted = ext.certified_knn(
                bank['support_xyz'].float().contiguous(), query.float().contiguous(), rows.contiguous(),
                bank['point_cache']['indices'].contiguous(), bank['point_cache']['centers'].contiguous(), self.tolerance)
        else:
            neighbors, accepted = self._torch_select(bank, query, rows)
        fallback = torch.where(~accepted)[0]
        if len(fallback):
            fallback_offsets = torch.bincount(batch[fallback], minlength=len(offsets)).cumsum(0)
            neighbors[fallback] = geometry.knn_query_dense(
                k=self.k, sup_xyz=bank['support_xyz'], sup_offset=bank['support_offset'],
                qry_xyz=query[fallback], qry_offset=fallback_offsets)
        return neighbors, {
            'point_cache_created': query.new_tensor(created, dtype=torch.long),
            'knn_certified_count': accepted.sum(),
            'knn_fallback_count': (~accepted).sum(),
        }
