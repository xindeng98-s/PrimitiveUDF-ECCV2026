"""
Point Transformer V2 backbone

PyG KNN + KNN interpolation backend

Notes:
- Uses torch_geometric.nn.knn (from torch-cluster)
- Uses torch_geometric.nn.unpool.knn_interpolate
- Keep self-neighbor (exclude_self=False)
"""

from copy import deepcopy
import math
import torch
import torch.nn as nn
import numpy as np
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn.pool import voxel_grid
from torch_scatter import segment_csr

import einops
from timm.models.layers import DropPath

# ---- PyG / torch-cluster ----
try:
    # PyG exposes knn (implemented in torch-cluster)
    from torch_geometric.nn import knn
except Exception:
    # fallback if your PyG build doesn't re-export it
    from torch_cluster import knn  # type: ignore

from torch_geometric.nn.unpool import knn_interpolate


# -----------------------------------------------------------------------------
# Small utils
# -----------------------------------------------------------------------------
def get_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def offset2batch(offset: torch.Tensor) -> torch.Tensor:
    """
    offset: [B] cumulative sum, offset[-1] == N
    returns: [N] batch id for each point
    """
    offset = offset.to(torch.long)
    counts = torch.diff(torch.cat([offset.new_zeros(1), offset], dim=0), dim=0)
    return torch.repeat_interleave(
        torch.arange(offset.numel(), device=offset.device, dtype=torch.long),
        counts,
    )


def batch2offset(batch: torch.Tensor) -> torch.Tensor:
    return torch.cumsum(batch.bincount(), dim=0).long()


# -----------------------------------------------------------------------------
# PyG KNN helpers
# -----------------------------------------------------------------------------
def _dense_knn_from_edges(
    y_idx: torch.Tensor,
    x_idx: torch.Tensor,
    M: int,
    k: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Pack sparse knn edges (y_idx,x_idx) into dense [M,k] with -1 padding.
    y_idx: query indices in [0, M)
    x_idx: support indices in [0, N_support)
    """
    y_idx = y_idx.to(torch.long)
    x_idx = x_idx.to(torch.long)

    perm = torch.argsort(y_idx)
    y = y_idx[perm]
    x = x_idx[perm]

    cnt = torch.bincount(y, minlength=M)  # [M]
    start = torch.cumsum(cnt, dim=0) - cnt
    local = torch.arange(y.numel(), device=device) - start[y]  # [E]

    out = torch.full((M, k), -1, device=device, dtype=torch.long)
    m = local < k
    out[y[m], local[m]] = x[m]
    return out


def knn_query_pyg(
    k: int,
    coord_x: torch.Tensor,
    offset_x: torch.Tensor,
    coord_y: torch.Tensor = None,
    offset_y: torch.Tensor = None,
    exclude_self: bool = False,
) -> torch.Tensor:
    """
    Dense knn query using PyG knn:
      support: coord_x + offset_x
      query:   coord_y + offset_y  (if None -> self knn)
    returns:
      reference_index: [Ny, k] global indices into coord_x, padded with -1
    """
    if coord_y is None:
        coord_y = coord_x
        offset_y = offset_x
    assert offset_y is not None

    device = coord_x.device
    batch_x = offset2batch(offset_x)
    batch_y = offset2batch(offset_y)

    # knn returns assign_index = [2, E], where row0 is y (query), row1 is x (support)
    assign_index = knn(
        x=coord_x,
        y=coord_y,
        k=(k + 1) if exclude_self else k,
        batch_x=batch_x,
        batch_y=batch_y,
    )
    y_idx, x_idx = assign_index[0], assign_index[1]

    if exclude_self:
        # only meaningful for self knn (x==y)
        m = x_idx != y_idx
        y_idx, x_idx = y_idx[m], x_idx[m]

    Ny = int(coord_y.shape[0])
    return _dense_knn_from_edges(y_idx, x_idx, Ny, k, device)


def grouping_dense(
    idx: torch.Tensor,      # [Nq, K] with -1
    feat: torch.Tensor,     # [Ns, C]
    coord: torch.Tensor,    # [Ns, 3]
    with_xyz: bool = False,
    query_coord: torch.Tensor = None,   # [Nq, 3], default coord[:Nq]
) -> torch.Tensor:
    """
    Dense masked grouping:
      with_xyz=True  -> [Nq,K,3+C], xyz = neighbor - query
      with_xyz=False -> [Nq,K,C]
      idx=-1 or out-of-range gathers all-zero features and relative vectors.
    """
    Ns = int(coord.shape[0])
    Nq, K = idx.shape
    idx = idx.to(torch.long)

    if query_coord is None:
        query_coord = coord[:Nq]

    mask = (idx >= 0) & (idx < Ns)
    idx_safe = idx.clamp(0, max(0, Ns - 1))

    gfeat = feat[idx_safe] * mask.unsqueeze(-1).to(feat.dtype)  # [Nq,K,C]

    if not with_xyz:
        return gfeat

    neigh = coord[idx_safe]  # [Nq,K,3]
    rel = (neigh - query_coord.unsqueeze(1)) * mask.unsqueeze(-1).to(coord.dtype)
    return torch.cat([rel, gfeat], dim=-1)


# -----------------------------------------------------------------------------
# Core modules
# -----------------------------------------------------------------------------
class PointBatchNorm(nn.Module):
    """
    Batch Normalization for Point Clouds data in shape of [B*N, C], [B*N, L, C]
    """
    def __init__(self, embed_channels):
        super().__init__()
        self.norm = nn.BatchNorm1d(embed_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:
            return (
                self.norm(input.transpose(1, 2).contiguous())
                .transpose(1, 2)
                .contiguous()
            )
        elif input.dim() == 2:
            return self.norm(input)
        else:
            raise NotImplementedError


class GroupedVectorAttention(nn.Module):
    def __init__(
        self,
        embed_channels,
        groups,
        attn_drop_rate=0.0,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
    ):
        super(GroupedVectorAttention, self).__init__()
        self.embed_channels = embed_channels
        self.groups = groups
        assert embed_channels % groups == 0
        self.attn_drop_rate = attn_drop_rate
        self.qkv_bias = qkv_bias
        self.pe_multiplier = pe_multiplier
        self.pe_bias = pe_bias

        self.linear_q = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True),
        )
        self.linear_k = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True),
        )
        self.linear_v = nn.Linear(embed_channels, embed_channels, bias=qkv_bias)

        if self.pe_multiplier:
            self.linear_p_multiplier = nn.Sequential(
                nn.Linear(3, embed_channels),
                PointBatchNorm(embed_channels),
                nn.ReLU(inplace=True),
                nn.Linear(embed_channels, embed_channels),
            )
        if self.pe_bias:
            self.linear_p_bias = nn.Sequential(
                nn.Linear(3, embed_channels),
                PointBatchNorm(embed_channels),
                nn.ReLU(inplace=True),
                nn.Linear(embed_channels, embed_channels),
            )

        self.weight_encoding = nn.Sequential(
            nn.Linear(embed_channels, groups),
            PointBatchNorm(groups),
            nn.ReLU(inplace=True),
            nn.Linear(groups, groups),
        )
        self.softmax = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attn_drop_rate)

    def forward(self, feat, coord, reference_index):
        # feat: [N,C], coord: [N,3], reference_index: [N,K]
        query = self.linear_q(feat)
        key0 = self.linear_k(feat)
        val0 = self.linear_v(feat)

        # masked dense grouping
        key_g = grouping_dense(reference_index, key0, coord=coord, with_xyz=True, query_coord=coord)
        val_g = grouping_dense(reference_index, val0, coord=coord, with_xyz=False, query_coord=coord)

        pos, key = key_g[:, :, 0:3], key_g[:, :, 3:]
        relation_qk = key - query.unsqueeze(1)

        if self.pe_multiplier:
            pem = self.linear_p_multiplier(pos)
            relation_qk = relation_qk * pem
        if self.pe_bias:
            peb = self.linear_p_bias(pos)
            relation_qk = relation_qk + peb
            val_g = val_g + peb

        weight = self.weight_encoding(relation_qk)
        weight = self.attn_drop(self.softmax(weight))

        # keep self neighbor; invalid neighbors (-1) are masked
        mask = (reference_index >= 0).to(weight.dtype)  # [N,K]
        weight = torch.einsum("n s g, n s -> n s g", weight, mask)

        val_g = einops.rearrange(val_g, "n ns (g i) -> n ns g i", g=self.groups)
        feat_out = torch.einsum("n s g i, n s g -> n g i", val_g, weight)
        feat_out = einops.rearrange(feat_out, "n g i -> n (g i)")
        return feat_out


class Block(nn.Module):
    def __init__(
        self,
        embed_channels,
        groups,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        enable_checkpoint=False,
    ):
        super(Block, self).__init__()
        self.attn = GroupedVectorAttention(
            embed_channels=embed_channels,
            groups=groups,
            qkv_bias=qkv_bias,
            attn_drop_rate=attn_drop_rate,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
        )
        self.fc1 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.fc3 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.norm1 = PointBatchNorm(embed_channels)
        self.norm2 = PointBatchNorm(embed_channels)
        self.norm3 = PointBatchNorm(embed_channels)
        self.act = nn.ReLU(inplace=True)
        self.enable_checkpoint = enable_checkpoint
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, points, reference_index):
        coord, feat, offset = points
        identity = feat
        feat = self.act(self.norm1(self.fc1(feat)))
        feat = (
            self.attn(feat, coord, reference_index)
            if not self.enable_checkpoint
            else checkpoint(self.attn, feat, coord, reference_index)
        )
        feat = self.act(self.norm2(feat))
        feat = self.norm3(self.fc3(feat))
        feat = identity + self.drop_path(feat)
        feat = self.act(feat)
        return [coord, feat, offset]


class BlockSequence(nn.Module):
    def __init__(
        self,
        depth,
        embed_channels,
        groups,
        neighbours=16,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        enable_checkpoint=False,
    ):
        super(BlockSequence, self).__init__()

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        elif isinstance(drop_path_rate, float):
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]
        else:
            drop_path_rates = [0.0 for _ in range(depth)]

        self.neighbours = neighbours
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                Block(
                    embed_channels=embed_channels,
                    groups=groups,
                    qkv_bias=qkv_bias,
                    pe_multiplier=pe_multiplier,
                    pe_bias=pe_bias,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=drop_path_rates[i],
                    enable_checkpoint=enable_checkpoint,
                )
            )

    def forward(self, points):
        coord, feat, offset = points

        # self knn, keep self neighbor (exclude_self=False)
        reference_index = knn_query_pyg(
            k=self.neighbours,
            coord_x=coord,
            offset_x=offset,
            coord_y=None,
            offset_y=None,
            exclude_self=False,
        )

        for block in self.blocks:
            points = block(points, reference_index)
        return points


class GridPool(nn.Module):
    """
    Partition-based Pooling (Grid Pooling)
    """
    def __init__(self, in_channels, out_channels, grid_size, bias=False):
        super(GridPool, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size

        self.fc = nn.Linear(in_channels, out_channels, bias=bias)
        self.norm = PointBatchNorm(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, points, start=None):
        coord, feat, offset = points
        batch = offset2batch(offset)
        feat = self.act(self.norm(self.fc(feat)))
        start = (
            segment_csr(
                coord,
                torch.cat([batch.new_zeros(1), torch.cumsum(batch.bincount(), dim=0)]),
                reduce="min",
            )
            if start is None
            else start
        )

        cluster = voxel_grid(pos=coord - start[batch], size=self.grid_size, batch=batch, start=0)
        unique, cluster, counts = torch.unique(cluster, sorted=True, return_inverse=True, return_counts=True)
        _, sorted_cluster_indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        coord = segment_csr(coord[sorted_cluster_indices], idx_ptr, reduce="mean")
        feat = segment_csr(feat[sorted_cluster_indices], idx_ptr, reduce="max")
        batch = batch[idx_ptr[:-1]]
        offset = batch2offset(batch)
        return [coord, feat, offset], cluster


def grid_sample(points, grid_size):
    coord, feat, offset = points
    batch = offset2batch(offset)
    start = segment_csr(
        coord,
        torch.cat([batch.new_zeros(1), torch.cumsum(batch.bincount(), dim=0)]),
        reduce="min",
    )
    cluster = voxel_grid(pos=coord - start[batch], size=grid_size, batch=batch, start=0)
    unique, cluster, counts = torch.unique(cluster, sorted=True, return_inverse=True, return_counts=True)
    _, sorted_cluster_indices = torch.sort(cluster)
    idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])

    idx_select_1 = idx_ptr[0:-1] + torch.randint(0, counts.max(), counts.shape, device=counts.device) % counts
    idx_select_2 = idx_ptr[0:-1] + (torch.randint(0, counts.max(), counts.shape, device=counts.device) + 1) % counts
    idx_select = torch.cat([idx_select_1, idx_select_2], dim=0)
    idx_select = torch.unique(idx_select)

    idx_unique = sorted_cluster_indices[idx_select]
    coord = coord[idx_unique]
    feat = feat[idx_unique]
    batch = batch[idx_unique]
    offset = batch2offset(batch)
    return (coord, feat, offset)


class UnpoolWithSkip(nn.Module):
    """
    Map Unpooling with skip connection
    """
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        bias=True,
        skip=True,
        backend="map",
    ):
        super(UnpoolWithSkip, self).__init__()
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels
        self.skip = skip
        self.backend = backend
        assert self.backend in ["map", "interp"]

        self.proj = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=bias),
            PointBatchNorm(out_channels),
            nn.ReLU(inplace=True),
        )
        self.proj_skip = nn.Sequential(
            nn.Linear(skip_channels, out_channels, bias=bias),
            PointBatchNorm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, points, skip_points, cluster=None):
        coord, feat, offset = points
        skip_coord, skip_feat, skip_offset = skip_points

        if self.backend == "map" and cluster is not None:
            feat = self.proj(feat)[cluster]
        else:
            # PyG knn_interpolate: interpolate features from (coord, feat) -> skip_coord
            batch_x = offset2batch(offset)
            batch_y = offset2batch(skip_offset)
            feat = knn_interpolate(
                x=self.proj(feat),   # [Ns,C]
                pos_x=coord,         # [Ns,3]
                pos_y=skip_coord,    # [Nt,3]
                batch_x=batch_x,
                batch_y=batch_y,
                k=3,
            )

        if self.skip:
            feat = feat + self.proj_skip(skip_feat)
        return [skip_coord, feat, skip_offset]


class Encoder(nn.Module):
    def __init__(
        self,
        depth,
        in_channels,
        embed_channels,
        groups,
        grid_size=None,
        neighbours=16,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=None,
        drop_path_rate=None,
        enable_checkpoint=False,
    ):
        super(Encoder, self).__init__()
        self.down = GridPool(in_channels=in_channels, out_channels=embed_channels, grid_size=grid_size)
        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate if attn_drop_rate is not None else 0.0,
            drop_path_rate=drop_path_rate if drop_path_rate is not None else 0.0,
            enable_checkpoint=enable_checkpoint,
        )

    def forward(self, points, start=None):
        points, cluster = self.down(points, start=start)
        return self.blocks(points), cluster


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        embed_channels,
        groups,
        depth,
        neighbours=16,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=None,
        drop_path_rate=None,
        enable_checkpoint=False,
        unpool_backend="map",
    ):
        super(Decoder, self).__init__()
        self.up = UnpoolWithSkip(
            in_channels=in_channels,
            out_channels=embed_channels,
            skip_channels=skip_channels,
            backend=unpool_backend,
        )
        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate if attn_drop_rate is not None else 0.0,
            drop_path_rate=drop_path_rate if drop_path_rate is not None else 0.0,
            enable_checkpoint=enable_checkpoint,
        )

    def forward(self, points, skip_points, cluster):
        points = self.up(points, skip_points, cluster)
        return self.blocks(points)


class GVAPatchEmbed(nn.Module):
    def __init__(
        self,
        depth,
        in_channels,
        embed_channels,
        groups,
        neighbours=16,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        enable_checkpoint=False,
    ):
        super(GVAPatchEmbed, self).__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels
        self.proj = nn.Sequential(
            nn.Linear(in_channels, embed_channels, bias=False),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            enable_checkpoint=enable_checkpoint,
        )

    def forward(self, points):
        coord, feat, offset = points
        feat = self.proj(feat)
        return self.blocks([coord, feat, offset])


class PointTransformerV2(nn.Module):
    def __init__(
        self,
        in_channels,
        patch_embed_depth=1,
        patch_embed_channels=48,
        patch_embed_groups=6,
        patch_embed_neighbours=8,
        enc_depths=(2, 2, 6, 2),
        enc_channels=(96, 192, 384, 512),
        enc_groups=(12, 24, 48, 64),
        enc_neighbours=(16, 16, 16, 16),
        dec_depths=(1, 1, 1, 1),
        dec_channels=(48, 96, 192, 384),
        dec_groups=(6, 12, 24, 48),
        dec_neighbours=(16, 16, 16, 16),
        grid_sizes=(0.03, 0.06, 0.12, 0.24),
        attn_qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0,
        enable_checkpoint=False,
        unpool_backend="map",
        get_2nd=False,
    ):
        super(PointTransformerV2, self).__init__()
        self.in_channels = in_channels
        self.num_stages = len(enc_depths)
        assert self.num_stages == len(dec_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(dec_channels)
        assert self.num_stages == len(enc_groups)
        assert self.num_stages == len(dec_groups)
        assert self.num_stages == len(enc_neighbours)
        assert self.num_stages == len(dec_neighbours)
        assert self.num_stages == len(grid_sizes)

        self.patch_embed = GVAPatchEmbed(
            in_channels=in_channels,
            embed_channels=patch_embed_channels,
            groups=patch_embed_groups,
            depth=patch_embed_depth,
            neighbours=patch_embed_neighbours,
            qkv_bias=attn_qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate,
            enable_checkpoint=enable_checkpoint,
        )

        enc_dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(enc_depths))]
        dec_dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(dec_depths))]

        enc_channels = [patch_embed_channels] + list(enc_channels)
        dec_channels = list(dec_channels) + [enc_channels[-1]]

        self.enc_stages = nn.ModuleList()
        self.dec_stages = nn.ModuleList()
        for i in range(self.num_stages):
            enc = Encoder(
                depth=enc_depths[i],
                in_channels=enc_channels[i],
                embed_channels=enc_channels[i + 1],
                groups=enc_groups[i],
                grid_size=grid_sizes[i],
                neighbours=enc_neighbours[i],
                qkv_bias=attn_qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=enc_dp_rates[sum(enc_depths[:i]) : sum(enc_depths[: i + 1])],
                enable_checkpoint=enable_checkpoint,
            )
            self.enc_stages.append(enc)

            dec = Decoder(
                depth=dec_depths[i],
                in_channels=dec_channels[i + 1],
                skip_channels=enc_channels[i],
                embed_channels=dec_channels[i],
                groups=dec_groups[i],
                neighbours=dec_neighbours[i],
                qkv_bias=attn_qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dec_dp_rates[sum(dec_depths[:i]) : sum(dec_depths[: i + 1])],
                enable_checkpoint=enable_checkpoint,
                unpool_backend=unpool_backend,
            )
            self.dec_stages.append(dec)

        self.get_2nd = get_2nd

    def forward(self, data_dict):
        coord = data_dict["coord"]
        feat = data_dict["feat"]
        offset = data_dict["offset"].to(torch.long)

        # input feature
        if feat is not None:
            feat = torch.cat([coord, feat], dim=-1)
        else:
            feat = coord
        points = [coord, feat, offset]

        points = self.patch_embed(points)

        skips = [[points]]
        for i in range(self.num_stages):
            points, cluster = self.enc_stages[i](points)
            skips[-1].append(cluster)
            skips.append([points])

        points = skips.pop(-1)[0]
        for i in reversed(range(self.num_stages)):
            skip_points, cluster = skips.pop(-1)
            if i == 0 and self.get_2nd:
                return points
            points = self.dec_stages[i](points, skip_points, cluster)

        return points
