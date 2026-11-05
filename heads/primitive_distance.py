"""Shared distance regression layers for V3."""
import torch
from torch import nn

class PrimitiveDistanceHead(nn.Module):
    """Distance head that shares one minimal primitive bank across query sets."""

    def __init__(
        self,
        *,
        in_channels: int = 48,
        hidden_dim: int = 32,
        out_dim: int = 1,
        sdf_flag: bool = False,
        primitive_dropout_p: float = 0.0,
        primitive_dropout_eval: bool = False,
        primitive_bank_neighbor_k: int = 16,
        primitive_bank_query_anchors: int = 2,
        primitive_bank_prims_per_anchor: int = 10,
        primitive_bank_keep_prims: int = 10,
    ) -> None:
        super().__init__()
        self.sdf = bool(sdf_flag)
        self.interpolators = self._make_interpolator(
            d_out=in_channels,
            prim_dim=48,
            keep_prims=primitive_bank_keep_prims,
            bank_neighbor_k=primitive_bank_neighbor_k,
            query_anchor_k=primitive_bank_query_anchors,
            bank_prims_per_anchor=primitive_bank_prims_per_anchor,
            detach_construct=True,
            primitive_dropout_p=primitive_dropout_p,
            primitive_dropout_eval=primitive_dropout_eval,
        )
        self.blocks = nn.Sequential(
            nn.Linear(in_channels + 3, 256),
            nn.ReLU(True),
            nn.Linear(256, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
        )
        self.outs = nn.Linear(hidden_dim, out_dim)

    def _make_interpolator(self, **kwargs):
        raise NotImplementedError("V3 supplies the interpolator")

    def build_bank(self, points):
        grid_center, _, grid_offset = points
        return points, self.interpolators.build_bank(grid_center, grid_offset)

    def predict(self, bank, query_xyz, query_offsets):
        points, topology = bank
        _, grid_feature, _ = points
        interpolated, debug = self.interpolators.predict(
            topology, grid_feature, query_xyz, query_offsets
        )
        prediction = self.outs(
            self.blocks(torch.cat([interpolated, query_xyz], dim=1))
        ).squeeze(-1)
        return (prediction if self.sdf else prediction.abs()), debug

    def forward(self, points, query_xyz, query_offsets):
        return self.predict(self.build_bank(points), query_xyz, query_offsets)
