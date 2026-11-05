from __future__ import annotations

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- geometry-only primitive modules ----
from heads.primitive_geometry import GeometryPrimitiveConstructor, TrianglePrimitiveFeatureEncoder, Att_pooling_regressor_logitsbias

# ---- PyG / torch-cluster KNN ----
try:
    from torch_geometric.nn import knn
except Exception:
    try:
        from torch_cluster import knn  # type: ignore
    except Exception:
        knn = None





class PrimitiveInterpolationLayers(nn.Module):
    def __init__(
        self,
        d_out: int,
        prim_dim: int = 48,
        keep_prims: int = 10,
        bank_neighbor_k: int = 16,
        query_anchor_k: int = 2,
        bank_prims_per_anchor: int = 10,
        detach_construct: bool = True,
        primitive_dropout_p: float = 0.0,
        primitive_dropout_eval: bool = False,
    ):
        super().__init__()
        self.d_out = int(d_out)
        self.prim_dim = int(prim_dim)
        self.keep_prims = int(keep_prims)
        self.bank_neighbor_k = int(bank_neighbor_k)
        self.query_anchor_k = int(query_anchor_k)
        self.bank_prims_per_anchor = int(bank_prims_per_anchor)
        self.primitive_dropout_p = float(primitive_dropout_p)
        self.primitive_dropout_eval = bool(primitive_dropout_eval)
        self.score_neg_inf = -1e9
        self.last_primitive_dropout_debug = None
        self.last_forward_profile = None
        self.profile_runtime = False
        self.print_profile = False

        if self.keep_prims <= 0:
            raise ValueError("primitive_bank_keep_prims must be positive.")
        if self.bank_neighbor_k <= 0:
            raise ValueError("primitive_bank_neighbor_k must be positive.")
        if self.query_anchor_k <= 0:
            raise ValueError("primitive_bank_query_anchors must be positive.")
        if self.bank_prims_per_anchor <= 0:
            raise ValueError("primitive_bank_prims_per_anchor must be positive.")

        self.geometry = GeometryPrimitiveConstructor(
            keep_prims=self.bank_prims_per_anchor,
            detach_construct=detach_construct,
        )

        self.prim_encoder = TrianglePrimitiveFeatureEncoder(
            point_feat_dim=self.d_out,
            prim_dim=self.prim_dim,
            hidden_dim=self.prim_dim,
        )

        self.prim_pool = Att_pooling_regressor_logitsbias(
            d_prim=self.prim_dim,
            pos_hidden=32,
            nh=2,
        )

        self.out_proj = (
            nn.Identity()
            if self.prim_dim == self.d_out
            else nn.Linear(self.prim_dim, self.d_out, bias=False)
        )

    def _apply_primitive_dropout(self, prim_mask: torch.Tensor):
        valid = prim_mask.bool()
        N, T = valid.shape
        device = valid.device

        p = float(self.primitive_dropout_p)
        active = (p > 0.0) and (self.training or self.primitive_dropout_eval)

        if (not active) or T == 0:
            drop_ratio = torch.zeros((N,), device=device, dtype=torch.float32)
            return valid, drop_ratio

        keep_random = torch.rand((N, T), device=device) >= p
        keep_mask = valid & keep_random

        has_valid = valid.any(dim=1)
        empty_after_drop = has_valid & (~keep_mask.any(dim=1))
        if empty_after_drop.any():
            rescue_score = torch.rand((N, T), device=device).masked_fill(~valid, -1.0)
            rescue_idx = rescue_score.argmax(dim=1)
            rows = torch.where(empty_after_drop)[0]
            keep_mask[rows, rescue_idx[rows]] = True

        drop_ratio = (
            (valid & (~keep_mask)).float().sum(dim=1)
            / valid.float().sum(dim=1).clamp_min(1.0)
        )
        return keep_mask, drop_ratio
