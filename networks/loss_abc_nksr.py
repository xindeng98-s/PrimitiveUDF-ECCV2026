"""Reconstruction and normal losses for PrimitiveUDF."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from networks.loss_common import (
    distance_l1_loss,
    flatten_distance_prediction,
    masked_normal_cosine_loss,
    nearest_surface_normal_target,
    prepare_distance_target,
    zero_distance_l1_loss,
)


def build_loss(cfg):
    return ReconstructionLoss(cfg)


class ReconstructionLoss:
    """Off/on L1 reconstruction plus optional query-gradient normal loss."""

    def __init__(self, cfg: Mapping[str, Any]):
        self.cfg = cfg
        _loss_weights(cfg)
        if int(cfg.get("normal_query_stride", 1)) <= 0:
            raise ValueError("normal_query_stride must be positive")

    def __call__(self, model, batch, mode, batch_idx=None):
        record = compute_loss(model, batch, self.cfg, mode)
        record["mode"] = mode
        record["batch_idx"] = batch_idx
        return record


def _loss_weights(cfg: Mapping[str, Any]) -> dict[str, float]:
    # Individual reconstruction weights override the shared weight.
    legacy_rec = float(cfg.get("rec_weight", 1.0))
    weights = {
        "off_rec": float(cfg.get("off_rec_weight", legacy_rec)),
        "on_rec": float(cfg.get("on_rec_weight", 0.1 * legacy_rec)),
        "norm": float(cfg.get("norm_weight", 0.0)),
        "on_norm": float(cfg.get("on_norm_weight", 0.0)),
    }
    for name, value in weights.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"loss weight {name!r} must be finite and >= 0, got {value}")
    return weights


def _model_device(model, batch) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        for value in batch.values():
            if torch.is_tensor(value):
                return value.device
    return torch.device("cpu")


def _to_device_tensor(value, name: str, device: torch.device) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"batch[{name!r}] must be a tensor")
    return value.to(device, non_blocking=True)


def _validate_offset(offset: torch.Tensor, total: int, name: str) -> torch.Tensor:
    if offset.ndim != 1 or offset.numel() == 0:
        raise ValueError(f"{name} must be a non-empty 1D cumulative offset tensor")
    offset = offset.to(torch.long)
    if torch.any(offset < 0) or torch.any(offset[1:] < offset[:-1]):
        raise ValueError(f"{name} must be non-negative and monotonically non-decreasing")
    if int(offset[-1].item()) != int(total):
        raise ValueError(f"{name}[-1]={int(offset[-1])} does not match tensor length {total}")
    return offset


def _prepare_inputs(model, batch, cfg: Mapping[str, Any]) -> dict[str, torch.Tensor | None]:
    device = _model_device(model, batch)
    on_xyz = _to_device_tensor(batch.get("on_surface_points"), "on_surface_points", device)
    off_xyz = _to_device_tensor(batch.get("off_surface_points"), "off_surface_points", device)
    moves = _to_device_tensor(batch.get("moves"), "moves", device)
    off_df_raw = _to_device_tensor(batch.get("off_df"), "off_df", device)
    on_offset = _to_device_tensor(batch.get("on_offsets"), "on_offsets", device)
    off_offset = _to_device_tensor(batch.get("off_offsets"), "off_offsets", device)

    if on_xyz.ndim != 2 or on_xyz.shape[1] != 3:
        raise ValueError(f"on_surface_points must have shape [N,3], got {tuple(on_xyz.shape)}")
    if off_xyz.ndim != 2 or off_xyz.shape[1] != 3:
        raise ValueError(f"off_surface_points must have shape [N,3], got {tuple(off_xyz.shape)}")
    if moves.shape != off_xyz.shape:
        raise ValueError(
            f"moves must match off_surface_points shape {tuple(off_xyz.shape)}, "
            f"got {tuple(moves.shape)}"
        )
    if off_xyz.shape[0] == 0:
        raise ValueError("reconstruction loss requires at least one off-surface query")

    on_offset = _validate_offset(on_offset, on_xyz.shape[0], "on_offsets")
    off_offset = _validate_offset(off_offset, off_xyz.shape[0], "off_offsets")
    if on_offset.numel() != off_offset.numel():
        raise ValueError("on_offsets and off_offsets must describe the same batch size")
    on_counts = torch.diff(torch.cat([on_offset.new_zeros(1), on_offset]))
    off_counts = torch.diff(torch.cat([off_offset.new_zeros(1), off_offset]))
    if torch.any((off_counts > 0) & (on_counts == 0)):
        raise ValueError("query-bearing batch items must contain support points")

    feature = None
    if int(cfg.get("in_dim", 3)) == 6:
        feature = _to_device_tensor(batch.get("feature"), "feature", device)
        if feature.ndim != 2 or feature.shape[0] != on_xyz.shape[0]:
            raise ValueError(
                "feature must have shape [num_on_surface_points,C], "
                f"got {tuple(feature.shape)}"
            )

    target = prepare_distance_target(
        off_df_raw,
        off_xyz.shape[0],
        is_sdf=bool(cfg.get("is_sdf", False)),
    )
    return {
        "on_xyz": on_xyz,
        "off_xyz": off_xyz,
        "moves": moves,
        "feature": feature,
        "target": target,
        "on_offset": on_offset,
        "off_offset": off_offset,
    }


def _head_forward(head, state, query, offset, need_grad: bool, label: str):
    if need_grad:
        query = query.detach().requires_grad_(True)
    output = head(state, query, offset)
    if not isinstance(output, (tuple, list)) or len(output) != 2:
        raise TypeError("head must return (prediction, debug)")
    prediction = flatten_distance_prediction(output[0], query.shape[0], f"{label}_prediction")

    gradient = None
    if need_grad:
        gradient = torch.autograd.grad(
            outputs=prediction,
            inputs=query,
            grad_outputs=torch.ones_like(prediction),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradient = F.normalize(gradient.float(), dim=-1, eps=1e-6)
    return prediction, output[1], query, gradient


def _strided_query_subset(offset: torch.Tensor, stride: int):
    counts = torch.diff(torch.cat([offset.new_zeros(1), offset]))
    starts = torch.repeat_interleave(offset - counts, counts)
    local_index = torch.arange(int(offset[-1]), device=offset.device) - starts
    index = torch.where(local_index.remainder(stride) == 0)[0]
    subset_offset = torch.cumsum((counts + stride - 1) // stride, dim=0)
    return index, subset_offset


def compute_loss(model, batch, cfg: Mapping[str, Any], mode: str):
    if mode not in {"train", "val", "validate", "test"}:
        raise ValueError(f"unsupported loss mode: {mode!r}")
    is_train = mode == "train"
    weights = _loss_weights(cfg)
    module = model.module if hasattr(model, "module") else model
    inputs = _prepare_inputs(module, batch, cfg)

    points = module.backbone(
        {
            "coord": inputs["on_xyz"],
            "feat": inputs["feature"],
            "offset": inputs["on_offset"],
        }
    )
    head = module.task
    head_state = points
    build_bank = getattr(head, "build_bank", None)
    if build_bank is not None:
        head_state = build_bank(points)
        head = head.predict

    need_off_grad = is_train and weights["norm"] > 0
    need_on_grad = is_train and weights["on_norm"] > 0
    normal_query_stride = int(cfg.get("normal_query_stride", 1))
    reuse_off_gradient = need_off_grad and normal_query_stride == 1
    off_pred, debug, off_query, off_gradient = _head_forward(
        head,
        head_state,
        inputs["off_xyz"],
        inputs["off_offset"],
        reuse_off_gradient,
        "off",
    )
    on_pred, _, _, on_gradient = _head_forward(
        head,
        head_state,
        inputs["moves"],
        inputs["off_offset"],
        need_on_grad,
        "on",
    )

    off_l1 = distance_l1_loss(off_pred, inputs["target"])
    on_l1 = zero_distance_l1_loss(on_pred)
    zero = off_l1.new_zeros(())
    normal_off = zero
    normal_on = zero
    if need_off_grad:
        normal_query = off_query
        normal_moves = inputs["moves"]
        normal_target_distance = inputs["target"]
        if not reuse_off_gradient:
            normal_index, normal_offset = _strided_query_subset(
                inputs["off_offset"], normal_query_stride
            )
            _, _, normal_query, off_gradient = _head_forward(
                head,
                head_state,
                inputs["off_xyz"][normal_index],
                normal_offset,
                True,
                "normal_off",
            )
            normal_moves = inputs["moves"][normal_index]
            normal_target_distance = inputs["target"][normal_index]
        normal_target, normal_valid = nearest_surface_normal_target(
            normal_query,
            normal_moves,
            normal_target_distance,
            is_sdf=bool(cfg.get("is_sdf", False)),
        )
        normal_off = masked_normal_cosine_loss(
            normal_target, off_gradient, normal_valid
        )
    if need_on_grad:
        normal_target, normal_valid = nearest_surface_normal_target(
            off_query,
            inputs["moves"],
            inputs["target"],
            is_sdf=bool(cfg.get("is_sdf", False)),
        )
        normal_on = masked_normal_cosine_loss(
            normal_target, on_gradient, normal_valid
        )

    loss_rec = weights["off_rec"] * off_l1 + weights["on_rec"] * on_l1
    loss_norm = weights["norm"] * normal_off + weights["on_norm"] * normal_on
    loss = loss_rec + loss_norm
    metrics = {
        "loss": loss,
        "rec_loss": loss_rec.detach(),
        "off": off_l1.detach(),
        "on": on_l1.detach(),
        "normal_off": normal_off.detach(),
        "normal_on": normal_on.detach(),
        "loss_rec": loss_rec.detach(),
        "loss_norm": loss_norm.detach(),
        "weight_off_rec": loss.new_tensor(weights["off_rec"]).detach(),
        "weight_on_rec": loss.new_tensor(weights["on_rec"]).detach(),
        "weight_norm": loss.new_tensor(weights["norm"]).detach(),
        "weight_on_norm": loss.new_tensor(weights["on_norm"]).detach(),
    }
    return {"loss": loss, "metrics": metrics, "debug": debug or {}}


# Individual reconstruction and normal-loss functions.
def compute_off_loss(df_pred, df_gt, grid_size=None):
    del grid_size
    return distance_l1_loss(df_pred.reshape(-1), df_gt.reshape(-1).abs())


def compute_on_loss(dense_pred, grid_size=None):
    del grid_size
    return zero_distance_l1_loss(dense_pred.reshape(-1))


def compute_on_normal_loss(dense_normals, gradient_org, grid_size=None):
    del grid_size
    valid = torch.ones(
        dense_normals.shape[0], dtype=torch.bool, device=dense_normals.device
    )
    return masked_normal_cosine_loss(dense_normals, gradient_org, valid)
