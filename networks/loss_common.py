"""Shared reconstruction and query-normal loss primitives.

The project uses nearest-surface ``moves`` as the zero-distance query paired
with every off-surface sample. Keeping the small scalar losses here makes the
default triangle head and the prototype head use exactly the same definitions.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def flatten_distance_prediction(
    value: torch.Tensor,
    expected_size: int,
    name: str,
) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a tensor")
    if value.numel() != int(expected_size):
        raise ValueError(
            f"{name} must contain {expected_size} scalars, got shape {tuple(value.shape)}"
        )
    return value.reshape(-1)


def prepare_distance_target(
    value: torch.Tensor,
    expected_size: int,
    *,
    is_sdf: bool,
) -> torch.Tensor:
    target = flatten_distance_prediction(value, expected_size, "off_df").float()
    return target if is_sdf else target.abs()


def distance_l1_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must have identical flattened shapes, "
            f"got {tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    return F.l1_loss(prediction.float(), target.float())


def zero_distance_l1_loss(prediction: torch.Tensor) -> torch.Tensor:
    return prediction.float().abs().mean()


def nearest_surface_normal_target(
    query: torch.Tensor,
    nearest_surface: torch.Tensor,
    distance_target: torch.Tensor,
    *,
    is_sdf: bool,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the target direction for a distance-field input gradient."""

    if query.shape != nearest_surface.shape or query.ndim != 2 or query.shape[1] != 3:
        raise ValueError(
            "query and nearest_surface must have matching [N,3] shapes, "
            f"got {tuple(query.shape)} and {tuple(nearest_surface.shape)}"
        )
    if distance_target.shape != query.shape[:1]:
        raise ValueError(
            f"distance_target must have shape {(query.shape[0],)}, "
            f"got {tuple(distance_target.shape)}"
        )

    displacement = query.detach().float() - nearest_surface.detach().float()
    displacement_norm = torch.linalg.norm(displacement, dim=-1)
    valid = displacement_norm > float(eps)
    target = F.normalize(displacement, dim=-1, eps=float(eps))
    if is_sdf:
        signs = torch.sign(distance_target.float())
        valid = valid & (signs != 0)
        target = target * signs.unsqueeze(-1)
    return target, valid


def masked_normal_cosine_loss(
    target: torch.Tensor,
    prediction: torch.Tensor | None,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    if prediction is None:
        return target.new_zeros(())
    if prediction.shape != target.shape:
        raise ValueError(
            "normal prediction and target must have identical shapes, "
            f"got {tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    if valid_mask.shape != target.shape[:1]:
        raise ValueError(
            f"valid_mask must have shape {(target.shape[0],)}, got {tuple(valid_mask.shape)}"
        )
    cosine = (target.float() * prediction.float()).sum(dim=-1).clamp(-1.0, 1.0)
    values = (1.0 - cosine)[valid_mask]
    return values.mean() if values.numel() else target.new_zeros(())
