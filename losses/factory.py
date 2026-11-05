from __future__ import annotations

from importlib import import_module
from typing import Any, Mapping


def build_loss(cfg: Mapping[str, Any]):
    module_name = cfg.get("loss_module")
    builder_name = cfg.get("loss_builder", "build_loss")
    if not module_name:
        raise ValueError("loss.loss_module must point to a loss python module.")

    module = import_module(str(module_name))
    if not hasattr(module, str(builder_name)):
        raise ValueError(f"{module_name} must define {builder_name}(cfg).")

    loss_fn = getattr(module, str(builder_name))(cfg)
    validate_loss(loss_fn, module_name)
    return loss_fn


def validate_loss(loss_fn, module_name: str) -> None:
    if not callable(loss_fn):
        raise ValueError(f"{module_name} builder must return a callable loss object.")
