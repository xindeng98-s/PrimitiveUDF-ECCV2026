from __future__ import annotations

from importlib import import_module
from typing import Any, Mapping


def build_model(cfg: Mapping[str, Any]):
    module_name = cfg.get("model_module")
    builder_name = cfg.get("model_builder", "build_model")
    if not module_name:
        raise ValueError("model.model_module must point to a model python module.")

    module = import_module(str(module_name))
    if not hasattr(module, str(builder_name)):
        raise ValueError(f"{module_name} must define {builder_name}(cfg).")

    model = getattr(module, str(builder_name))(cfg)
    validate_model(model, module_name)
    return model


def validate_model(model, module_name: str) -> None:
    if not callable(model):
        raise ValueError(f"{module_name} builder must return a callable model object.")
