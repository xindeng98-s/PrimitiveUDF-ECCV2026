from __future__ import annotations

from importlib import import_module
from typing import Any, Callable, Mapping


def build_dataset(cfg: Mapping[str, Any], mode: str):
    module_name = cfg.get("dataset_module")
    builder_name = cfg.get("dataset_builder", "build_dataset")
    if not module_name:
        raise ValueError("data.dataset_module must point to a dataset python module.")

    module = import_module(str(module_name))
    if not hasattr(module, str(builder_name)):
        raise ValueError(f"{module_name} must define {builder_name}(cfg, mode).")

    dataset = getattr(module, str(builder_name))(cfg, mode)
    validate_dataset(dataset, module_name)
    return dataset


def validate_dataset(dataset, module_name: str) -> None:
    required = ("collate_fn_pt2", "worker_init_fn")
    missing = [name for name in required if not hasattr(dataset, name)]
    if missing:
        missing_names = ", ".join(missing)
        raise ValueError(f"{module_name} dataset is missing required methods: {missing_names}")


def collate_fn(dataset) -> Callable:
    return dataset.collate_fn_pt2


def worker_init_fn(dataset) -> Callable:
    return dataset.worker_init_fn
