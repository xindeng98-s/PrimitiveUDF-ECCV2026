from __future__ import annotations

import csv
import math
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs_hydra"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.factory import build_dataset, collate_fn, worker_init_fn  # noqa: E402
from models.factory import build_model  # noqa: E402


CONFIG_GROUPS = (
    "experiment",
    "runtime",
    "data",
    "model",
    "loss",
    "optim",
    "scheduler",
    "train",
    "trainer",
    "logger",
    "checkpoint",
)

GENERATION_CHECKPOINT_CONFIG_KEYS = (
    "model_module",
    "model_builder",
    "backbone_module",
    "backbone_builder",
    "head_module",
    "head_builder",
    "in_dim",
    "is_sdf",
    "grid_first",
    "init_grid",
    "sub_sampling_ratio",
    "num_neighbors",
    "num_interp",
)

def compose_flat_config(
    overrides: list[str] | None = None, *, config_name: str = "config"
) -> dict[str, Any]:
    overrides = list(overrides or [])
    with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base=None):
        hydra_cfg = compose(config_name=config_name, overrides=overrides)
    return flatten_config(hydra_cfg)


def flatten_config(hydra_cfg) -> dict[str, Any]:
    resolved = OmegaConf.to_container(hydra_cfg, resolve=True)
    cfg: dict[str, Any] = {}
    for group_name in CONFIG_GROUPS:
        group = resolved.get(group_name, {})
        if isinstance(group, dict):
            cfg.update(group)

    for section in ("preparation", "evaluation"):
        cfg[section] = resolved.get(section, {})

    experiment = resolved.get("experiment", {})
    if isinstance(experiment, dict):
        cfg["exp_name"] = experiment.get("name", cfg.get("exp_name"))

    cfg["sample_ratio"] = np.array(cfg["sample_ratio"], dtype=float)
    cfg["sample_std_dev"] = np.array(cfg["sample_std_dev"], dtype=float)
    return cfg


def make_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_val_loader(cfg: dict[str, Any]) -> DataLoader:
    dataset = build_dataset(cfg, "val")
    return DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["val_num_workers"],
        pin_memory=bool(cfg.get("val_pin_memory", False)),
        drop_last=False,
        persistent_workers=(cfg["val_num_workers"] > 0),
        collate_fn=collate_fn(dataset),
        worker_init_fn=worker_init_fn(dataset),
    )


def _checkpoint_hparams(checkpoint_data: Mapping[str, Any]) -> Mapping[str, Any] | None:
    hyper_parameters = checkpoint_data.get("hyper_parameters")
    if not isinstance(hyper_parameters, Mapping):
        return None
    nested_cfg = hyper_parameters.get("cfg")
    return nested_cfg if isinstance(nested_cfg, Mapping) else hyper_parameters


def _normalized_config_value(value: Any) -> Any:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return [_normalized_config_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _normalized_config_value(item) for key, item in value.items()}
    return value


def generation_checkpoint_config_keys(cfg):
    keys = [k for k in GENERATION_CHECKPOINT_CONFIG_KEYS if k != 'num_interp']
    return tuple(keys + ['primitive_bank_neighbor_k', 'primitive_bank_keep_prims'])


def validate_checkpoint_config(
    cfg: Mapping[str, Any],
    checkpoint_cfg: Mapping[str, Any],
    keys: tuple[str, ...] | None = None,
) -> None:
    mismatches = []
    for key in keys or generation_checkpoint_config_keys(cfg):
        if key not in cfg:
            mismatches.append(f"{key}: missing from runtime config")
            continue
        if key not in checkpoint_cfg:
            mismatches.append(f"{key}: missing from checkpoint metadata")
            continue
        runtime_value = _normalized_config_value(cfg[key])
        checkpoint_value = _normalized_config_value(checkpoint_cfg[key])
        if runtime_value != checkpoint_value:
            mismatches.append(
                f"{key}: runtime={runtime_value!r}, checkpoint={checkpoint_value!r}"
            )
    if mismatches:
        details = "; ".join(mismatches)
        raise ValueError(f"Checkpoint/config mismatch for generation: {details}")


def load_model(
    cfg: dict[str, Any],
    checkpoint: str | None,
    device: torch.device,
    *,
    strict: bool = False,
    validate_config: bool = False,
):
    model = build_model(cfg).to(device)
    notes = []
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device)
        if not isinstance(ckpt, Mapping):
            raise TypeError("Checkpoint must contain a state-dict mapping.")
        if validate_config:
            checkpoint_cfg = _checkpoint_hparams(ckpt)
            if checkpoint_cfg is None:
                if "state_dict" in ckpt:
                    raise ValueError(
                        "Lightning checkpoint is missing hyper_parameters needed "
                        "for generation config validation."
                    )
                notes.append("checkpoint_config=unavailable")
            else:
                validate_checkpoint_config(cfg, checkpoint_cfg)
        state_dict = ckpt.get("state_dict", ckpt)
        if not isinstance(state_dict, Mapping):
            raise TypeError("Checkpoint state_dict must be a mapping.")
        stripped = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                stripped[key[len("model.") :]] = value
            else:
                stripped[key] = value
        try:
            missing, unexpected = model.load_state_dict(stripped, strict=strict)
        except RuntimeError as exc:
            if strict:
                raise RuntimeError(
                    f"Checkpoint {checkpoint!r} is not strictly compatible with "
                    "the configured model."
                ) from exc
            raise
        if missing:
            notes.append(f"missing={len(missing)}")
        if unexpected:
            notes.append(f"unexpected={len(unexpected)}")
    model.eval()
    return model, ";".join(notes)
