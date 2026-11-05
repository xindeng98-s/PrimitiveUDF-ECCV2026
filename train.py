from __future__ import annotations

import datetime
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch.utils.data import DataLoader

from datasets.factory import build_dataset, collate_fn, worker_init_fn
from losses.factory import build_loss
from models.factory import build_model


PROJECT_ROOT = Path(__file__).resolve().parent
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


def seed_torch(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True


def flatten_config(hydra_cfg: DictConfig) -> Dict[str, Any]:
    resolved = OmegaConf.to_container(hydra_cfg, resolve=True)
    cfg: Dict[str, Any] = {}
    for group_name in CONFIG_GROUPS:
        group = resolved.get(group_name, {})
        if isinstance(group, Mapping):
            cfg.update(group)

    experiment = resolved.get("experiment", {})
    if isinstance(experiment, Mapping):
        cfg["exp_name"] = experiment.get("name", cfg.get("exp_name"))

    cfg["sample_ratio"] = np.array(cfg["sample_ratio"], dtype=float)
    cfg["sample_std_dev"] = np.array(cfg["sample_std_dev"], dtype=float)
    assert np.sum(cfg["sample_ratio"]) == 1
    assert not np.any(cfg["sample_ratio"] < 0)
    assert len(cfg["sample_ratio"]) == len(cfg["sample_std_dev"])
    return cfg


def config_for_wandb(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    out = {}
    for key, value in cfg.items():
        out[key] = value.tolist() if isinstance(value, np.ndarray) else value
    return out


def prime_rank_zero_from_environment() -> None:
    """Prevent eager loggers from initializing on non-zero DDP ranks."""
    rank_keys = (
        "RANK",
        "LOCAL_RANK",
        "NODE_RANK",
        "GROUP_RANK",
        "SLURM_PROCID",
        "JSM_NAMESPACE_RANK",
    )
    observed_ranks = [int(os.environ[key]) for key in rank_keys if key in os.environ]
    is_zero = not observed_ranks or all(rank == 0 for rank in observed_ranks)

    rank_zero_only.rank = 0 if is_zero else 1


def scheduler_steps_per_epoch(trainer: Any, fallback_steps: int) -> int:
    """Resolve optimizer steps from Lightning's distributed train loader."""
    fallback_steps = max(1, int(fallback_steps))
    if trainer is None:
        return fallback_steps

    _ = trainer.estimated_stepping_batches
    num_batches = float(trainer.num_training_batches)
    if not math.isfinite(num_batches):
        return fallback_steps
    accumulation = max(1, int(trainer.accumulate_grad_batches))
    return max(1, int(math.ceil(num_batches / accumulation)))


class RangeUDFDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Mapping[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            if self.train_dataset is None:
                self.train_dataset = self._build_dataset("train")
            if self.val_dataset is None:
                self.val_dataset = self._build_dataset("val")
        elif stage in ("validate", "test"):
            if self.val_dataset is None:
                self.val_dataset = self._build_dataset("val")

    def _build_dataset(self, mode: str):
        return build_dataset(self.cfg, mode)

    def train_dataloader(self):
        self.setup("fit")
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg["batch_size"],
            shuffle=True,
            num_workers=self.cfg["train_num_workers"],
            pin_memory=False,
            drop_last=False,
            collate_fn=collate_fn(self.train_dataset),
            worker_init_fn=worker_init_fn(self.train_dataset),
        )

    def val_dataloader(self):
        self.setup("validate")
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg["batch_size"],
            shuffle=False,
            num_workers=self.cfg["val_num_workers"],
            pin_memory=self.cfg["val_pin_memory"],
            drop_last=False,
            persistent_workers=(self.cfg["val_num_workers"] > 0),
            collate_fn=collate_fn(self.val_dataset),
            worker_init_fn=worker_init_fn(self.val_dataset),
        )

    def iters_per_epoch(self) -> int:
        self.setup("fit")
        return len(self.train_dataloader())


class RangeUDFModule(pl.LightningModule):
    def __init__(self, cfg: Mapping[str, Any], iters_per_epoch: int):
        super().__init__()
        self.cfg = cfg
        self.model = build_model(cfg)
        self.loss_fn = build_loss(cfg)
        self.iters_per_epoch = iters_per_epoch
        self.save_hyperparameters(config_for_wandb(cfg), ignore=["cfg", "iters_per_epoch"])

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        record = self._loss_record(batch, batch_idx, mode="train")
        batch_size = infer_batch_size(batch)
        epoch_only_logging = bool(self.cfg.get("epoch_only_logging", False))
        self.log_dict(
            prefixed_metrics("train", record["metrics"]),
            on_step=not epoch_only_logging,
            on_epoch=True,
            prog_bar=not epoch_only_logging,
            batch_size=batch_size,
            sync_dist=(getattr(self.trainer, "world_size", 1) > 1),
        )
        if self.current_epoch < 2:
            self.log_dict(
                primitive_debug_metrics(record["debug"]),
                on_step=not epoch_only_logging,
                on_epoch=epoch_only_logging,
                batch_size=batch_size,
            )
        return record["loss"]

    def validation_step(self, batch, batch_idx):
        ctx = torch.enable_grad() if need_val_grad(self.cfg) else torch.no_grad()
        with ctx:
            record = self._loss_record(batch, batch_idx, mode="val")
        batch_size = infer_batch_size(batch)
        self.log_dict(
            prefixed_metrics("val", record["metrics"]),
            on_step=False,
            on_epoch=True,
            prog_bar=not bool(self.cfg.get("epoch_only_logging", False)),
            batch_size=batch_size,
            sync_dist=(getattr(self.trainer, "world_size", 1) > 1),
        )
        return {
            "val_loss": record["metrics"]["loss"].detach(),
            "val_rec_loss": record["metrics"]["rec_loss"].detach(),
            "batch_size": batch_size,
        }

    def on_train_start(self):
        # Lightning restores optimizer state after scheduler construction.
        # Enforce the floor before the first resumed optimizer step as well.
        if self.cfg.get("scheduler") == "lambda":
            min_lr = float(self.cfg.get("min_lr", 0.0))
            for optimizer in self.trainer.optimizers:
                for group in optimizer.param_groups:
                    group["lr"] = max(group["lr"], min_lr)
            for config in self.trainer.lr_scheduler_configs:
                scheduler = config.scheduler
                scheduler._last_lr = [g["lr"] for g in scheduler.optimizer.param_groups]
            print(f"Training start learning rates: {[g['lr'] for opt in self.trainer.optimizers for g in opt.param_groups]}")

    def configure_optimizers(self):
        optimizer = build_optimizer(self.model.parameters(), self.cfg)
        steps_per_epoch = scheduler_steps_per_epoch(
            getattr(self, "_trainer", None),
            self.iters_per_epoch,
        )
        scheduler = build_scheduler(optimizer, self.cfg, steps_per_epoch)
        if scheduler is None:
            return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def _loss_record(self, batch, batch_idx: int, mode: str) -> Dict[str, Any]:
        set_epoch = getattr(self.loss_fn, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(int(self.current_epoch))
        return self.loss_fn(self.model, batch, mode=mode, batch_idx=batch_idx)


def build_optimizer(parameters, cfg: Mapping[str, Any]):
    name = cfg.get("optimizer", "Adam")
    lr = float(cfg.get("lr", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 1e-2))
    if name == "AdamW":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if name == "Adam":
        return torch.optim.Adam(parameters, lr=lr)
    if name == "SGD":
        return torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "RMSprop":
        return torch.optim.RMSprop(parameters, lr=lr, momentum=0.9)
    if name == "Adadelta":
        return torch.optim.Adadelta(parameters, lr=lr)
    return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer, cfg: Mapping[str, Any], iters_per_epoch: int):
    name = cfg.get("scheduler")
    if not name:
        return None
    if name == "lambda":
        decay_milestones = cfg.get("decay_milestones", [200, 400])
        if not bool(cfg.get("decay", True)):
            decay_milestones = []
        decay_points = [iters_per_epoch * int(epoch) for epoch in decay_milestones]
        decay_ratio = float(cfg.get("decay_ratio", 0.3))
        warmup_steps = max(0, int(cfg.get("warmup_steps", 1000)))
        warmup_start_factor = float(cfg.get("warmup_start_factor", 0.1))
        scheduler_step_offset = int(cfg.get("scheduler_step_offset", 0))
        if not 0.0 <= warmup_start_factor <= 1.0:
            raise ValueError("scheduler.warmup_start_factor must be between 0 and 1.")

        min_lr = float(cfg.get("min_lr", 0.0))
        base_lr = float(optimizer.param_groups[0]["lr"])
        if not math.isfinite(min_lr) or not 0.0 <= min_lr <= base_lr:
            raise ValueError("scheduler.min_lr must be finite and between zero and the base learning rate.")
        min_factor = min_lr / base_lr if base_lr > 0 else 0.0

        def lambda_cumulative(iteration):
            schedule_iteration = max(0, int(iteration) + scheduler_step_offset)
            if warmup_steps > 0 and schedule_iteration < warmup_steps:
                progress = float(schedule_iteration) / float(warmup_steps)
                return max(min_factor, warmup_start_factor + (1.0 - warmup_start_factor) * progress)
            k = sum(1 for point in decay_points if schedule_iteration >= point)
            return max(min_factor, (decay_ratio ** k) if k > 0 else 1.0)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_cumulative)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=12500, gamma=0.7)
    if name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(cfg.get("lr", 1e-3)),
            steps_per_epoch=max(1, int(iters_per_epoch)),
            epochs=int(cfg.get("num_epochs", 1)),
            pct_start=0.1,
        )
    return None


def need_val_grad(cfg: Mapping[str, Any]) -> bool:
    mode = cfg.get("norm_val", "")
    if mode is None:
        return False
    if isinstance(mode, bool):
        return mode
    return str(mode) in {"grad", "move_dis", "move_chamfer"}


def infer_batch_size(batch: Any) -> float:
    if "path" in batch:
        return float(len(batch["path"]))
    if "on_offsets" in batch and torch.is_tensor(batch["on_offsets"]):
        return float(batch["on_offsets"].numel())
    if "off_offsets" in batch and torch.is_tensor(batch["off_offsets"]):
        return float(batch["off_offsets"].numel())
    raise RuntimeError(f"Cannot infer batch size from keys: {list(batch.keys())}")


def prefixed_metrics(prefix: str, metrics: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {f"{prefix}/{key}": value for key, value in metrics.items()}


def primitive_debug_metrics(debug: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    metrics = {}
    for key in (
        "anchor_triangle_count",
        "anchor_point_primitive_count",
        "anchor_valid_primitive_count",
        "anchor_fallback_used",
        "anchor_point_only",
        "query_triangle_count",
        "query_point_primitive_count",
        "query_valid_primitive_count",
        "query_fallback_used",
        "query_point_only",
    ):
        value = debug.get(key)
        if value is not None and torch.is_tensor(value):
            suffix = "rate" if value.dtype == torch.bool else "mean"
            metrics[f"primitive/{key}_{suffix}"] = value.float().mean()
    return metrics


def experiment_dir(cfg: Dict[str, Any]) -> Path:
    log_dir = cfg["log_dir"]
    if log_dir is None:
        log_dir = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        cfg["log_dir"] = log_dir
    root = Path(cfg.get("experiment_root", PROJECT_ROOT / "experiments")).expanduser()
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    exp_dir = root / cfg["exp_name"] / str(log_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def build_logger(cfg: Mapping[str, Any], exp_dir: Path):
    run_name = cfg.get("logger_run_name") or f"{cfg['exp_name']}/{cfg['log_dir']}"
    return WandbLogger(
        project=cfg["logger_project"],
        entity=cfg.get("logger_entity"),
        name=run_name,
        save_dir=str(exp_dir),
        offline=bool(cfg.get("logger_offline", False)),
        log_model=bool(cfg.get("logger_log_model", False)),
        config=config_for_wandb(cfg),
    )


def build_callbacks(cfg: Mapping[str, Any], exp_dir: Path):
    checkpoint_dir = exp_dir / str(cfg.get("dirname", "checkpoints"))
    return [
        LearningRateMonitor(
            logging_interval=(
                "epoch" if bool(cfg.get("epoch_only_logging", False)) else "step"
            )
        ),
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename=cfg.get("filename", "epoch_{epoch:03d}"),
            monitor=cfg.get("monitor", "val/rec_loss"),
            mode=cfg.get("mode", "min"),
            save_top_k=int(cfg.get("save_top_k", 5)),
            save_last=bool(cfg.get("save_last", True)),
            auto_insert_metric_name=False,
        ),
    ]


@hydra.main(version_base=None, config_path="configs_hydra", config_name="config")
def main(hydra_cfg: DictConfig) -> None:
    cfg = flatten_config(hydra_cfg)
    prime_rank_zero_from_environment()
    seed_torch(int(cfg["seed"]))
    exp_dir = experiment_dir(cfg)

    data = RangeUDFDataModule(cfg)
    run_mode = str(cfg.get("run_mode", "fit"))
    if run_mode == "fit":
        data.setup("fit")
        module = RangeUDFModule(cfg, iters_per_epoch=data.iters_per_epoch())
    elif run_mode in {"eval", "validate"}:
        data.setup("validate")
        module = RangeUDFModule(cfg, iters_per_epoch=1)
    else:
        raise ValueError(f"Unknown runtime.run_mode: {run_mode}")

    trainer = pl.Trainer(
        accelerator=cfg["accelerator"],
        devices=cfg["devices"],
        num_nodes=cfg["num_nodes"],
        strategy=cfg["strategy"],
        precision=cfg["precision"],
        sync_batchnorm=cfg["sync_batchnorm"],
        replace_sampler_ddp=cfg["replace_sampler_ddp"],
        max_epochs=cfg["num_epochs"],
        logger=build_logger(cfg, exp_dir),
        callbacks=build_callbacks(cfg, exp_dir),
        gradient_clip_val=cfg["gradient_clip_val"],
        accumulate_grad_batches=cfg["accumulate_grad_batches"],
        log_every_n_steps=cfg["log_every_n_steps"],
        num_sanity_val_steps=cfg["num_sanity_val_steps"],
        limit_train_batches=cfg["limit_train_batches"],
        limit_val_batches=cfg["limit_val_batches"],
        deterministic=cfg["deterministic"],
        fast_dev_run=cfg["fast_dev_run"],
        enable_progress_bar=bool(cfg.get("enable_progress_bar", True)),
        enable_model_summary=False,
    )
    if run_mode == "fit":
        trainer.fit(module, datamodule=data, ckpt_path=cfg.get("resume_ckpt"))
    else:
        ckpt_path = cfg.get("eval_ckpt") or cfg.get("resume_ckpt")
        trainer.validate(module, datamodule=data, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
