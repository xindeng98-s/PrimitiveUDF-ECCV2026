from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.factory import build_dataset, collate_fn, worker_init_fn  # noqa: E402
from evaluation.checkpoints import (
    compose_flat_config,
    load_model,
    make_device,
)  # noqa: E402
from generation import (  # noqa: E402
    GENERATION_TYPES,
    build_mesh_generator,
    resolve_generation_type,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct V3 UDF meshes."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Lightning checkpoint or raw model state dict.",
    )
    parser.add_argument(
        "--config-name",
        default="config",
        help="Hydra config name. Defaults to the repository base config.",
    )
    parser.add_argument("--split", default="test", help="Dataset split to reconstruct.")
    parser.add_argument(
        "--output-dir", default=None, help="Directory for generated .obj files."
    )
    parser.add_argument(
        "--ckpt-name", default=None, help="Output label. Defaults to checkpoint stem."
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Defaults to runtime.seed from Hydra config.",
    )
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--bbox", type=float, default=None)
    parser.add_argument("--points-batch-size", type=int, default=None)
    parser.add_argument(
        "--field-type",
        choices=GENERATION_TYPES,
        default="udf",
        help=(
            "Unsigned-distance mesh generation."
        ),
    )
    parser.add_argument(
        "--query-margin-voxels",
        type=int,
        default=4,
        help="UDF-only dense-mask dilation radius.",
    )
    parser.add_argument(
        "--filter",
        type=float,
        default=None,
        dest="filter_denominator",
        help="UDF-only face filter.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--override", action="append", default=[], help="Hydra override, repeatable."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = compose_flat_config(args.override, config_name=args.config_name)
    normalize_project_paths(cfg)
    seed_generation(int(cfg.get("seed", 20) if args.seed is None else args.seed))
    cfg["batch_size"] = 1
    device = make_device(args.device)
    loader = build_loader(cfg, args.split, args.num_workers)
    model, notes = load_model(
        cfg,
        args.checkpoint,
        device,
        strict=True,
        validate_config=True,
    )
    field_type = resolve_generation_type(cfg, args.field_type)
    generator_kwargs = {
        "resolution": args.resolution,
        "bbox": args.bbox,
        "points_batch_size": args.points_batch_size,
    }
    generator_kwargs.update(query_margin_voxels=args.query_margin_voxels,
                            filter_denominator=args.filter_denominator)
    generator = build_mesh_generator(
        model,
        cfg,
        device=device,
        field_type=field_type,
        **generator_kwargs,
    )

    out_dir = resolve_output_dir(cfg, args, field_type)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for batch_idx, batch in enumerate(tqdm(loader, desc="reconstruct")):
        if args.limit is not None and batch_idx >= args.limit:
            break
        rel_path = str(batch["path"][0])
        mesh_path = mesh_output_path(out_dir, rel_path)
        mesh_path.parent.mkdir(parents=True, exist_ok=True)
        if args.skip_existing and mesh_path.exists():
            rows.append(
                row_for(batch_idx, rel_path, mesh_path, "skipped", None, field_type)
            )
            continue

        mesh = generator.generate_mesh(batch)
        if mesh is None:
            rows.append(
                row_for(
                    batch_idx,
                    rel_path,
                    mesh_path,
                    "empty",
                    generator.last_stats,
                    field_type,
                )
            )
            continue

        mesh.export(mesh_path)
        rows.append(
            row_for(
                batch_idx, rel_path, mesh_path, "ok", generator.last_stats, field_type
            )
        )

    write_manifest(out_dir / "manifest.csv", rows, notes)
    print(
        f"Wrote {sum(r['status'] == 'ok' for r in rows)} {field_type.upper()} meshes to {out_dir}"
    )


def build_loader(
    cfg: dict[str, Any], split: str, num_workers: int | None
) -> DataLoader:
    dataset = build_dataset(cfg, split)
    workers = cfg.get("val_num_workers", 0) if num_workers is None else num_workers
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(workers),
        pin_memory=bool(cfg.get("val_pin_memory", False)),
        drop_last=False,
        persistent_workers=(int(workers) > 0),
        collate_fn=collate_fn(dataset),
        worker_init_fn=worker_init_fn(dataset),
    )


def seed_generation(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_project_paths(cfg: dict[str, Any]) -> None:
    for key in ("data_dir", "raw_data_dir", "split_file"):
        value = cfg.get(key)
        if not value:
            continue
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            cfg[key] = str(PROJECT_ROOT / path)


def resolve_output_dir(
    cfg: dict[str, Any], args: argparse.Namespace, field_type: str = "udf"
) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser()
    ckpt_label = args.ckpt_name or Path(args.checkpoint).stem
    if not args.ckpt_name and getattr(args, "inference_head", None):
        ckpt_label = f"{ckpt_label}_{args.inference_head}"
    ckpt_path = Path(args.checkpoint).expanduser()
    if ckpt_path.parent.name == "checkpoints":
        exp_dir = ckpt_path.parent.parent
    else:
        exp_dir = experiment_dir(cfg)
    suffix = "sdf_mesh" if field_type == "sdf" else "mesh"
    return exp_dir / "evaluation" / f"{ckpt_label}_{suffix}"


def experiment_dir(cfg: dict[str, Any]) -> Path:
    root = Path(
        str(cfg.get("experiment_root", PROJECT_ROOT / "experiments"))
    ).expanduser()
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    log_dir = cfg.get("log_dir")
    if log_dir is None:
        raise ValueError(
            "runtime.log_dir is required when --output-dir cannot be inferred from --checkpoint."
        )
    return root / str(cfg["exp_name"]) / str(log_dir)


def mesh_output_path(out_dir: Path, rel_path: str) -> Path:
    parts = normalized_parts(rel_path)
    candidate = out_dir.joinpath(*parts[:-1], f"{parts[-1]}.obj")
    output_root = out_dir.expanduser().resolve()
    resolved_candidate = candidate.expanduser().resolve()
    try:
        resolved_candidate.relative_to(output_root)
    except ValueError as exc:
        raise ValueError(
            f"Dataset path {rel_path!r} resolves outside output directory {out_dir}."
        ) from exc
    return candidate


def normalized_parts(path: str) -> list[str]:
    raw_path = str(path)
    if "\x00" in raw_path:
        raise ValueError("Dataset path cannot contain a NUL byte.")
    portable_parts = raw_path.replace("\\", "/").split("/")
    if any(part in (".", "..") for part in portable_parts):
        raise ValueError(f"Dataset path traversal is not allowed: {path!r}.")
    parts = [part for part in portable_parts if part]
    if not parts:
        raise ValueError("Dataset path cannot be empty.")
    return parts


def row_for(
    batch_idx: int,
    rel_path: str,
    mesh_path: Path,
    status: str,
    stats,
    field_type: str | None = None,
) -> dict[str, Any]:
    row = {
        "batch_idx": batch_idx,
        "path": rel_path,
        "mesh_path": str(mesh_path),
        "status": status,
    }
    if field_type is not None:
        row["field_type"] = field_type
    if stats is not None:
        row.update(stats.__dict__)
    return row


def write_manifest(path: Path, rows: list[dict[str, Any]], notes: str) -> None:
    if not rows:
        return
    columns = sorted({key for row in rows for key in row})
    if "notes" not in columns:
        columns.append("notes")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["notes"] = notes
            writer.writerow(out)


if __name__ == "__main__":
    main()
