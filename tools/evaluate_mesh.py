from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

for thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(thread_env, "1")


import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.mesh_metrics import evaluate_mesh_against_points  # noqa: E402
from evaluation.checkpoints import compose_flat_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate reconstructed meshes against GT point npz files."
    )
    parser.add_argument(
        "--config-name",
        default="config",
        help="Hydra config name. Defaults to the repository base config.",
    )
    parser.add_argument(
        "--mesh-dir",
        default=None,
        help="Directory containing <room>/<scene>.obj outputs.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint path used to infer default mesh-dir.",
    )
    parser.add_argument(
        "--ckpt-name",
        default=None,
        help="Optional output label used with default mesh-dir.",
    )
    parser.add_argument(
        "--field-type",
        choices=("udf", "sdf", "auto"),
        default="udf",
        help="Selects the default *_mesh or *_sdf_mesh directory; auto follows model.is_sdf.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--summary-txt", default=None)
    parser.add_argument("--sample-points", type=positive_int, default=None)
    parser.add_argument(
        "--reference-points",
        type=positive_int,
        default=None,
        help=(
            "Number of GT surface points sampled without replacement."
        ),
    )
    parser.add_argument("--bbox-margin", type=float, default=None)
    parser.add_argument("--distance-threshold", type=float, default=None)
    parser.add_argument(
        "--no-reference-filter",
        action="store_true", default=None,
        help=(
            "Disable the vertex-distance filter against sampled GT reference "
            "points. The GT bounding-box filter remains enabled."
        ),
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--threads", type=positive_int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-repair", action="store_true", default=None)
    parser.add_argument(
        "--override", action="append", default=[], help="Hydra override, repeatable."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = compose_flat_config(args.override, config_name=args.config_name)
    for key, value in cfg["evaluation"].items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    normalize_project_paths(cfg)
    mesh_dir = resolve_mesh_dir(cfg, args)
    paths = list(split_paths(cfg, args.split))
    if args.limit is not None:
        paths = paths[: max(args.limit, 0)]
    rows = evaluate_rows(cfg, args, mesh_dir, paths)

    output_csv = (
        Path(args.output_csv).expanduser()
        if args.output_csv
        else mesh_dir / "evaluate_mesh.csv"
    )
    summary_txt = (
        Path(args.summary_txt).expanduser()
        if args.summary_txt
        else mesh_dir / "evaluate_mesh_summary.txt"
    )
    write_csv(output_csv, rows)
    write_summary(summary_txt, rows)
    print(f"Wrote evaluation CSV to {output_csv}")
    print(f"Wrote summary to {summary_txt}")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def evaluate_rows(
    cfg: dict[str, Any],
    args: argparse.Namespace,
    mesh_dir: Path,
    paths: list[str],
) -> list[dict[str, Any]]:
    worker = partial(evaluate_one, cfg=cfg, args=args, mesh_dir=mesh_dir)
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        return list(
            tqdm(executor.map(worker, paths), total=len(paths), desc="evaluate")
        )


def evaluate_one(
    rel_path: str,
    *,
    cfg: dict[str, Any],
    args: argparse.Namespace,
    mesh_dir: Path,
) -> dict[str, Any]:
    mesh_path = mesh_output_path(mesh_dir, rel_path)
    gt_path = gt_npz_path(cfg, rel_path)
    row: dict[str, Any] = {
        "path": rel_path,
        "mesh_path": str(mesh_path),
        "gt_path": str(gt_path),
    }
    try:
        result = evaluate_mesh_against_points(
            mesh_path,
            gt_path,
            sample_points=args.sample_points,
            reference_points=args.reference_points,
            bbox_margin=args.bbox_margin,
            distance_threshold=args.distance_threshold,
            seed=args.seed,
            repair=not args.no_repair,
            reference_filter=not getattr(args, "no_reference_filter", False),
        )
        row.update(result.to_dict())
        row["status"] = "ok"
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = str(exc)
    return row


def resolve_field_type(cfg: dict[str, Any], requested: str) -> str:
    if requested == "auto":
        return "sdf" if bool(cfg.get("is_sdf", False)) else "udf"
    return requested


def resolve_mesh_dir(cfg: dict[str, Any], args: argparse.Namespace) -> Path:
    if args.mesh_dir:
        return Path(args.mesh_dir).expanduser()
    field_type = resolve_field_type(cfg, getattr(args, "field_type", "udf"))
    suffix = "sdf_mesh" if field_type == "sdf" else "mesh"
    if args.checkpoint:
        ckpt = Path(args.checkpoint).expanduser()
        label = args.ckpt_name or ckpt.stem
        exp_dir = (
            ckpt.parent.parent
            if ckpt.parent.name == "checkpoints"
            else experiment_dir(cfg)
        )
        return exp_dir / "evaluation" / f"{label}_{suffix}"
    label = args.ckpt_name or "mesh"
    return experiment_dir(cfg) / "evaluation" / f"{label}_{suffix}"


def normalize_project_paths(cfg: dict[str, Any]) -> None:
    for key in ("data_dir", "raw_data_dir", "split_file"):
        value = cfg.get(key)
        if not value:
            continue
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            cfg[key] = str(PROJECT_ROOT / path)


def experiment_dir(cfg: dict[str, Any]) -> Path:
    root = Path(
        str(cfg.get("experiment_root", PROJECT_ROOT / "experiments"))
    ).expanduser()
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    log_dir = cfg.get("log_dir")
    if log_dir is None:
        raise ValueError(
            "runtime.log_dir is required when --mesh-dir cannot be inferred."
        )
    return root / str(cfg["exp_name"]) / str(log_dir)


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


def _safe_output_path(out_dir: Path, parts: list[str]) -> Path:
    candidate = out_dir.joinpath(*parts)
    output_root = out_dir.expanduser().resolve()
    try:
        candidate.expanduser().resolve().relative_to(output_root)
    except ValueError as exc:
        raise ValueError(
            f"Mesh path resolves outside output directory {out_dir}."
        ) from exc
    return candidate


def full_mesh_output_path(out_dir: Path, rel_path: str) -> Path:
    parts = normalized_parts(rel_path)
    return _safe_output_path(out_dir, [*parts[:-1], f"{parts[-1]}.obj"])


def legacy_mesh_output_path(out_dir: Path, rel_path: str) -> Path:
    parts = normalized_parts(rel_path)
    if len(parts) >= 2:
        legacy_parts = [parts[-2], f"{parts[-1]}.obj"]
    else:
        legacy_parts = [parts[-1], f"{parts[-1]}.obj"]
    return _safe_output_path(out_dir, legacy_parts)


def mesh_output_path(out_dir: Path, rel_path: str) -> Path:
    full_path = full_mesh_output_path(out_dir, rel_path)
    if full_path.exists():
        return full_path
    legacy_path = legacy_mesh_output_path(out_dir, rel_path)
    return legacy_path if legacy_path.exists() else full_path


def split_paths(cfg: dict[str, Any], split: str):
    data = np.load(cfg["split_file"])
    for path in sorted(data[split]):
        yield str(path)


def gt_npz_path(cfg: dict[str, Any], rel_path: str) -> Path:
    parts = normalized_parts(rel_path)
    data_root = Path(str(cfg["data_dir"])).expanduser()
    candidate = data_root.joinpath(*parts) / "on_surface_points.npz"
    try:
        candidate.resolve().relative_to(data_root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"Dataset path {rel_path!r} resolves outside data_dir."
        ) from exc
    return candidate


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    metrics = [
        "chamfer_l1",
        "chamfer_l2",
        "completeness2",
        "accuracy2",
        "normals",
        "fscore_002",
        "fscore_005",
        "fscore_010",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(
            f"total={len(rows)} ok={len(ok_rows)} failed={len(rows) - len(ok_rows)}\n"
        )
        for metric in metrics:
            values = np.array(
                [float(row[metric]) for row in ok_rows if metric in row],
                dtype=np.float64,
            )
            if values.size == 0:
                f.write(f"{metric}: nan\n")
                continue
            f.write(
                f"{metric}: mean={np.nanmean(values)} median={np.nanmedian(values)}\n"
            )


if __name__ == "__main__":
    main()
