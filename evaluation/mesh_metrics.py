from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

try:
    from pykdtree.kdtree import KDTree
except Exception:  # pragma: no cover - fallback depends on env
    KDTree = None
    from scipy.spatial import cKDTree

try:
    import pymeshlab
except Exception:  # pragma: no cover - pymeshlab is optional
    pymeshlab = None


@dataclass
class MeshEvalResult:
    completeness2: float
    accuracy2: float
    chamfer_l1: float
    chamfer_l2: float
    normals: float
    fscore_002: float
    fscore_005: float
    fscore_010: float
    num_pred_vertices: int
    num_pred_faces: int
    num_gt_points: int
    num_eval_points: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_mesh_against_points(
    mesh_path: str | Path,
    gt_npz_path: str | Path,
    *,
    sample_points: int = 500000,
    reference_points: int = 10000,
    bbox_margin: float = 0.007,
    distance_threshold: float = 0.02,
    seed: int = 20,
    repair: bool = True,
    reference_filter: bool = True,
) -> MeshEvalResult:
    gt_points, gt_normals = load_gt_points(
        gt_npz_path,
        target_points=reference_points,
        seed=seed,
    )
    mesh = as_mesh(trimesh.load(mesh_path, process=False))
    if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"Empty predicted mesh: {mesh_path}")

    ref_points = None
    if reference_filter:
        rng = np.random.RandomState(seed)
        ref_idx = rng.choice(
            len(gt_points), min(reference_points, len(gt_points)), replace=False
        )
        ref_points = gt_points[ref_idx]

    filtered = filter_mesh_by_gt(
        mesh,
        gt_points,
        ref_points,
        bbox_margin=bbox_margin,
        distance_threshold=distance_threshold,
        repair=repair,
    )
    if len(filtered.vertices) == 0 or len(filtered.faces) == 0:
        raise ValueError(f"Predicted mesh has no faces after filtering: {mesh_path}")

    pointcloud_pred, face_idx = trimesh.sample.sample_surface(
        filtered, int(sample_points), seed=seed
    )
    normals_pred = np.asarray(filtered.face_normals[face_idx]).astype(np.float32)
    metrics = eval_pointcloud(
        pointcloud_pred.astype(np.float32),
        gt_points.astype(np.float32),
        normals_pred=normals_pred,
        normals_gt=gt_normals.astype(np.float32) if gt_normals is not None else None,
    )
    return MeshEvalResult(
        completeness2=float(metrics[0]),
        accuracy2=float(metrics[1]),
        chamfer_l1=float(metrics[2]),
        chamfer_l2=float(metrics[3]),
        normals=float(metrics[4]),
        fscore_002=float(metrics[5]),
        fscore_005=float(metrics[6]),
        fscore_010=float(metrics[7]),
        num_pred_vertices=int(len(filtered.vertices)),
        num_pred_faces=int(len(filtered.faces)),
        num_gt_points=int(len(gt_points)),
        num_eval_points=int(sample_points),
    )


def load_gt_points(
    path: str | Path,
    *,
    target_points: int | None = None,
    seed: int = 20,
) -> tuple[np.ndarray, np.ndarray | None]:
    path = Path(path)
    with np.load(path) as data:
        points_key = "points" if "points" in data else "point_cloud"
        normals_key = "normals" if "normals" in data else "normal" if "normal" in data else None
        points = np.asarray(data[points_key]).astype(np.float32)
        normals = np.asarray(data[normals_key]).astype(np.float32) if normals_key is not None else None

    if target_points is None or target_points == len(points):
        return points, normals
    if target_points < 1 or target_points > len(points):
        raise ValueError(f"Requested {target_points} GT points but {path} has {len(points)} surface points")
    indices = np.random.RandomState(seed).choice(len(points), target_points, replace=False)
    return points[indices], None if normals is None else normals[indices]


def filter_mesh_by_gt(
    mesh: trimesh.Trimesh,
    gt_points: np.ndarray,
    reference_points: np.ndarray | None,
    *,
    bbox_margin: float,
    distance_threshold: float,
    repair: bool,
) -> trimesh.Trimesh:
    pred_verts = np.asarray(mesh.vertices)
    pred_faces = np.asarray(mesh.faces)
    gt_min = gt_points.min(axis=0) - float(bbox_margin)
    gt_max = gt_points.max(axis=0) + float(bbox_margin)
    bbox_mask = np.all((pred_verts >= gt_min) & (pred_verts <= gt_max), axis=1)
    keep_vertex = bbox_mask
    if reference_points is not None:
        dists, _ = distance_p2p(
            pred_verts.astype(np.float32),
            reference_points.astype(np.float32),
            None,
            None,
        )
        keep_vertex = keep_vertex & (dists < float(distance_threshold))
    keep_face = keep_vertex[pred_faces].all(axis=1)
    filtered_faces = pred_faces[keep_face]
    filtered = trimesh.Trimesh(vertices=pred_verts, faces=filtered_faces, process=False)
    if repair:
        filtered = repair_mesh(filtered)
    return filtered


def repair_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    if pymeshlab is not None and len(mesh.vertices) > 0 and len(mesh.faces) > 0:
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(mesh.vertices), face_matrix=np.asarray(mesh.faces)), "pred")
        try:
            ms.meshing_repair_non_manifold_edges()
            ms.meshing_close_holes()
        except Exception:
            pass
        processed = ms.current_mesh()
        out = trimesh.Trimesh(
            vertices=processed.vertex_matrix().copy(),
            faces=processed.face_matrix().copy(),
            process=False,
        )
        ms.clear()
        return out

    out = mesh.process(validate=False)
    if hasattr(out, "remove_duplicate_faces"):
        out.remove_duplicate_faces()
    if hasattr(out, "remove_degenerate_faces"):
        out.remove_degenerate_faces()
    out.fill_holes()
    return trimesh.Trimesh(out.vertices, out.faces, process=False)


def eval_pointcloud(
    pointcloud_pred: np.ndarray,
    pointcloud_gt: np.ndarray,
    thresholds: np.ndarray = np.linspace(1.0 / 1000.0, 1.0, 1000),
    normals_pred: np.ndarray | None = None,
    normals_gt: np.ndarray | None = None,
) -> tuple[float, float, float, float, float, float, float, float]:
    completeness, completeness_normals = distance_p2p(pointcloud_gt, pointcloud_pred, normals_gt, normals_pred)
    recall = get_threshold_percentage(completeness, thresholds)
    completeness2 = np.square(completeness).mean()
    completeness = completeness.mean()

    accuracy, accuracy_normals = distance_p2p(pointcloud_pred, pointcloud_gt, normals_pred, normals_gt)
    precision = get_threshold_percentage(accuracy, thresholds)
    accuracy2 = np.square(accuracy).mean()
    accuracy = accuracy.mean()

    chamfer_l2 = 0.5 * completeness2 + 0.5 * accuracy2
    chamfer_l1 = 0.5 * completeness + 0.5 * accuracy
    f_scores = [
        safe_fscore(precision[i], recall[i])
        for i in range(len(precision))
    ]

    if normals_pred is not None and normals_gt is not None:
        normals_correctness = 0.5 * completeness_normals.mean() + 0.5 * accuracy_normals.mean()
    else:
        normals_correctness = np.nan

    return (
        completeness2,
        accuracy2,
        chamfer_l1,
        chamfer_l2,
        normals_correctness,
        f_scores[1],
        f_scores[4],
        f_scores[9],
    )


def distance_p2p(
    pointcloud_pred: np.ndarray,
    pointcloud_gt: np.ndarray,
    normals_pred: np.ndarray | None,
    normals_gt: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    if KDTree is not None:
        dist, idx = KDTree(pointcloud_gt).query(pointcloud_pred)
    else:
        dist, idx = cKDTree(pointcloud_gt).query(pointcloud_pred)

    if normals_pred is None or normals_gt is None:
        return dist, None

    normals_pred = safe_normalize(normals_pred)
    normals_gt = safe_normalize(normals_gt)
    normals_dot_product = np.abs((normals_gt[idx] * normals_pred).sum(axis=-1))
    return dist, normals_dot_product


def safe_normalize(normals: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(normals, axis=-1, keepdims=True)
    denom = np.where(denom < 1e-8, 1.0, denom)
    return np.nan_to_num(normals / denom, nan=0.0, posinf=0.0, neginf=0.0)


def get_threshold_percentage(dist: np.ndarray, thresholds: np.ndarray) -> list[float]:
    return [(dist <= t).mean() for t in thresholds]


def safe_fscore(precision: float, recall: float) -> float:
    denom = precision + recall
    return float(2.0 * precision * recall / denom) if denom > 0 else 0.0


def as_mesh(scene_or_mesh) -> trimesh.Trimesh | None:
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            return None
        return trimesh.util.concatenate(
            tuple(
                trimesh.Trimesh(vertices=g.vertices, faces=g.faces, process=False)
                for g in scene_or_mesh.geometry.values()
            )
        )
    if isinstance(scene_or_mesh, trimesh.Trimesh):
        return scene_or_mesh
    raise TypeError(f"Expected trimesh.Trimesh or trimesh.Scene, got {type(scene_or_mesh)!r}")
