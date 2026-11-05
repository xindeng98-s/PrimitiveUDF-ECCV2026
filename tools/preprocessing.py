"""Surface sampling and unsigned mesh-distance helpers."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

def sigma_name(sigma: float) -> str:
    return format(float(sigma), ".15g")

def derive_seed(base_seed: int, relative_path: str, stream: str) -> int:
    payload = f"{base_seed}\0{relative_path}\0{stream}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)

def _face_geometry(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    valid = np.isfinite(double_area) & (double_area > 0.0)
    if not np.any(valid):
        raise ValueError("mesh has no non-degenerate triangles")
    valid_faces = faces[valid]
    valid_double_area = double_area[valid]
    normals = cross[valid] / valid_double_area[:, None]
    probabilities = valid_double_area / valid_double_area.sum()
    return valid_faces, normals, probabilities

def sample_mesh_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    count: int,
    rng: np.random.Generator,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    if count <= 0:
        raise ValueError(f"surface sample count must be positive, got {count}")
    valid_faces, face_normals, probabilities = _face_geometry(vertices, faces)
    face_indices = rng.choice(len(valid_faces), size=count, replace=True, p=probabilities)
    triangles = vertices[valid_faces[face_indices]]

    root_u = np.sqrt(rng.random(count))
    v = rng.random(count)
    weights = np.stack(
        (
            1.0 - root_u,
            root_u * (1.0 - v),
            root_u * v,
        ),
        axis=1,
    )
    points = np.einsum("nij,ni->nj", triangles, weights)
    normals = face_normals[face_indices]
    return points.astype(dtype, copy=False), normals.astype(dtype, copy=False)

def build_raycasting_scene(
    vertices: np.ndarray,
    faces: np.ndarray,
    ray_threads: int,
):
    try:
        import open3d as o3d
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("open3d is required for exact mesh closest-point queries") from exc

    tensor_mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(vertices.astype(np.float32, copy=False)),
        o3d.core.Tensor(faces.astype(np.uint32, copy=False)),
    )
    scene = o3d.t.geometry.RaycastingScene(nthreads=ray_threads)
    scene.add_triangles(tensor_mesh)
    return scene

def closest_points_on_mesh(
    scene,
    query_points: np.ndarray,
    chunk_size: int,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    if chunk_size <= 0:
        raise ValueError(f"query chunk size must be positive, got {chunk_size}")
    import open3d as o3d

    query_float32 = np.asarray(query_points, dtype=np.float32)
    closest_parts: list[np.ndarray] = []
    for start in range(0, len(query_float32), chunk_size):
        query_chunk = o3d.core.Tensor(query_float32[start : start + chunk_size])
        answer = scene.compute_closest_points(query_chunk)
        closest_parts.append(answer["points"].numpy())
    nearest = np.concatenate(closest_parts, axis=0).astype(dtype, copy=False)
    stored_queries = query_float32.astype(dtype, copy=False)
    distances = np.linalg.norm(stored_queries - nearest, axis=1).astype(dtype, copy=False)
    return nearest, distances

def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
