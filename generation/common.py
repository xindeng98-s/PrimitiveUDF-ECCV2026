from __future__ import annotations

import time
from dataclasses import dataclass

import trimesh


@dataclass
class MeshGenerationStats:
    resolution: int
    bbox: float
    voxel_size: float
    total_grid_points: int
    queried_grid_points: int
    elapsed_seconds: float
    vertices: int
    faces: int
    field_type: str = "udf"
    dense_fallback: bool = False


def make_generation_stats(
    *,
    field_type: str,
    start: float,
    resolution: int,
    bbox: float,
    voxel_size: float,
    total_grid_points: int,
    queried_grid_points: int,
    vertices: int,
    faces: int,
    dense_fallback: bool = False,
) -> MeshGenerationStats:
    return MeshGenerationStats(
        field_type=field_type,
        resolution=int(resolution),
        bbox=float(bbox),
        voxel_size=float(voxel_size),
        total_grid_points=int(total_grid_points),
        queried_grid_points=int(queried_grid_points),
        elapsed_seconds=time.perf_counter() - start,
        vertices=int(vertices),
        faces=int(faces),
        dense_fallback=bool(dense_fallback),
    )


def clean_trimesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.process(validate=False)
    if hasattr(mesh, "remove_duplicate_faces"):
        mesh.remove_duplicate_faces()
    if hasattr(mesh, "remove_degenerate_faces"):
        mesh.remove_degenerate_faces()
    mesh.fill_holes()

    stable = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
    n_verts, n_faces = -1, -1
    for _ in range(10):
        if (n_verts, n_faces) == (len(stable.vertices), len(stable.faces)):
            break
        n_verts, n_faces = len(stable.vertices), len(stable.faces)
        stable = stable.process(validate=False)
        if hasattr(stable, "remove_duplicate_faces"):
            stable.remove_duplicate_faces()
        if hasattr(stable, "remove_degenerate_faces"):
            stable.remove_degenerate_faces()
        stable = trimesh.Trimesh(stable.vertices, stable.faces, process=False)
    return stable
