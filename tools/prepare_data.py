"""Prepare the shared unsigned-distance dataset from raw triangle meshes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import trimesh
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.pointcloud import sample_parts
from tools.preprocessing import (atomic_savez, build_raycasting_scene,
    closest_points_on_mesh, derive_seed, sample_mesh_surface, sigma_name)


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def prepare_sample(mesh_path, output, sample, settings, overwrite=False):
    mesh_path, output = Path(mesh_path), Path(output)
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)
    fingerprint = dict(settings, source_sha256=hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
                       sample=sample, format_version=1)
    marker = output / 'preparation.json'
    expected = ['on_surface_points.npz'] + [f'boundary_{sigma_name(s)}_samples.npz' for s in settings['sigmas']]
    if not overwrite and output.exists():
        if marker.exists() and json.loads(marker.read_text())['inputs'] == fingerprint and all((output / f).is_file() for f in expected):
            return 'skipped'
        raise FileExistsError(f'{output}: incomplete or different data; use --overwrite to regenerate')
    mesh = trimesh.load(str(mesh_path), force='mesh', process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all() or not len(faces):
        raise ValueError(f'{mesh_path}: expected a finite triangle mesh')
    center = (vertices.max(0) + vertices.min(0)) / 2
    scale = float(np.ptp(vertices, axis=0).max())
    if settings['normalization'] == 'none':
        center, scale = np.zeros(3), 1.0
    elif settings['normalization'] != 'unit_aabb':
        raise ValueError('normalization must be unit_aabb or none')
    if scale <= 0:
        raise ValueError(f'{mesh_path}: zero mesh extent')
    vertices = ((vertices - center) / scale).astype(np.float32)
    scene = build_raycasting_scene(vertices, faces, settings['ray_threads'])
    output.mkdir(parents=True, exist_ok=True)
    if marker.exists():
        marker.unlink()  # A failed overwrite must never look complete.
    rng = np.random.default_rng(derive_seed(settings['seed'], sample, 'surface'))
    points, normals = sample_mesh_surface(vertices, faces, settings['surface_points'], rng, np.float32)
    atomic_savez(output / 'on_surface_points.npz', points=points, normals=normals)
    for sigma in settings['sigmas']:
        rng = np.random.default_rng(derive_seed(settings['seed'], sample, 'boundary_' + sigma_name(sigma)))
        points, _ = sample_mesh_surface(vertices, faces, settings['query_points_per_sigma'], rng, np.float32)
        points = (points + rng.normal(0, sigma, points.shape)).astype(np.float32)
        nearest, df = closest_points_on_mesh(scene, points, settings['query_chunk_size'], np.float32)
        atomic_savez(output / f'boundary_{sigma_name(sigma)}_samples.npz',
                     points=points, nearest_points=nearest, df=df)
    metadata = dict(inputs=fingerprint, transform=dict(center=center.tolist(), scale=scale))
    temporary = marker.with_suffix('.tmp')
    temporary.write_text(json.dumps(metadata, indent=2) + '\n')
    temporary.replace(marker)
    return 'prepared'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-name', required=True, choices=['scannet', 'matterport3d', 'shapenet_cars'])
    parser.add_argument('--override', action='append', default=[], help='Hydra override; repeatable')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    with initialize_config_dir(config_dir=str(ROOT / 'configs_hydra'), version_base=None):
        cfg = OmegaConf.to_container(compose(config_name=args.config_name, overrides=args.override), resolve=True)
    settings = cfg['preparation']
    sigmas = settings['sigmas']
    if len(sigmas) != 3 or len(set(sigmas)) != 3 or not all(np.isfinite(s) and s > 0 for s in sigmas):
        raise ValueError('Configure three distinct, positive, finite sigmas')
    if set(sigmas) != set(cfg['data']['sample_std_dev']):
        raise ValueError('preparation.sigmas must match data.sample_std_dev')
    for key in ['surface_points', 'query_points_per_sigma', 'query_chunk_size', 'ray_threads']:
        if settings[key] < 1:
            raise ValueError(f'preparation.{key} must be positive')
    mesh_root = project_path(settings['mesh_root'])
    output_root = project_path(cfg['data']['data_dir'])
    with np.load(project_path(cfg['data']['split_file'])) as splits:
        samples = sorted({str(s).strip('/') for split in ['train', 'val', 'test'] for s in splits[split]})
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError('--limit must be positive')
        samples = samples[:args.limit]
    for i, sample in enumerate(samples):
        parts = sample_parts(sample)
        relative_mesh = settings['mesh_pattern'].format(sample=sample, house=parts[0], region=parts[-1])
        mesh_path = mesh_root.joinpath(*sample_parts(relative_mesh))
        status = prepare_sample(mesh_path, output_root.joinpath(*parts), sample, settings, args.overwrite)
        print(f'[{i+1}/{len(samples)}] {sample}: {status}', flush=True)


if __name__ == '__main__':
    main()
