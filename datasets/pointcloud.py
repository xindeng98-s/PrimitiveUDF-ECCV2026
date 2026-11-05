"""Shared point-cloud and unsigned-distance loader for all release datasets."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def sample_parts(path):
    parts = path.replace('\\', '/').strip('/').split('/')
    if any(p in ('', '.', '..') for p in parts):
        raise ValueError(f'Invalid sample path: {path!r}')
    return parts


def build_dataset(cfg, mode):
    return PointCloudDataset(cfg, mode)


class PointCloudDataset(Dataset):
    def __init__(self, cfg, mode):
        if mode not in ('train', 'val', 'test'):
            raise ValueError(f'Unknown split: {mode}')
        self.cfg, self.mode = cfg, mode
        self.root = Path(cfg['data_dir'])
        with np.load(cfg['split_file']) as splits:
            self.data = splits[mode].copy()
        self.noise = float(cfg['noise'])
        if not np.isfinite(self.noise) or self.noise < 0:
            raise ValueError('data.noise must be finite and non-negative')
        ratios = np.asarray(cfg['sample_ratio'], dtype=float)
        self.sigmas = cfg['sample_std_dev']
        if (len(ratios) != len(self.sigmas) or not np.isfinite(ratios).all()
                or np.any(ratios < 0) or not np.isclose(ratios.sum(), 1)):
            raise ValueError('sample_ratio must match sample_std_dev and sum to one')
        if cfg['num_points'] < 1 or cfg['num_sample_points_training'] < 1:
            raise ValueError('Point counts must be positive')
        counts = ratios * cfg['num_sample_points_training']
        self.counts = np.floor(counts).astype(int)
        remainder = cfg['num_sample_points_training'] - self.counts.sum()
        self.counts[np.argsort(-(counts-self.counts), kind='stable')[:remainder]] += 1

    def __len__(self):
        return len(self.data)

    def _sample_input(self, directory):
        with np.load(directory / 'on_surface_points.npz') as source:
            points = np.asarray(source['points'], dtype=np.float32)
            normals = np.asarray(source['normals'], dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or normals.shape != points.shape or not len(points):
            raise ValueError(f'{directory}: expected nonempty aligned points/normals [N,3]')
        rng = np.random if self.mode == 'train' else np.random.RandomState(0)
        count = self.cfg['num_points']
        indices = rng.choice(len(points), count, replace=count > len(points))
        points, normals = points[indices], normals[indices]
        if self.noise:
            points = points + self.noise * rng.randn(*points.shape).astype(np.float32)
        if not self.cfg.get('fixed_input', False):
            order = torch.randperm(len(points)).numpy()
            points, normals = points[order], normals[order]
        feature = points if self.cfg['in_dim'] == 3 else normals
        return points, feature

    def __getitem__(self, index):
        path = str(self.data[index])
        directory = self.root.joinpath(*sample_parts(path))
        points, feature = self._sample_input(directory)
        batch = dict(feature=torch.from_numpy(feature), on_surface_points=torch.from_numpy(points),
                     on_offsets=torch.tensor([len(points)]), path=path)
        if self.mode == 'test':
            return batch
        queries, distances, nearest = [], [], []
        for sigma, count in zip(self.sigmas, self.counts):
            source_path = directory / f'boundary_{float(sigma):.15g}_samples.npz'
            with np.load(source_path) as source:
                query = np.asarray(source['points'], dtype=np.float32)
                df = np.asarray(source['df'], dtype=np.float32).reshape(-1)
                near = np.asarray(source['nearest_points'], dtype=np.float32)
            if (query.ndim != 2 or query.shape[1] != 3 or near.shape != query.shape
                    or len(df) != len(query) or not len(query)):
                raise ValueError(f'{source_path}: inconsistent query/distance/nearest arrays')
            if not np.isfinite(df).all() or (df < 0).any():
                raise ValueError(f'{source_path}: df must be finite and unsigned')
            indices = torch.randint(len(query), (int(count),)).numpy()
            queries.append(query[indices])
            distances.append(df[indices])
            nearest.append(near[indices])
        batch.update(off_surface_points=torch.from_numpy(np.concatenate(queries)),
                     off_df=torch.from_numpy(np.concatenate(distances)),
                     moves=torch.from_numpy(np.concatenate(nearest)),
                     off_offsets=torch.tensor([sum(self.counts)]))
        return batch

    @staticmethod
    def collate_fn_pt2(batch):
        result = {}
        for key in batch[0]:
            values = [sample[key] for sample in batch]
            value = values if isinstance(values[0], str) else torch.cat(values)
            result[key] = value.cumsum(0) if key.endswith('_offsets') else value
        return result

    @staticmethod
    def worker_init_fn(worker_id):
        np.random.seed(torch.initial_seed() % (2**32))
