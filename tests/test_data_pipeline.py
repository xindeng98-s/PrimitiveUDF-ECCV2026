"""Exercise mesh preparation, shared loading and configured evaluation counts."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import trimesh

from datasets.pointcloud import build_dataset
from evaluation.checkpoints import compose_flat_config
from evaluation.mesh_metrics import load_gt_points
from tools.prepare_data import prepare_sample


class DataPipelineTests(unittest.TestCase):
    def test_mesh_to_shared_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mesh = trimesh.Trimesh(vertices=[[0,0,0],[2,0,0],[0,2,0]], faces=[[0,1,2]], process=False)
            mesh.export(root / 'mesh.ply')  # Open, non-watertight surface.
            settings = dict(compose_flat_config(config_name='scannet')['preparation'])
            settings.update(surface_points=100, query_points_per_sigma=80, query_chunk_size=30)
            output = root / 'data' / 'sample'
            prepare_sample(root/'mesh.ply', output, 'sample', settings)
            self.assertEqual(prepare_sample(root/'mesh.ply', output, 'sample', settings), 'skipped')
            for sigma in settings['sigmas']:
                with np.load(output/f'boundary_{sigma}_samples.npz') as data:
                    self.assertEqual(data['points'].shape, (80,3))
                    self.assertNotIn('sdf', data.files)
                    np.testing.assert_allclose(data['df'], np.linalg.norm(data['points']-data['nearest_points'], axis=1))
                    np.testing.assert_allclose(data['nearest_points'][:,2], 0, atol=1e-6)
            np.savez(root/'split.npz', train=['sample'], val=['/sample'], test=['/sample'])
            for name in ('scannet','matterport3d','shapenet_cars'):
                cfg = compose_flat_config(config_name=name)
                cfg.update(data_dir=str(root/'data'), split_file=str(root/'split.npz'),
                           num_points=20, num_sample_points_training=100)
                train = build_dataset(cfg, 'train')
                batch = train.collate_fn_pt2([train[0], train[0]])
                self.assertEqual(batch['off_df'].shape, (200,))
                self.assertEqual(batch['on_offsets'].tolist(), [20,40])
                self.assertEqual(build_dataset(cfg, 'val')[0]['off_df'].shape, (100,))
                self.assertEqual(build_dataset(cfg, 'test')[0]['on_surface_points'].shape, (20,3))
            changed = dict(settings, seed=settings['seed']+1)
            with self.assertRaises(FileExistsError):
                prepare_sample(root/'mesh.ply', output, 'sample', changed)

    def test_gt_count_independent_of_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'surface.npz'
            points = np.arange(90, dtype=np.float32).reshape(30,3)
            np.savez(path, points=points, normals=points+1)
            subset, normals = load_gt_points(path, target_points=10)
            self.assertEqual(len(subset), 10)
            np.testing.assert_array_equal(normals, subset+1)
            np.testing.assert_array_equal(subset, load_gt_points(path,target_points=10)[0])
            with self.assertRaises(ValueError):
                load_gt_points(path,target_points=31)
            # The public evaluator must forward the count even without filtering.
            from evaluation.mesh_metrics import evaluate_mesh_against_points
            with patch('evaluation.mesh_metrics.load_gt_points', side_effect=RuntimeError('stop')) as load:
                with self.assertRaisesRegex(RuntimeError, 'stop'):
                    evaluate_mesh_against_points('unused.obj', path, reference_points=10, reference_filter=False)
                self.assertEqual(load.call_args.kwargs['target_points'], 10)

    def test_eval_config(self):
        for name in ('scannet','matterport3d','shapenet_cars'):
            cfg = compose_flat_config(config_name=name)
            self.assertEqual(cfg['evaluation']['reference_points'],100000)
            self.assertEqual(cfg['evaluation']['sample_points'],100000 if name=='shapenet_cars' else 500000)
            self.assertEqual(cfg['dataset_module'],'datasets.pointcloud')
            self.assertEqual(cfg['preparation']['surface_points'],100000)
            self.assertEqual(cfg['preparation']['query_points_per_sigma'],100000)
