"""Validate the three release configurations and optionally bundled weights."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.checkpoints import compose_flat_config, load_model
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', action='store_true')
    args = parser.parse_args()
    for name in ('scannet', 'matterport3d', 'shapenet_cars'):
        cfg = compose_flat_config(config_name=name)
        assert cfg['head_module'] == 'heads.primitive_bank_v3'
        assert cfg['primitive_bank_neighbor_k'] == 16
        assert cfg['primitive_bank_keep_prims'] == 10
        assert not cfg['is_sdf']
        if args.checkpoints:
            load_model(cfg, str(Path(__file__).resolve().parents[1] / 'pretrained' / (name+'.ckpt')),
                       torch.device('cpu'), strict=True, validate_config=True)
        print(name + ': OK')
if __name__ == '__main__':
    main()
