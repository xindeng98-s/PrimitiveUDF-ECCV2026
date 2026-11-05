from __future__ import annotations

from importlib import import_module

import torch
import torch.nn as nn


def build_model(opt):
    return D3F(
        opt,
        backbone=build_component(opt, "backbone"),
        head=build_component(opt, "head"),
    )


def build_component(opt, component_name):
    module_name = opt.get(f"{component_name}_module")
    builder_name = opt.get(f"{component_name}_builder")
    if not module_name or not builder_name:
        raise ValueError(f"model.{component_name}_module and model.{component_name}_builder are required.")

    module = import_module(str(module_name))
    if not hasattr(module, str(builder_name)):
        raise ValueError(f"{module_name} must define {builder_name}(cfg).")
    return getattr(module, str(builder_name))(opt)


class D3F(nn.Module):
    def __init__(self, opt, backbone, head):
        super().__init__()
        self.opt = opt
        self.backbone = backbone
        self.task = head
        self.head = head

    def forward(self, batch):
        device = next(self.parameters()).device

        off_surface_points = batch["off_surface_points"].to(device, non_blocking=True)
        off_offsets = batch["off_offsets"].to(device, non_blocking=True).to(torch.long)

        data_dict = {
            "coord": batch["on_surface_points"].to(device, non_blocking=True),
            "feat": batch["feature"].to(device, non_blocking=True) if self.opt["in_dim"] == 6 else None,
            "offset": batch["on_offsets"].to(device, non_blocking=True).to(torch.long),
        }

        points = self.backbone(data_dict)
        return self.task(points, off_surface_points, off_offsets)
