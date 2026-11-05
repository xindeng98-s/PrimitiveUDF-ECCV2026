from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
import trimesh

from .common import MeshGenerationStats, clean_trimesh, make_generation_stats

try:
    from networks.custom_mc import udf_mc_lewiner
except Exception:  # pragma: no cover - exercised in runtime env
    try:
        # Also accept a separately installed marching-cubes extension.
        from _marching_cubes_lewiner import udf_mc_lewiner
    except Exception as exc:
        raise ImportError(
            "Failed to import custom UDF marching cubes. Build it with "
            "`cd networks/custom_mc && python setup.py build_ext --inplace`."
        ) from exc


class MeshUDFGenerator:
    """Extract a surface from the PrimitiveUDF model."""

    field_type = "udf"

    def __init__(
        self,
        model: torch.nn.Module,
        cfg: Mapping[str, Any],
        *,
        device: torch.device,
        resolution: int | None = None,
        bbox: float | None = None,
        points_batch_size: int | None = None,
        query_margin_voxels: int = 4,
        filter_denominator: float | None = None,
    ) -> None:
        self.model = model.to(device).eval()
        self.cfg = cfg
        self.device = device
        self.resolution = int(resolution or cfg.get("gen_res", 256))
        self.bbox = float(bbox or cfg.get("gen_bbox", 0.55))
        self.points_batch_size = int(
            points_batch_size or cfg.get("num_sample_points_generation", 50000)
        )
        self.query_margin_voxels = int(query_margin_voxels)
        self.filter_denominator = float(filter_denominator or cfg.get("filter", 3.0))
        self.last_stats: MeshGenerationStats | None = None

        for param in self.model.parameters():
            param.requires_grad_(False)

        p = torch.linspace(-self.bbox, self.bbox, self.resolution)
        try:
            px, py, pz = torch.meshgrid(p, p, p, indexing="ij")
        except TypeError:
            px, py, pz = torch.meshgrid([p, p, p])
        self.grid_points = torch.stack([px, py, pz], dim=-1).reshape(-1, 3)

    def generate_mesh(self, batch: Mapping[str, Any]) -> trimesh.Trimesh | None:
        start = time.perf_counter()
        data_dict = self._encoder_input(batch)

        with torch.no_grad():
            encoded_points = self.model.backbone(data_dict)

        query_mask = self.build_query_mask_from_input_points(data_dict["coord"])
        query_points = self.grid_points[query_mask]
        if query_points.numel() == 0:
            self.last_stats = self._stats(start, 0, 0, 0)
            return None

        head = self.model.task
        head_state = encoded_points
        build_bank = getattr(head, "build_bank", None)
        if build_bank is not None:
            with torch.no_grad():
                head_state = build_bank(encoded_points)
            head = head.predict

        dist, direction = self._query_grid(head, head_state, query_points)
        dense_dist, dense_direction = self._scatter_to_dense(
            query_mask, dist, direction
        )

        voxel_size = self._voxel_size()
        try:
            verts, faces, _, _ = udf_mc_lewiner(
                dense_dist.numpy(),
                dense_direction.numpy(),
                spacing=[voxel_size] * 3,
            )
        except RuntimeError:
            self.last_stats = self._stats(start, int(query_points.shape[0]), 0, 0)
            return None

        if verts.shape[0] < 3 or faces.shape[0] < 1:
            self.last_stats = self._stats(
                start,
                int(query_points.shape[0]),
                int(verts.shape[0]),
                int(faces.shape[0]),
            )
            return None

        verts = verts - self.bbox
        pred_df_verts = self._query_vertex_distance(head, head_state, verts)
        keep = np.max(pred_df_verts[faces], axis=1)[:, 0] < (
            voxel_size / self.filter_denominator
        )
        filtered_faces = faces[keep]
        if filtered_faces.shape[0] < 1:
            self.last_stats = self._stats(
                start, int(query_points.shape[0]), int(verts.shape[0]), 0
            )
            return None

        mesh = clean_trimesh(trimesh.Trimesh(verts, filtered_faces, process=False))
        self.last_stats = self._stats(
            start, int(query_points.shape[0]), len(mesh.vertices), len(mesh.faces)
        )
        return mesh

    def build_query_mask_from_input_points(
        self, on_surface_points: torch.Tensor
    ) -> torch.Tensor:
        n = self.resolution
        voxel_size = self._voxel_size()
        pts = on_surface_points.detach().to(self.device)
        idx = torch.round((pts + self.bbox) / voxel_size).long().clamp(0, n - 1)

        occ = torch.zeros((1, 1, n, n, n), dtype=torch.float32, device=self.device)
        occ[0, 0, idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0

        margin = self.query_margin_voxels
        query_mask = (
            F.max_pool3d(occ, kernel_size=2 * margin + 1, stride=1, padding=margin) > 0
        )
        return query_mask[0, 0].reshape(-1).detach().cpu()

    def _encoder_input(
        self, batch: Mapping[str, Any]
    ) -> dict[str, torch.Tensor | None]:
        return {
            "coord": batch["on_surface_points"].to(self.device, non_blocking=True),
            "feat": (
                batch["feature"].to(self.device, non_blocking=True)
                if int(self.cfg.get("in_dim", 3)) == 6
                else None
            ),
            "offset": batch["on_offsets"]
            .to(self.device, non_blocking=True)
            .to(torch.long),
        }

    def _query_grid(
        self, head: Any, head_state: Any, query_points: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred_dist = []
        pred_direction = []
        for chunk in torch.split(query_points, self.points_batch_size):
            samples = (
                chunk.to(self.device, non_blocking=True).detach().requires_grad_(True)
            )
            offsets = torch.tensor([len(samples)], dtype=torch.long, device=self.device)
            df_pred, _ = head(head_state, samples, offsets)
            df_pred = df_pred.reshape(-1).abs()

            gradients = torch.autograd.grad(
                outputs=df_pred,
                inputs=samples,
                grad_outputs=torch.ones_like(df_pred),
                create_graph=False,
                retain_graph=False,
                only_inputs=True,
            )[0]
            pred_dist.append(df_pred.detach().cpu())
            pred_direction.append(
                (-F.normalize(gradients, dim=1, eps=1e-6)).detach().cpu()
            )

        return torch.cat(pred_dist, dim=0), torch.cat(pred_direction, dim=0)

    def _scatter_to_dense(
        self,
        query_mask: torch.Tensor,
        pred_dist: torch.Tensor,
        pred_direction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_total = int(self.grid_points.shape[0])
        fill_dist = float(self._voxel_size() * (self.query_margin_voxels + 2))
        dense_dist = torch.full((n_total,), fill_dist, dtype=torch.float32)
        dense_direction = torch.zeros((n_total, 3), dtype=torch.float32)
        dense_direction[:, 0] = 1.0
        dense_dist[query_mask] = pred_dist.float()
        dense_direction[query_mask] = pred_direction.float()
        n = self.resolution
        return dense_dist.reshape(n, n, n), dense_direction.reshape(n, n, n, 3)

    def _query_vertex_distance(
        self, head: Any, head_state: Any, verts: np.ndarray
    ) -> np.ndarray:
        verts_t = torch.from_numpy(verts).float().to(self.device)
        pred = []
        with torch.no_grad():
            for chunk in torch.split(verts_t, self.points_batch_size):
                offsets = torch.tensor(
                    [len(chunk)], dtype=torch.long, device=self.device
                )
                df_pred, _ = head(head_state, chunk, offsets)
                pred.append(df_pred.reshape(-1, 1).abs())
        return torch.cat(pred, dim=0).cpu().numpy()

    def _voxel_size(self) -> float:
        return self.bbox * 2.0 / float(self.resolution - 1)

    def _stats(
        self, start: float, queried: int, vertices: int, faces: int
    ) -> MeshGenerationStats:
        return make_generation_stats(
            field_type=self.field_type,
            start=start,
            resolution=self.resolution,
            bbox=self.bbox,
            voxel_size=self._voxel_size(),
            total_grid_points=int(self.grid_points.shape[0]),
            queried_grid_points=int(queried),
            vertices=int(vertices),
            faces=int(faces),
        )
