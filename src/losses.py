"""Losses for TrackNetFast and the legacy heatmap models."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FastTrackNetLoss(nn.Module):
    """Loss for :class:`models.TrackNetFast.TrackNetFast`.

    Takes the model output dict and the full-resolution heatmap targets from
    ``TennisDataset`` (Gaussian blobs, all-zero for invisible frames).
    Combines a CenterNet focal loss on max-pooled targets, an L1 subpixel
    offset loss at the ground-truth cell, and BCE for visibility.
    """

    def __init__(self, offset_weight: float = 1.0,
                 visibility_weight: float = 0.1, gamma: float = 2.0):
        super().__init__()
        self.offset_weight = offset_weight
        self.visibility_weight = visibility_weight
        self.gamma = gamma

    def forward(self, outputs: dict[str, torch.Tensor], target: torch.Tensor):
        heatmap_logits = outputs["heatmap"].float()
        offsets_logits = outputs["offset"].float()
        visibility_logits = outputs["visibility"].float()
        target = target.float()

        batch, frames, heat_h, heat_w = heatmap_logits.shape

        heat_target = F.adaptive_max_pool2d(target, (heat_h, heat_w))
        probability = heatmap_logits.sigmoid().clamp(1e-4, 1.0 - 1e-4)

        positive = heat_target
        negative = (1.0 - heat_target) ** 4
        positive_loss = -((1.0 - probability) ** self.gamma) * torch.log(probability)
        negative_loss = -(probability ** 2) * torch.log(1.0 - probability)
        num_positive = positive.sum().clamp_min(1.0)
        heatmap_loss = (
            (positive_loss * positive).sum() + (negative_loss * negative).sum()
        ) / num_positive

        height, width = target.shape[-2:]
        coords, visible = self._target_coords(target)

        stride_x = width / heat_w
        stride_y = height / heat_h
        cell_x = torch.floor(coords[..., 0] / stride_x).long().clamp(0, heat_w - 1)
        cell_y = torch.floor(coords[..., 1] / stride_y).long().clamp(0, heat_h - 1)
        target_offset_x = coords[..., 0] / stride_x - cell_x.to(coords.dtype)
        target_offset_y = coords[..., 1] / stride_y - cell_y.to(coords.dtype)

        flat_index = (cell_y * heat_w + cell_x).reshape(batch, frames, 1)
        offsets = offsets_logits.sigmoid().flatten(3)  # B, T, 2, H*W
        predicted_x = offsets[:, :, 0].gather(2, flat_index).squeeze(-1)
        predicted_y = offsets[:, :, 1].gather(2, flat_index).squeeze(-1)

        offset_error = (predicted_x - target_offset_x).abs() + (
            predicted_y - target_offset_y
        ).abs()
        if visible.any():
            offset_loss = offset_error[visible].mean()
        else:
            offset_loss = heatmap_logits.sum() * 0.0

        visibility_target = visible.to(heatmap_logits.dtype)
        visibility_loss = F.binary_cross_entropy_with_logits(
            visibility_logits, visibility_target
        )

        total = (
            heatmap_loss
            + self.offset_weight * offset_loss
            + self.visibility_weight * visibility_loss
        )
        return total, {
            "heatmap": heatmap_loss.detach(),
            "offset": offset_loss.detach(),
            "visibility": visibility_loss.detach(),
        }

    @staticmethod
    def _target_coords(target: torch.Tensor):
        """Intensity-weighted centroid of each target heatmap."""
        batch, frames, height, width = target.shape
        mass = target.sum(dim=(2, 3)).clamp_min(1e-6)  # B, T

        xs = torch.arange(width, device=target.device, dtype=target.dtype)
        ys = torch.arange(height, device=target.device, dtype=target.dtype)
        coord_x = (target.sum(dim=2) * xs).sum(dim=-1) / mass
        coord_y = (target.sum(dim=3) * ys).sum(dim=-1) / mass

        coords = torch.stack([coord_x, coord_y], dim=-1)  # B, T, 2
        visible = target.amax(dim=(2, 3)) > 0
        return coords, visible


def model_loss(criterion, outputs, labels):
    """Apply ``criterion`` to any model output contract.

    Returns ``(loss, parts)``; ``parts`` is filled for the dict contract only.
    For ``tuple`` outputs the model's own motion loss is added, which is how
    the TrackNetV4 family expects to be trained.
    """
    if isinstance(outputs, dict):
        return criterion(outputs, labels)
    if isinstance(outputs, (tuple, list)):
        heatmaps, motion_loss = outputs
        return criterion(heatmaps.float(), labels) + motion_loss, {}
    return criterion(outputs.float(), labels), {}
