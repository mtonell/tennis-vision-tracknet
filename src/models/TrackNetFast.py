"""TrackNetFast: a fast 3-frame ball tracker for 288x512 input.

MobileNetV3-Large encoder (taps at /2../32), a small FPN neck, and a
CenterNet-style head.  The stem takes 9 RGB channels (3 frames) plus two
luma-difference channels.

``model(x)`` with ``x`` of shape (B, 9, H, W) in [0, 1] returns a dict of
``heatmap``, ``offset`` and ``visibility`` logits; see :func:`decode_outputs`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_LUMA_WEIGHTS = (0.299, 0.587, 0.114)


def _conv_bn_act(in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, groups: int = 1) -> nn.Sequential:
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                  groups=groups, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True),
    )


def _depthwise_separable(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        _conv_bn_act(in_channels, in_channels, 3, 1, groups=in_channels),
        _conv_bn_act(in_channels, out_channels, 1),
    )


class MobileNetV3Encoder(nn.Module):
    """MobileNetV3-Large with taps at /2, /4, /8, /16 and /32.

    The stem is rebuilt for ``in_channels`` from the pretrained 3-channel
    weights: RGB filters are replicated per frame and divided by 3, extra
    channels are zeroed.
    """

    def __init__(self, in_channels: int = 11, pretrained: bool = True,
                 extra_std: float = 0.0):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        features = mobilenet_v3_large(weights=weights).features

        old_conv = features[0][0]
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            weight = old_conv.weight
            new_conv.weight[:, :9].copy_(weight.repeat(1, 3, 1, 1) / 3.0)
            if in_channels > 9:
                if extra_std > 0:
                    # Non-zero init so extra channels get a gradient at step 0.
                    nn.init.normal_(new_conv.weight[:, 9:], std=extra_std)
                else:
                    new_conv.weight[:, 9:].zero_()
        features[0][0] = new_conv

        self.stem = features[0]
        self.stage0 = nn.Sequential(*features[1:2])    # /2,  16 ch
        self.stage1 = nn.Sequential(*features[2:4])    # /4,  24 ch
        self.stage2 = nn.Sequential(*features[4:7])    # /8,  40 ch
        self.stage3 = nn.Sequential(*features[7:13])   # /16, 112 ch
        self.stage4 = nn.Sequential(*features[13:17])  # /32, 960 ch

        self.out_channels = self._infer_channels(in_channels)

    def _infer_channels(self, in_channels: int):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 64, 64)
            channels = [out.shape[1] for out in self.forward(dummy)]
        self.train(was_training)
        return channels

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        f2 = self.stage0(x)
        f4 = self.stage1(f2)
        f8 = self.stage2(f4)
        f16 = self.stage3(f8)
        f32 = self.stage4(f16)
        return f2, f4, f8, f16, f32


class FPN(nn.Module):
    """Minimal top-down FPN: 1x1 laterals, nearest upsample + add, 3x3 smooth."""

    def __init__(self, in_channels: list[int], out_channels: int = 64):
        super().__init__()
        self.laterals = nn.ModuleList(
            nn.Conv2d(channels, out_channels, 1) for channels in in_channels
        )
        self.output_conv = _depthwise_separable(out_channels, out_channels)

    def forward(self, features):
        laterals = [lateral(f) for lateral, f in zip(self.laterals, features)]
        merged = laterals[-1]
        for index in range(len(laterals) - 2, -1, -1):
            merged = laterals[index] + F.interpolate(
                merged, size=laterals[index].shape[-2:], mode="nearest"
            )
        return self.output_conv(merged)


class TrackNetFast(nn.Module):
    """Fast 3-in / 3-out ball tracker.

    Args:
        pretrained: load ImageNet weights for MobileNetV3-Large.
        use_motion_diffs: add the two luma-difference channels to the stem.
        fpn_channels: width of the FPN neck and head.
    """

    # Marks the dict-output contract; eval.py keys off this.
    outputs_dict = True

    def __init__(self, pretrained: bool = True,
                 use_motion_diffs: bool = True, fpn_channels: int = 64):
        super().__init__()
        self.use_motion_diffs = use_motion_diffs
        self.in_channels = 11 if use_motion_diffs else 9

        self.encoder = MobileNetV3Encoder(self.in_channels, pretrained=pretrained)
        taps = self.encoder.out_channels
        self.neck = FPN(taps, fpn_channels)

        self.heatmap_head = nn.Conv2d(fpn_channels, 3, 1)
        self.offset_head = nn.Conv2d(fpn_channels, 3 * 2, 1)
        # Visibility comes from each frame's heatmap peak, so it stays
        # per-frame; a pooled head would return one shared vector.
        self.visibility_scale = nn.Parameter(torch.tensor(1.0))
        self.visibility_bias = nn.Parameter(torch.tensor(0.0))

        # Background-heavy start: early on almost every cell is negative.
        nn.init.constant_(self.heatmap_head.bias, -4.0)

        self.register_buffer(
            "rgb_mean", torch.tensor(IMAGENET_MEAN * 3).view(9, 1, 1)
        )
        self.register_buffer(
            "rgb_std", torch.tensor(IMAGENET_STD * 3).view(9, 1, 1)
        )
        self.register_buffer(
            "luma_weights", torch.tensor(_LUMA_WEIGHTS).view(1, 3, 1, 1, 1)
        )

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 9:
            raise ValueError(
                f"TrackNetFast expects (B, 9, H, W) input, got {tuple(x.shape)}"
            )
        batch, _, height, width = x.shape
        frames = x.reshape(batch, 3, 3, height, width)  # B, T, C, H, W

        rgb = frames.reshape(batch, 9, height, width)
        rgb = (rgb - self.rgb_mean.to(rgb.dtype)) / self.rgb_std.to(rgb.dtype)

        if not self.use_motion_diffs:
            return rgb

        luma = (frames * self.luma_weights.to(frames.dtype)).sum(dim=2)
        prev_diff = (luma[:, 1] - luma[:, 0]).abs().unsqueeze(1)   # for frame t-1
        curr_diff = (luma[:, 2] - luma[:, 1]).abs().unsqueeze(1)   # for frame t
        return torch.cat([rgb, prev_diff, curr_diff], dim=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(self._prepare_input(x))
        neck = self.neck(features)

        heatmap = self.heatmap_head(neck)
        batch, _, height, width = heatmap.shape
        offset = self.offset_head(neck).view(batch, 3, 2, height, width)
        visibility = heatmap.amax(dim=(2, 3)) * self.visibility_scale + self.visibility_bias
        return {"heatmap": heatmap, "offset": offset, "visibility": visibility}


@torch.no_grad()
def decode_outputs(outputs: dict[str, torch.Tensor], height: int,
                   width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode an output dict to pixel coordinates.

    Returns ``(coords, visible)`` with ``coords`` of shape (B, 3, 2) in
    input-image pixels.
    """
    heatmap = outputs["heatmap"].float().sigmoid()
    batch, frames, heat_h, heat_w = heatmap.shape
    flat_index = heatmap.flatten(2).argmax(dim=-1).view(batch, frames)

    offset = outputs["offset"].float().sigmoid().flatten(3)  # B, T, 2, H*W
    gather_index = flat_index.unsqueeze(-1)
    offset_x = offset[:, :, 0].gather(2, gather_index).squeeze(-1)
    offset_y = offset[:, :, 1].gather(2, gather_index).squeeze(-1)

    cell_x = (flat_index % heat_w).float()
    cell_y = torch.div(flat_index, heat_w, rounding_mode="floor").float()

    stride_x = width / heat_w
    stride_y = height / heat_h
    x = (cell_x + offset_x) * stride_x
    y = (cell_y + offset_y) * stride_y

    visible = outputs["visibility"].float().sigmoid() > 0.5
    return torch.stack([x, y], dim=-1), visible
