"""The model roster: names, output contracts and builders.

``contract`` is what ``model(x)`` returns: ``dict`` (TrackNetFast's heatmap,
offset and visibility heads), ``tuple`` (heatmap plus internal motion loss) or
``tensor`` (full-resolution probabilities).  Legacy models have no visibility
head, so eval scores them by heatmap peak and no numbers are reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch.nn as nn

from constants import HEIGHT, WIDTH
from models.others.TrackNetConvNeXt import TrackNetConvNeXt
from models.others.TrackNetLight import TrackNetAblation
from models.others.TrackNetMobile import TrackNetMobile
from models.others.TrackNetV2 import TrackNetV2
from models.others.TrackNetV4 import TrackNetV4
from models.TrackNetFast import TrackNetFast


@dataclass(frozen=True)
class ModelSpec:
    """One selectable architecture."""

    name: str
    contract: str
    build: Callable[..., nn.Module]
    note: str = ""


MODELS: dict[str, ModelSpec] = {
    "TrackNetFast": ModelSpec(
        name="TrackNetFast",
        contract="dict",
        build=lambda h, w, pretrained=True: TrackNetFast(pretrained=pretrained),
        note="MobileNetV3 encoder + FPN, stride-2 heatmap, offset and "
             "visibility heads. The reported model.",
    ),
    "Baseline_TrackNetV2": ModelSpec(
        name="Baseline_TrackNetV2",
        contract="tensor",
        build=lambda h, w, pretrained=True: TrackNetV2(h, w),
        note="PyTorch port of TrackNetV2, full-resolution heatmaps.",
    ),
    "TrackNetV4_TypeA": ModelSpec(
        name="TrackNetV4_TypeA",
        contract="tuple",
        build=lambda h, w, pretrained=True: TrackNetV4(
            h, w, fusion_layer_type="TypeA"),
        note="PyTorch port of TrackNetV4 with motion-fusion TypeA.",
    ),
    "TrackNetV4_TypeB": ModelSpec(
        name="TrackNetV4_TypeB",
        contract="tuple",
        build=lambda h, w, pretrained=True: TrackNetV4(
            h, w, fusion_layer_type="TypeB"),
        note="PyTorch port of TrackNetV4 with motion-fusion TypeB.",
    ),
    "TrackNet-light1.1": ModelSpec(
        name="TrackNet-light1.1",
        contract="tuple",
        build=lambda h, w, pretrained=True: TrackNetAblation(
            h, w, half_channels=True, depthwise=False),
        note="TrackNetV4 ablation: halved channels only.",
    ),
    "TrackNet-light1.2": ModelSpec(
        name="TrackNet-light1.2",
        contract="tuple",
        build=lambda h, w, pretrained=True: TrackNetAblation(
            h, w, half_channels=False, depthwise=True),
        note="TrackNetV4 ablation: depthwise-separable convolutions only.",
    ),
    "TrackNet-light2": ModelSpec(
        name="TrackNet-light2",
        contract="tuple",
        build=lambda h, w, pretrained=True: TrackNetAblation(
            h, w, half_channels=True, depthwise=True),
        note="TrackNetV4 ablation: halved channels and depthwise separable.",
    ),
    "TrackNet-Mobile": ModelSpec(
        name="TrackNet-Mobile",
        contract="tuple",
        build=lambda h, w, pretrained=True: TrackNetMobile(h, w),
        note="MobileNetV3-Small encoder, full-resolution heatmaps.",
    ),
    "TrackNetConvNeXt": ModelSpec(
        name="TrackNetConvNeXt",
        contract="tensor",
        build=lambda h, w, pretrained=True: TrackNetConvNeXt(h, w),
        note="ConvNeXt-Tiny backbone, full-resolution heatmaps.",
    ),
}

#: Ordered names for ``argparse`` ``choices``.
MODEL_NAMES = tuple(MODELS)


def build_model(name: str, height: int = HEIGHT, width: int = WIDTH,
                pretrained: bool = True) -> nn.Module:
    """Instantiate a registered model by name."""
    try:
        spec = MODELS[name]
    except KeyError:
        raise ValueError(
            f"Unknown model {name!r}. Available: {', '.join(MODEL_NAMES)}"
        ) from None
    return spec.build(height, width, pretrained)


def contract_of(name: str) -> str:
    """Return ``"dict"``, ``"tuple"`` or ``"tensor"`` for a model name."""
    return MODELS[name].contract
