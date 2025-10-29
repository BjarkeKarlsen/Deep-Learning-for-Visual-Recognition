"""Backbone factory for tamper segmentation encoders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights


BackboneOutput = Tuple[List[torch.Tensor], torch.Tensor]


class BaseBackbone(nn.Module):
    """Base class returning skip features and a deep feature map."""

    def __init__(self) -> None:
        super().__init__()
        self.out_channels: List[int] = []
        self.deep_channels: int = 0

    def forward(self, x: torch.Tensor) -> BackboneOutput:  # pragma: no cover - interface only
        raise NotImplementedError


class SimpleBackbone(BaseBackbone):
    """Original lightweight encoder built from plain conv blocks."""

    def __init__(self, in_channels: int = 3, base_width: int = 16) -> None:
        super().__init__()
        widths = [base_width, base_width * 2, base_width * 4, base_width * 8]

        self.enc1 = ConvBlock(in_channels, widths[0])
        self.enc2 = ConvBlock(widths[0], widths[1])
        self.enc3 = ConvBlock(widths[1], widths[2])
        self.enc4 = ConvBlock(widths[2], widths[3])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.out_channels = widths
        self.deep_channels = widths[-1]

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        deep = self.pool(e4)
        return [e1, e2, e3, e4], deep


class ResNet34Backbone(BaseBackbone):
    """ResNet-34 backbone with optional ImageNet pretraining."""

    _STAGES = ("input_proj", "stem", "layer1", "layer2", "layer3")

    def __init__(self, *, pretrained: bool, trainable_layers: int, in_channels: int = 3) -> None:
        super().__init__()
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        resnet = resnet34(weights=weights)

        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Optional fusion of the deepest features to enrich context
        self.deep_merge = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.out_channels = [64, 64, 64, 128]
        self.deep_channels = 256

        self._freeze_stages(trainable_layers)

    def _freeze_stages(self, trainable_layers: int) -> None:
        trainable_layers = max(0, min(trainable_layers, len(self._STAGES)))
        stages_in_order = [
            self.input_proj,
            nn.Sequential(self.conv1, self.bn1),
            self.layer1,
            self.layer2,
            self.layer3,
        ]
        num_frozen = len(stages_in_order) - trainable_layers
        for stage in stages_in_order[:num_frozen]:
            for param in stage.parameters():
                param.requires_grad = False

        # layer4 is used only for context enrichment; keep it frozen unless explicitly requested
        if trainable_layers < len(self._STAGES):
            for param in self.layer4.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        input_spatial = x.shape[2:]
        skip1 = self.input_proj(x)  # H

        x = self.conv1(x)
        x = self.bn1(x)
        stem = self.relu(x)  # H/2

        x = self.maxpool(stem)  # H/4
        layer1 = self.layer1(x)  # H/4
        layer2 = self.layer2(layer1)  # H/8
        layer3 = self.layer3(layer2)  # H/16

        # Incorporate layer4 context while keeping resolution at H/16.
        deep = layer3
        layer4 = self.layer4(layer3)  # H/32
        layer4_up = F.interpolate(layer4, size=layer3.shape[2:], mode="bilinear", align_corners=False)
        deep = self.deep_merge(layer4_up + layer3)

        # Ensure skip maps stay aligned with input size hierarchy.
        skip2 = stem
        skip3 = layer1
        skip4 = layer2

        # Stem skip (H/2) needs interpolation to match input when decoder upsamples.
        if skip2.shape[2:] != (input_spatial[0] // 2, input_spatial[1] // 2):
            skip2 = F.interpolate(skip2, size=(input_spatial[0] // 2, input_spatial[1] // 2), mode="bilinear", align_corners=False)

        if skip3.shape[2:] != (input_spatial[0] // 4, input_spatial[1] // 4):
            skip3 = F.interpolate(skip3, size=(input_spatial[0] // 4, input_spatial[1] // 4), mode="bilinear", align_corners=False)

        if skip4.shape[2:] != (input_spatial[0] // 8, input_spatial[1] // 8):
            skip4 = F.interpolate(skip4, size=(input_spatial[0] // 8, input_spatial[1] // 8), mode="bilinear", align_corners=False)

        return [skip1, skip2, skip3, skip4], deep


def build_backbone(
    *,
    name: str,
    pretrained: bool,
    trainable_layers: int,
    in_channels: int,
    base_width: int,
) -> BaseBackbone:
    """Instantiate the requested backbone."""
    name = (name or "custom").lower()
    if name == "custom":
        return SimpleBackbone(in_channels=in_channels, base_width=base_width)
    if name == "resnet34":
        return ResNet34Backbone(pretrained=pretrained, trainable_layers=trainable_layers, in_channels=in_channels)

    raise ValueError(
        f"Unsupported backbone '{name}'. Available options: 'custom', 'resnet34'."
    )


class ConvBlock(nn.Module):
    """Two conv layers with batch norm and ReLU activation."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
