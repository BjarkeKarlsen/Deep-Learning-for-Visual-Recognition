from __future__ import annotations

import torch.nn as nn
from torchvision.models import (
    resnet18,
    resnet34,
    resnet50,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
)

from deepfake.config.schema import ModelConfig


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Two conv layers with batch norm to keep the shallow network stable."""

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class BaselineClassifier(nn.Module):
    """Compact CNN with three downsampling stages and a small MLP head."""

    def __init__(self, num_classes: int = 3, base_width: int = 32):
        super().__init__()

        # FEATURE EXTRACTOR DOWNSAMPLES AND EXPANDS CHANNEL DEPTH.
        widths = [base_width, base_width * 2, base_width * 4]
        self.stem = _conv_block(3, widths[0])
        self.stage1 = _conv_block(widths[0], widths[1])
        self.stage2 = _conv_block(widths[1], widths[2])

        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # LINEAR HEAD MAPS THE EMBEDDING TO THE TARGET CLASSES.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(widths[2], widths[2]),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(widths[2], num_classes),
        )

    def forward(self, x):
        """Return class logits for a batch of RGB images shaped (N, 3, H, W)."""
        x = self.stem(x)
        x = self.downsample(x)

        x = self.stage1(x)
        x = self.downsample(x)

        x = self.stage2(x)
        x = self.global_pool(x)

        logits = self.classifier(x)
        return logits


def _set_resnet_trainable_layers(model: nn.Module, trainable_layers: int) -> None:
    """Freeze all layers then unfreeze the requested number of deepest stages."""

    for param in model.parameters():
        param.requires_grad = False

    # Always keep the classification head trainable.
    for param in model.fc.parameters():
        param.requires_grad = True

    stages = [
        model.layer4,
        model.layer3,
        model.layer2,
        model.layer1,
        nn.Sequential(model.conv1, model.bn1),
    ]
    trainable_layers = max(0, min(trainable_layers, len(stages)))
    for stage in stages[:trainable_layers]:
        for param in stage.parameters():
            param.requires_grad = True


def _build_resnet_classifier(name: str, num_classes: int, pretrained: bool, trainable_layers: int) -> nn.Module:
    """Instantiate a torchvision ResNet backbone with a custom classification head."""

    name = name.lower()
    if name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
    elif name == "resnet34":
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        model = resnet34(weights=weights)
    elif name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
    else:
        raise ValueError(
            f"Unsupported ResNet backbone '{name}'. Choose from: resnet18, resnet34, resnet50."
        )

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    _set_resnet_trainable_layers(model, trainable_layers)
    return model


def build_classification_model(model_cfg: ModelConfig) -> nn.Module:
    """Return the configured classifier architecture (custom CNN or ResNet)."""

    backbone_cfg = getattr(model_cfg, "backbone", None)
    backbone_name = getattr(backbone_cfg, "name", "custom").lower() if backbone_cfg else "custom"

    if backbone_name == "custom":
        return BaselineClassifier(num_classes=model_cfg.num_classes, base_width=model_cfg.base_width)

    return _build_resnet_classifier(
        name=backbone_name,
        num_classes=model_cfg.num_classes,
        pretrained=getattr(backbone_cfg, "pretrained", False),
        trainable_layers=getattr(backbone_cfg, "trainable_layers", 0),
    )
