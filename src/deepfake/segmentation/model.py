import torch
import torch.nn as nn
import torch.nn.functional as F

from deepfake.config.schema import ModelConfig
from deepfake.segmentation.backbones import build_backbone, ConvBlock


class DecoderBlock(nn.Module):
    """Bilinear upsampling followed by feature fusion and refinement."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.block = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class TamperSegmentationModel(nn.Module):
    """U-Net style decoder paired with either the legacy or a pretrained backbone."""

    def __init__(
        self,
        *,
        model_cfg: ModelConfig,
        in_channels: int = 3,
        out_channels: int = 1,
    ) -> None:
        super().__init__()

        backbone_cfg = getattr(model_cfg, "backbone", None)
        base_width = getattr(model_cfg, "base_width", 16)
        backbone = build_backbone(
            name=getattr(backbone_cfg, "name", "custom"),
            pretrained=getattr(backbone_cfg, "pretrained", False),
            trainable_layers=getattr(backbone_cfg, "trainable_layers", 4),
            in_channels=in_channels,
            base_width=base_width,
        )
        self.backbone = backbone

        encoder_channels = backbone.out_channels
        if len(encoder_channels) != 4:
            raise ValueError(f"Expected backbone to provide 4 skip feature channels, got {len(encoder_channels)}")
        c0, c1, c2, c3 = encoder_channels
        deep_channels = backbone.deep_channels

        self.bottleneck = ConvBlock(deep_channels, deep_channels)
        self.dec4 = DecoderBlock(deep_channels, c3, c3)
        self.dec3 = DecoderBlock(c3, c2, c2)
        self.dec2 = DecoderBlock(c2, c1, c1)
        self.dec1 = DecoderBlock(c1, c0, c0)
        self.head = nn.Conv2d(c0, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Produce tamper mask logits aligned with the input resolution."""
        input_size = x.shape[2:]
        skips, deep = self.backbone(x)
        if len(skips) != 4:
            raise ValueError(f"Expected 4 skip feature maps, received {len(skips)}")

        h = self.bottleneck(deep)
        h = self.dec4(h, skips[3])
        h = self.dec3(h, skips[2])
        h = self.dec2(h, skips[1])
        h = self.dec1(h, skips[0])

        logits = self.head(h)
        if logits.shape[2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return logits
