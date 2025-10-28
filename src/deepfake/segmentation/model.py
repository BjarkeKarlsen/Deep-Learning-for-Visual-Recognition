import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two plain conv layers with batch norm and ReLU activation."""

    def __init__(self, in_channels: int, out_channels: int):
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


class DecoderBlock(nn.Module):
    """Bilinear upsampling followed by feature fusion and refinement."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ):
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
    """Lightweight U-Net variant tuned for tamper mask prediction."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_width: int = 16,
    ):
        super().__init__()

        widths = [base_width, base_width * 2, base_width * 4, base_width * 8]

        # ENCODER STACK CAPTURES MULTI-SCALE FEATURES.
        self.enc1 = ConvBlock(in_channels, widths[0])
        self.enc2 = ConvBlock(widths[0], widths[1])
        self.enc3 = ConvBlock(widths[1], widths[2])
        self.enc4 = ConvBlock(widths[2], widths[3])

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = ConvBlock(widths[3], widths[3] * 2)

        # DECODER REBUILDS RESOLUTION WITH SKIP CONNECTIONS.
        self.dec4 = DecoderBlock(widths[3] * 2, widths[3], widths[3])
        self.dec3 = DecoderBlock(widths[3], widths[2], widths[2])
        self.dec2 = DecoderBlock(widths[2], widths[1], widths[1])
        self.dec1 = DecoderBlock(widths[1], widths[0], widths[0])

        # FINAL CONV OUTPUTS A SINGLE-CHANNEL MASK LOGIT.
        self.head = nn.Conv2d(widths[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Produce a single-channel tamper mask logits tensor aligned with the input resolution."""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        bottleneck = self.bottleneck(self.pool(e4))

        d4 = self.dec4(bottleneck, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        return self.head(d1)
