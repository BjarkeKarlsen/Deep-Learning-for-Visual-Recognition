import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Two-conv residual block with optional channel projection and dropout."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            self.proj_bn = nn.BatchNorm2d(out_channels)
        else:
            self.proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.proj is not None:
            residual = self.proj(residual)
            residual = self.proj_bn(residual)

        out += residual
        out = self.activation(out)
        return out


class DecoderBlock(nn.Module):
    """Upsample then refine with a residual block."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = ResidualBlock(out_channels + skip_channels, out_channels, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class TamperSegmentationModel(nn.Module):
    """Residual U-Net variant tuned for tamper mask prediction."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_width: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()

        widths = [base_width, base_width * 2, base_width * 4, base_width * 8]

        self.enc1 = ResidualBlock(in_channels, widths[0], dropout)
        self.enc2 = ResidualBlock(widths[0], widths[1], dropout)
        self.enc3 = ResidualBlock(widths[1], widths[2], dropout)
        self.enc4 = ResidualBlock(widths[2], widths[3], dropout)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = ResidualBlock(widths[3], widths[3] * 2, dropout)

        self.dec4 = DecoderBlock(widths[3] * 2, widths[3], widths[3], dropout)
        self.dec3 = DecoderBlock(widths[3], widths[2], widths[2], dropout)
        self.dec2 = DecoderBlock(widths[2], widths[1], widths[1], dropout)
        self.dec1 = DecoderBlock(widths[1], widths[0], widths[0], dropout)

        self.head = nn.Conv2d(widths[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
