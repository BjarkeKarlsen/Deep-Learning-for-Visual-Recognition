import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_channels: int, out_channels: int, *, kernel_size: int = 3) -> nn.Sequential:
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class SimpleUNet(nn.Module):
    """A lightweight U-Net suitable for 224x224 inputs."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_width: int = 32):
        super().__init__()
        self.enc1 = _conv_block(in_channels, base_width)
        self.enc2 = _conv_block(base_width, base_width * 2)
        self.enc3 = _conv_block(base_width * 2, base_width * 4)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = _conv_block(base_width * 4, base_width * 8)

        self.up3 = nn.ConvTranspose2d(base_width * 8, base_width * 4, kernel_size=2, stride=2)
        self.dec3 = _conv_block(base_width * 8, base_width * 4)

        self.up2 = nn.ConvTranspose2d(base_width * 4, base_width * 2, kernel_size=2, stride=2)
        self.dec2 = _conv_block(base_width * 4, base_width * 2)

        self.up1 = nn.ConvTranspose2d(base_width * 2, base_width, kernel_size=2, stride=2)
        self.dec1 = _conv_block(base_width * 2, base_width)

        self.head = nn.Conv2d(base_width, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.head(d1)
