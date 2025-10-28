import torch.nn as nn


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
