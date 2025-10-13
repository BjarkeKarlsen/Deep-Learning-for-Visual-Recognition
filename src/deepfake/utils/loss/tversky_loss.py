import torch
import torch.nn as nn

class TverskyLoss(nn.Module):
    """
    Tversky loss for segmentation tasks.
    """
    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Raw logits of shape (N, 1, H, W)
            targets: Ground truth masks of shape (N, 1, H, W)
        """
        inputs = torch.sigmoid(inputs)

        # Flatten tensors
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)

        # True positives, false positives, and false negatives
        tp = (inputs_flat * targets_flat).sum()
        fp = ((1 - targets_flat) * inputs_flat).sum()
        fn = (targets_flat * (1 - inputs_flat)).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky