import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    """
    Dice loss for segmentation tasks.
    """
    def __init__(self, smooth=1.0):
        super().__init__()
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

        intersection = (inputs_flat * targets_flat).sum()
        dice = (2. * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1 - dice