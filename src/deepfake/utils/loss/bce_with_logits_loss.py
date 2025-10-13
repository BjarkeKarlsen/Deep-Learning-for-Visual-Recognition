import torch.nn as nn

class BCEWithLogitsLoss(nn.Module):
    """
    Binary cross-entropy with logits loss for segmentation tasks.
    """
    def __init__(self, weight=None, reduction='mean', pos_weight=None):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(weight=weight, reduction=reduction, pos_weight=pos_weight)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Raw logits of shape (N, 1, H, W)
            targets: Ground truth masks of shape (N, 1, H, W)
        """
        return self.loss_fn(inputs, targets)