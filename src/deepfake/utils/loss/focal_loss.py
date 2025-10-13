import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiClassFocalLoss(nn.Module):
    """
    Focal loss for multi-class classification.
    Supports both gamma (focusing parameter) and alpha (class weighting).
    """
    def __init__(self, gamma: float = 2.0, alpha=None, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.weight = weight
        self.reduction = reduction
        
        # If alpha is provided as a single float, convert to tensor for weighting
        if isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha])
        elif isinstance(alpha, list):
            self.alpha = torch.tensor(alpha)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) logits
            targets: (N,) class indices
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        
        # Compute probabilities and pt
        pt = torch.exp(-ce_loss)
        
        # Apply alpha weighting if specified
        if self.alpha is not None:
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)
            # Select alpha values for each target class
            at = self.alpha.gather(0, targets) if self.alpha.numel() > 1 else self.alpha
            logpt = -ce_loss
            focal_loss = at * (1 - pt) ** self.gamma * (-logpt)
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss