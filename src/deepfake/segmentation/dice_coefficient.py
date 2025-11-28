
import torch


def dice_coefficient(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Computes the Dice Coefficient over a batch:
      Inputs:
        logits:  Tensor of shape (N,1,H,W)
        targets: Tensor of shape (N,1,H,W)
    Returns:
        single scalar Dice score across all N×H×W pixels
    """
    # SIMPLE BINARY DICE METRIC USED DURING TRAIN AND VALIDATION LOOPS.
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float().view(-1)
    targs = (targets > 0.5).float().view(-1)
    intersection = (preds * targs).sum()
    union = preds.sum() + targs.sum()
    return (2.0 * intersection + eps) / (union + eps)


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Differentiable Dice loss that keeps small foreground regions influential."""
    probs = torch.sigmoid(logits)
    targets = targets.float()
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    denominator = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()
