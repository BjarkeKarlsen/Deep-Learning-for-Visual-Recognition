
import torch
import torch.nn as nn

# Metrics
def iou_metric(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> float:
    """
    Computes Intersection over Union (IoU) for binary segmentation.
    
    Args:
        preds: Predicted binary masks, shape (B, 1, H, W) or (B, H, W), values 0 or 1.
        targets: Ground-truth binary masks, same shape as preds.
        eps: Small epsilon to avoid division by zero.
        
    Returns:
        Mean IoU over the batch as a float.
    """
    # Ensure shape (B, H, W)
    if preds.ndim == 4 and preds.size(1) == 1:
        preds = preds.squeeze(1)
    if targets.ndim == 4 and targets.size(1) == 1:
        targets = targets.squeeze(1)
    
    # Flatten per-sample
    preds_flat = preds.view(preds.size(0), -1).float()
    targets_flat = targets.view(targets.size(0), -1).float()
    
    # Intersection and union
    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1) - intersection
    
    # Compute IoU, avoid divide by zero
    iou = (intersection + eps) / (union + eps)
    
    # Return average over batch
    return float(iou.mean().clamp(0.0, 1.0))


def dice_metric(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> float:
    """
    Computes Dice coefficient for binary segmentation.
    
    Args:
        preds: Predicted binary masks, shape (B, 1, H, W) or (B, H, W), values 0 or 1.
        targets: Ground-truth binary masks, same shape as preds.
        eps: Small epsilon to avoid division by zero.
        
    Returns:
        Mean Dice coefficient over the batch as a float.
    """
    # Ensure shape (B, H, W)
    if preds.ndim == 4 and preds.size(1) == 1:
        preds = preds.squeeze(1)
    if targets.ndim == 4 and targets.size(1) == 1:
        targets = targets.squeeze(1)
    
    # Flatten per-sample
    preds_flat = preds.view(preds.size(0), -1).float()
    targets_flat = targets.view(targets.size(0), -1).float()
    
    # Intersection
    intersection = (preds_flat * targets_flat).sum(dim=1)
    dice = (2. * intersection + eps) / (preds_flat.sum(dim=1) + targets_flat.sum(dim=1) + eps)
    
    # Return average over batch
    return float(dice.mean().clamp(0.0, 1.0))