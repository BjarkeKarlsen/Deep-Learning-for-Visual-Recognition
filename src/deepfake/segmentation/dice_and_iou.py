import torch
from typing import Dict


def dice_and_iou(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> Dict[str, torch.Tensor]:
    """Compute dataset-level overlap metrics for binary tamper masks."""
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()  # 0.5 aligns with the binary tamper-versus-background assumption
    targets = (targets > 0.5).float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)

    overlap = (preds * targets).sum(dim=(1, 2, 3))
    union_iou = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - overlap
    iou = (overlap + eps) / (union_iou + eps)
    return {"dice": dice.mean(), "iou": iou.mean()}