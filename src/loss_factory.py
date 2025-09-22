import torch
import torch.nn as nn

from .loss.bce_with_logits_loss import BCEWithLogitsLoss
from .loss.cross_entropy_loss import CrossEntropyLoss
from .loss.dice_loss import DiceLoss
from .loss.focal_loss import FocalLoss
from .loss.tversky_loss import TverskyLoss

class CombinedLoss(nn.Module):
    """
    Combined loss for multi-task learning (classification + segmentation).
    """
    def __init__(self, cls_loss, seg_loss, cls_weight=1.0, seg_weight=1.0):
        super().__init__()
        self.cls_loss = cls_loss
        self.seg_loss = seg_loss
        self.cls_weight = cls_weight
        self.seg_weight = seg_weight
    
    def forward(self, cls_outputs, seg_outputs, cls_targets, seg_targets):
        """
        Args:
            cls_outputs: Classification logits of shape (N, num_classes)
            seg_outputs: Segmentation logits of shape (N, 1, H, W)
            cls_targets: Classification labels of shape (N,)
            seg_targets: Segmentation masks of shape (N, 1, H, W)
        """
        cls_loss_val = self.cls_loss(cls_outputs, cls_targets)
        seg_loss_val = self.seg_loss(seg_outputs, seg_targets)
        
        total_loss = self.cls_weight * cls_loss_val + self.seg_weight * seg_loss_val
        return total_loss, cls_loss_val, seg_loss_val


class LossFactory:
    """
    Factory class to create different loss functions.
    """
    
    @staticmethod
    def create_classification_loss(loss_type='crossentropy', **kwargs):
        """Create classification loss function."""
        if loss_type.lower() == 'crossentropy':
            return CrossEntropyLoss(**kwargs)
        elif loss_type.lower() == 'focal':
            return FocalLoss(**kwargs)
        else:
            raise ValueError(f"Unknown classification loss type: {loss_type}")
    
    @staticmethod
    def create_segmentation_loss(loss_type='bce', **kwargs):
        """Create segmentation loss function."""
        if loss_type.lower() == 'bce':
            return BCEWithLogitsLoss(**kwargs)
        elif loss_type.lower() == 'dice':
            return DiceLoss(**kwargs)
        elif loss_type.lower() == 'tversky':
            return TverskyLoss(**kwargs)
        elif loss_type.lower() == 'focal':
            return FocalLoss(**kwargs)
        else:
            raise ValueError(f"Unknown segmentation loss type: {loss_type}")
    
    @staticmethod
    def create_combined_loss(cls_loss_type='crossentropy', seg_loss_type='bce', 
                           cls_weight=1.0, seg_weight=1.0, **kwargs):
        """Create combined loss for multi-task learning."""
        cls_loss = LossFactory.create_classification_loss(cls_loss_type, **kwargs.get('cls_kwargs', {}))
        seg_loss = LossFactory.create_segmentation_loss(seg_loss_type, **kwargs.get('seg_kwargs', {}))
        
        return CombinedLoss(cls_loss, seg_loss, cls_weight, seg_weight)

