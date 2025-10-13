import torch.nn as nn
from deepfake.utils.loss.bce_with_logits_loss import BCEWithLogitsLoss
from deepfake.utils.loss.cross_entropy_loss   import CrossEntropyLoss
from deepfake.utils.loss.dice_loss            import DiceLoss
from deepfake.utils.loss.focal_loss           import MultiClassFocalLoss
from deepfake.utils.loss.tversky_loss         import TverskyLoss


class SegmentationLoss(nn.Module):
    """
    Wraps multiple segmentation sub-losses into one criterion.
    Returns weighted sum of all segmentation losses.
    """
    LOSS_MAP = {
        'bce':      BCEWithLogitsLoss,
        'dice':     DiceLoss,
        'tversky':  TverskyLoss,
        'focal':    MultiClassFocalLoss,
    }

    def __init__(self,
                 types: list[str],
                 weights: list[float],
                 global_kwargs: dict = None,
                 per_kwargs: list[dict] = None):
        super().__init__()
        if weights is None:
            weights = [1.0] * len(types)
        assert len(types) == len(weights), "Types and weights must align"
        self.losses = nn.ModuleList()
        self.weights = weights

        for i, key in enumerate(types):
            cls = self.LOSS_MAP.get(key.lower())
            if cls is None:
                raise ValueError(f"Unknown segmentation loss type: {key}")
            kwargs = {}
            if global_kwargs:
                kwargs.update(global_kwargs)
            if per_kwargs and i < len(per_kwargs) and per_kwargs[i]:
                kwargs.update(per_kwargs[i])
            self.losses.append(cls(**kwargs))

    def forward(self, preds, masks):
        loss_vals = []
        for w, loss_fn in zip(self.weights, self.losses):
            raw = loss_fn(preds, masks)
            loss_vals.append(raw)
        total = sum(w * v for w, v in zip(self.weights, loss_vals))
        return total, loss_vals

class ClassificationLoss(nn.Module):
    """
    Wraps multiple classification sub-losses into one criterion.
    Returns weighted sum of all classification losses.
    """
    LOSS_MAP = {
        'crossentropy': CrossEntropyLoss,
        'focal':         MultiClassFocalLoss,
    }

    def __init__(self,
                 types: list[str],
                 weights: list[float],
                 global_kwargs: dict = None,
                 per_kwargs: list[dict] = None):
        super().__init__()
        if weights is None:
            weights = [1.0] * len(types)
        assert len(types) == len(weights), "Types and weights must align"
        self.losses = nn.ModuleList()
        self.weights = weights

        for i, key in enumerate(types):
            cls = self.LOSS_MAP.get(key.lower())
            if cls is None:
                raise ValueError(f"Unknown classification loss type: {key}")
            kwargs = {}
            if global_kwargs:
                kwargs.update(global_kwargs)
            if per_kwargs and i < len(per_kwargs) and per_kwargs[i]:
                kwargs.update(per_kwargs[i])
            self.losses.append(cls(**kwargs))

    def forward(self, logits, targets):
        loss_vals = []
        for w, loss_fn in zip(self.weights, self.losses):
            raw = loss_fn(logits, targets)
            loss_vals.append(raw)
        total = sum(w * v for w, v in zip(self.weights, loss_vals))
        return total, loss_vals


class LossFactory:
    """
    Extended factory to create single-task loss modules.
    """
    @staticmethod
    def create_segmentation_loss(types: list[str],
                                 weights: list[float] = None,
                                 global_kwargs: dict = None,
                                 per_kwargs: list[dict] = None) -> nn.Module:
        """
        Returns a SegmentationLoss module that wraps multiple sub-losses.
        Usage:
            criterion = LossFactory.create_segmentation_loss(
                types=cfg.loss.seg_types,
                weights=cfg.loss.seg_weights,
                global_kwargs=cfg.loss.seg_global_kwargs,
                per_kwargs=cfg.loss.seg_per_kwargs
            )
            loss = criterion(logits, masks)
        """
        return SegmentationLoss(types, weights, global_kwargs, per_kwargs)

    @staticmethod
    def create_classification_loss(types: list[str],
                                   weights: list[float] = None,
                                   global_kwargs: dict = None,
                                   per_kwargs: list[dict] = None) -> nn.Module:
        """
        Returns a ClassificationLoss module that wraps multiple sub-losses.
        Usage:
            criterion = LossFactory.create_classification_loss(
                types=cfg.loss.cls_types,
                weights=cfg.loss.cls_weights,
                global_kwargs=cfg.loss.cls_global_kwargs,
                per_kwargs=cfg.loss.cls_per_kwargs
            )
            loss = criterion(logits, labels)
        """
        return ClassificationLoss(types, weights, global_kwargs, per_kwargs)
