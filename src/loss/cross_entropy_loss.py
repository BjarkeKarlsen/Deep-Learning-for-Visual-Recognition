import torch.nn as nn

class CrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss for classification tasks.
    """
    def __init__(self, weight=None, reduction='mean'):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(weight=weight, reduction=reduction)
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits of shape (N, C) where C is number of classes
            targets: Ground truth labels of shape (N,)
        """
        return self.loss_fn(inputs, targets)