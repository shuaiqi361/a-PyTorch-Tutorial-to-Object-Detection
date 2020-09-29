import torch
import torch.nn as nn
import torch.nn.functional as F


class SmoothL1Loss(nn.Module):
    def __init__(self, beta=1.0 / 9.0, reduction='mean'):
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, pred, target, weights=None):
        num = pred.size(0)

        x = (pred - target).abs()
        l1 = x - 0.5 * self.beta
        l2 = 0.5 * x ** 2 / self.beta
        l1_loss = torch.where(x >= self.beta, l1, l2)

        if weights is not None and weights.sum() > 1e-6:
            assert pred.size(0) == target.size(0) == weights.size(0)
            return (l1_loss * weights).sum() / weights.sum()
        else:
            if self.reduction == 'mean':
                return l1_loss.sum() / num
            else:
                return l1_loss.sum()
