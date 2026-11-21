"""
Loss functions for rPPG estimation.
Implements Negative Pearson Correlation Loss as specified in the Siamese-rPPG paper.
"""

import torch
import torch.nn as nn


class NegativePearsonLoss(nn.Module):
    """
    Negative Pearson Correlation Loss:
    L = 1 - r(x, y) or -r(x, y)
    where r(x, y) = cov(x, y) / (std(x) * std(y))
    """
    def __init__(self, eps: float = 1e-7):
        super(NegativePearsonLoss, self).__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Estimated rPPG signal tensor of shape (Batch, Time)
            target: Ground truth PPG signal tensor of shape (Batch, Time)
        Returns:
            Scalar loss (1 - Pearson correlation coefficient, averaged over batch)
        """
        # Ensure 2D tensor (Batch, Time)
        if pred.dim() > 2:
            pred = pred.view(pred.size(0), -1)
        if target.dim() > 2:
            target = target.view(target.size(0), -1)

        # Center signals around mean
        pred_mean = torch.mean(pred, dim=-1, keepdim=True)
        target_mean = torch.mean(target, dim=-1, keepdim=True)

        pred_centered = pred - pred_mean
        target_centered = target - target_mean

        # Covariance
        covariance = torch.sum(pred_centered * target_centered, dim=-1)

        # Variance & Standard deviation
        pred_var = torch.sum(pred_centered ** 2, dim=-1)
        target_var = torch.sum(target_centered ** 2, dim=-1)

        denominator = torch.sqrt(pred_var * target_var) + self.eps

        # Pearson correlation r in range [-1, 1]
        r = covariance / denominator

        # Loss: 1 - r (range [0, 2], lower is better, 0 means perfect correlation)
        loss = 1.0 - torch.mean(r)
        return loss


def pearson_correlation(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Computes Pearson Correlation Coefficient between two 1D/2D signals.
    Returns:
        r tensor of shape (Batch,)
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if y.dim() == 1:
        y = y.unsqueeze(0)

    x_mean = torch.mean(x, dim=-1, keepdim=True)
    y_mean = torch.mean(y, dim=-1, keepdim=True)

    x_c = x - x_mean
    y_c = y - y_mean

    cov = torch.sum(x_c * y_c, dim=-1)
    std = torch.sqrt(torch.sum(x_c ** 2, dim=-1) * torch.sum(y_c ** 2, dim=-1)) + eps
    return cov / std
