"""
Evaluation metrics for JEPA-RobustViT.
Includes accuracy, Expected Calibration Error (ECE), and AverageMeter.
"""

import torch
import numpy as np


class AverageMeter:
    """Tracks mean and current value of a metric across batches."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute top-1 accuracy.

    Args:
        outputs: logits of shape (N, C)
        labels:  ground truth of shape (N,)

    Returns:
        accuracy as a float in [0, 100]
    """
    preds = outputs.argmax(dim=1)
    correct = (preds == labels).sum().item()
    return 100.0 * correct / labels.size(0)


def compute_ece(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    Lower is better. Used to measure prediction confidence calibration.

    Args:
        outputs:  logits of shape (N, C)
        labels:   ground truth of shape (N,)
        n_bins:   number of confidence bins

    Returns:
        ECE as a float in [0, 1]
    """
    softmax = torch.nn.functional.softmax(outputs, dim=1)
    confidences, predictions = softmax.max(dim=1)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1, device=outputs.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=outputs.device)

    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.item()