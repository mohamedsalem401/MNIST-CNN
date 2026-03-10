from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == labels).float().mean().item())


def cross_entropy_per_sample(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels, reduction="none")


def expected_calibration_error(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    bins: int = 15,
) -> float:
    confidences, predictions = probabilities.max(dim=1)
    accuracies = predictions.eq(labels)

    boundaries = torch.linspace(0, 1, bins + 1, device=probabilities.device)
    ece = torch.zeros(1, device=probabilities.device)

    for i in range(bins):
        left, right = boundaries[i], boundaries[i + 1]
        in_bin = confidences.gt(left) & confidences.le(right)
        prop = in_bin.float().mean()
        if prop.item() > 0:
            acc = accuracies[in_bin].float().mean()
            conf = confidences[in_bin].mean()
            ece += torch.abs(conf - acc) * prop

    return float(ece.item())


def compute_confusion_matrix(preds: torch.Tensor, labels: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long, device=preds.device)
    for t, p in zip(labels, preds):
        cm[int(t.item()), int(p.item())] += 1
    return cm


def per_class_precision_recall_f1(cm: torch.Tensor) -> Dict[str, Dict[str, float]]:
    num_classes = cm.size(0)
    metrics: Dict[str, Dict[str, float]] = {}

    for cls in range(num_classes):
        tp = float(cm[cls, cls].item())
        fp = float(cm[:, cls].sum().item() - tp)
        fn = float(cm[cls, :].sum().item() - tp)

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        metrics[str(cls)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(cm[cls, :].sum().item()),
        }

    return metrics


def summarize_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    losses: torch.Tensor,
    ece_bins: int,
    with_per_class: bool,
    with_confusion: bool,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    probs = torch.softmax(logits, dim=1)
    preds = logits.argmax(dim=1)

    summary = {
        "loss": float(losses.mean().item()),
        "accuracy": float((preds == labels).float().mean().item()),
        "ece": expected_calibration_error(probs, labels, bins=ece_bins),
    }

    details: Dict[str, object] = {
        "predictions": preds.detach().cpu(),
        "probabilities": probs.detach().cpu(),
    }

    if with_confusion or with_per_class:
        cm = compute_confusion_matrix(preds, labels, num_classes=logits.size(1)).detach().cpu()
        if with_confusion:
            details["confusion_matrix"] = cm
        if with_per_class:
            details["per_class"] = per_class_precision_recall_f1(cm)

    return summary, details
