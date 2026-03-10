from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from mnist_hybrid.config import TargetConfig


@dataclass
class TargetOutput:
    values: torch.Tensor
    value_type: str


class TargetBuilder:
    def __init__(self, config: TargetConfig, feature_dim: int, num_classes: int, device: torch.device) -> None:
        self.config = config
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.device = device

        self.centroids = torch.zeros((num_classes, feature_dim), device=device)
        self.centroid_initialized = torch.zeros((num_classes,), dtype=torch.bool, device=device)

    def _update_centroids(self, hidden: torch.Tensor, labels: torch.Tensor) -> None:
        momentum = self.config.centroid_momentum
        for cls in labels.unique():
            cls_int = int(cls.item())
            mask = labels == cls
            if not bool(mask.any()):
                continue
            cls_mean = hidden[mask].mean(dim=0)
            if not bool(self.centroid_initialized[cls_int]):
                self.centroids[cls_int] = cls_mean
                self.centroid_initialized[cls_int] = True
            else:
                self.centroids[cls_int] = momentum * self.centroids[cls_int] + (1.0 - momentum) * cls_mean

    def _require_grad(self, hidden_grad: Optional[torch.Tensor]) -> torch.Tensor:
        if hidden_grad is None:
            raise RuntimeError("Selected target_construction requires hidden gradients, but hidden_grad is None.")
        return hidden_grad

    def build(
        self,
        hidden_pre: torch.Tensor,
        labels: torch.Tensor,
        hidden_grad: Optional[torch.Tensor],
    ) -> TargetOutput:
        mode = self.config.target_construction
        value_type = self.config.value_type

        if mode == "current_hidden":
            if value_type == "absolute":
                values = hidden_pre.detach()
            else:
                values = torch.zeros_like(hidden_pre)
            return TargetOutput(values=values, value_type=value_type)

        if mode == "grad_delta":
            grad = self._require_grad(hidden_grad)
            delta = -self.config.gradient_step_size * grad.detach()
            return TargetOutput(values=delta, value_type="delta")

        if mode == "grad_target":
            grad = self._require_grad(hidden_grad)
            target = hidden_pre.detach() - self.config.gradient_step_size * grad.detach()
            return TargetOutput(values=target, value_type="absolute")

        if mode == "class_centroid":
            self._update_centroids(hidden_pre.detach(), labels.detach())
            target = self.centroids[labels]
            if value_type == "absolute":
                return TargetOutput(values=target.detach(), value_type="absolute")
            return TargetOutput(values=(target - hidden_pre.detach()), value_type="delta")

        raise ValueError(f"Unsupported target construction: {mode}")
