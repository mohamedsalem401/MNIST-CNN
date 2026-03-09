from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import nn
import torch.nn.functional as F

from mnist_hybrid.models.base import ForwardResult, IntervenableModel, InterventionFn


class SmallCNNClassifier(IntervenableModel):
    def __init__(
        self,
        channels: List[int],
        latent_dim: int = 128,
        num_classes: int = 10,
        dropout: float = 0.0,
        extra_parametric_layer: bool = False,
    ) -> None:
        super().__init__()
        if len(channels) < 2:
            raise ValueError("SmallCNNClassifier expects at least two channel sizes")

        self.conv1 = nn.Conv2d(1, channels[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        flattened_dim = channels[1] * 7 * 7
        self.fc1 = nn.Linear(flattened_dim, latent_dim)
        self.extra_layer = nn.Linear(latent_dim, latent_dim) if extra_parametric_layer else None
        self.classifier = nn.Linear(latent_dim, num_classes)

    def candidate_layers(self) -> List[str]:
        names = ["conv_early", "conv_middle", "fc_late"]
        if self.extra_layer is not None:
            names.append("extra_parametric")
        names.append("penultimate")
        return names

    def _apply(
        self,
        layer_name: str,
        tensor: torch.Tensor,
        intervention_fn: Optional[InterventionFn],
        intervention_log: Dict[str, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        if intervention_fn is None:
            return tensor
        original_shape = tensor.shape

        # Memory intervention operates on vectors. For convolutional activations,
        # we map each sample to a flattened representation and restore shape.
        if tensor.dim() == 4:
            flat = tensor.flatten(1)
            post = intervention_fn(layer_name, flat)
            post = post.view(original_shape)
            pre_flat = flat
            post_flat = post.flatten(1)
        else:
            pre_flat = tensor
            post = intervention_fn(layer_name, tensor)
            post_flat = post

        if post_flat is not pre_flat:
            intervention_log[layer_name] = {
                "pre": pre_flat.detach(),
                "post": post_flat.detach(),
                "delta_norm": (post_flat - pre_flat).norm(dim=1).detach(),
            }
        return post

    def forward_intervenable(
        self,
        x: torch.Tensor,
        intervention_fn: Optional[InterventionFn] = None,
        capture_activations: bool = False,
    ) -> ForwardResult:
        activations: Dict[str, torch.Tensor] = {}
        intervention_log: Dict[str, Dict[str, torch.Tensor]] = {}

        x = F.relu(self.conv1(x))
        x = self._apply("conv_early", x, intervention_fn, intervention_log)
        x = self.pool(x)
        if capture_activations:
            activations["conv_early"] = x.flatten(1).detach()

        x = F.relu(self.conv2(x))
        x = self._apply("conv_middle", x, intervention_fn, intervention_log)
        x = self.pool(x)
        if capture_activations:
            activations["conv_middle"] = x.flatten(1).detach()

        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self._apply("fc_late", x, intervention_fn, intervention_log)
        x = self.dropout(x)
        if capture_activations:
            activations["fc_late"] = x.detach()

        if self.extra_layer is not None:
            x = F.relu(self.extra_layer(x))
            x = self._apply("extra_parametric", x, intervention_fn, intervention_log)
            if capture_activations:
                activations["extra_parametric"] = x.detach()

        x = self._apply("penultimate", x, intervention_fn, intervention_log)
        if capture_activations:
            activations["penultimate"] = x.detach()

        logits = self.classifier(x)
        if capture_activations:
            activations["logits"] = logits.detach()
        return ForwardResult(logits=logits, activations=activations, intervention_log=intervention_log)
