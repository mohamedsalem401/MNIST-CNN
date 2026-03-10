from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import nn
import torch.nn.functional as F

from mnist_hybrid.models.base import ForwardResult, IntervenableModel, InterventionFn


class MLPClassifier(IntervenableModel):
    def __init__(
        self,
        hidden_sizes: List[int],
        num_classes: int = 10,
        dropout: float = 0.0,
        extra_parametric_layer: bool = False,
    ) -> None:
        super().__init__()
        if not hidden_sizes:
            raise ValueError("MLP requires at least one hidden layer")

        self.hidden_sizes = list(hidden_sizes)
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        input_dim = 28 * 28
        layer_sizes = [input_dim, *self.hidden_sizes]
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]
        )

        self.extra_layer = None
        extra_in_dim = self.hidden_sizes[-1]
        if extra_parametric_layer:
            self.extra_layer = nn.Linear(self.hidden_sizes[-1], self.hidden_sizes[-1])
            extra_in_dim = self.hidden_sizes[-1]

        self.classifier = nn.Linear(extra_in_dim, num_classes)

    def candidate_layers(self) -> List[str]:
        names = [f"hidden_{idx}" for idx in range(len(self.hidden_layers))]
        if self.extra_layer is not None:
            names.append("extra_parametric")
        names.append("penultimate")
        return names

    def _apply_intervention(
        self,
        layer_name: str,
        tensor: torch.Tensor,
        intervention_fn: Optional[InterventionFn],
        intervention_log: Dict[str, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        if intervention_fn is None:
            return tensor
        pre = tensor
        post = intervention_fn(layer_name, pre)
        if post is not pre:
            intervention_log[layer_name] = {
                "pre": pre.detach(),
                "post": post.detach(),
                "delta_norm": (post - pre).norm(dim=1).detach(),
            }
        return post

    def forward_intervenable(
        self,
        x: torch.Tensor,
        intervention_fn: Optional[InterventionFn] = None,
        capture_activations: bool = False,
    ) -> ForwardResult:
        x = x.view(x.size(0), -1)
        activations: Dict[str, torch.Tensor] = {}
        intervention_log: Dict[str, Dict[str, torch.Tensor]] = {}

        for idx, layer in enumerate(self.hidden_layers):
            x = F.relu(layer(x))
            layer_name = f"hidden_{idx}"
            x = self._apply_intervention(layer_name, x, intervention_fn, intervention_log)
            x = self.dropout(x)
            if capture_activations:
                activations[layer_name] = x.detach()

        if self.extra_layer is not None:
            x = F.relu(self.extra_layer(x))
            x = self._apply_intervention("extra_parametric", x, intervention_fn, intervention_log)
            if capture_activations:
                activations["extra_parametric"] = x.detach()

        x = self._apply_intervention("penultimate", x, intervention_fn, intervention_log)
        if capture_activations:
            activations["penultimate"] = x.detach()

        logits = self.classifier(x)
        if capture_activations:
            activations["logits"] = logits.detach()

        return ForwardResult(logits=logits, activations=activations, intervention_log=intervention_log)
