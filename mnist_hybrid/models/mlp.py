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
        growth_enabled: bool = False,
    ) -> None:
        super().__init__()
        if not hidden_sizes:
            raise ValueError("MLP requires at least one hidden layer")

        self.hidden_sizes = list(hidden_sizes)
        self.num_classes = num_classes
        self.extra_parametric_layer = extra_parametric_layer
        self.growth_enabled = growth_enabled
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.hidden_layers: nn.ModuleList = nn.ModuleList()
        self.extra_layer: Optional[nn.Linear] = None
        self.classifier = nn.Linear(self.hidden_sizes[-1], num_classes)
        self._rebuild_network(self.hidden_sizes, preserve=False)

    @staticmethod
    def _copy_linear_params(src: nn.Linear, dst: nn.Linear) -> None:
        out_dim = min(src.weight.size(0), dst.weight.size(0))
        in_dim = min(src.weight.size(1), dst.weight.size(1))
        with torch.no_grad():
            dst.weight[:out_dim, :in_dim].copy_(src.weight[:out_dim, :in_dim])
            dst.bias[:out_dim].copy_(src.bias[:out_dim])

    def _rebuild_network(self, hidden_sizes: List[int], preserve: bool) -> None:
        old_hidden_layers = self.hidden_layers if preserve else None
        old_extra_layer = self.extra_layer if preserve else None
        old_classifier = self.classifier if preserve else None
        try:
            current_device = next(self.parameters()).device
        except StopIteration:
            current_device = torch.device("cpu")

        input_dim = 28 * 28
        layer_sizes = [input_dim, *hidden_sizes]
        new_hidden_layers = nn.ModuleList(
            [nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]
        )

        new_extra_layer: Optional[nn.Linear] = None
        if self.extra_parametric_layer:
            new_extra_layer = nn.Linear(hidden_sizes[-1], hidden_sizes[-1])
        new_classifier = nn.Linear(hidden_sizes[-1], self.num_classes)

        if preserve and old_hidden_layers is not None:
            shared_layers = min(len(old_hidden_layers), len(new_hidden_layers))
            for idx in range(shared_layers):
                self._copy_linear_params(old_hidden_layers[idx], new_hidden_layers[idx])

            if old_extra_layer is not None and new_extra_layer is not None:
                self._copy_linear_params(old_extra_layer, new_extra_layer)

            if old_classifier is not None:
                self._copy_linear_params(old_classifier, new_classifier)

        self.hidden_sizes = list(hidden_sizes)
        self.hidden_layers = new_hidden_layers.to(current_device)
        self.extra_layer = new_extra_layer.to(current_device) if new_extra_layer is not None else None
        self.classifier = new_classifier.to(current_device)

    def candidate_layers(self) -> List[str]:
        names = [f"hidden_{idx}" for idx in range(len(self.hidden_layers))]
        if self.extra_layer is not None:
            names.append("extra_parametric")
        names.append("penultimate")
        return names

    def supports_growth(self) -> bool:
        return True

    def growth_state(self) -> Dict[str, int]:
        return {
            "current_depth": len(self.hidden_sizes),
            "current_width": max(self.hidden_sizes),
        }

    def grow(
        self,
        growth_mode: str,
        growth_amount_width: int,
        growth_amount_depth: int,
        max_width: int,
        max_depth: int,
    ) -> Dict[str, object]:
        if not self.growth_enabled:
            return {"grew": False, "reason": "growth_disabled"}

        old_sizes = list(self.hidden_sizes)
        new_sizes = list(self.hidden_sizes)

        grow_width = growth_mode in {"width", "width_and_depth"} and growth_amount_width > 0
        grow_depth = growth_mode in {"depth", "width_and_depth"} and growth_amount_depth > 0

        if grow_width:
            for idx, width in enumerate(new_sizes):
                target = width + growth_amount_width
                if max_width > 0:
                    target = min(target, max_width)
                new_sizes[idx] = max(width, target)

        if grow_depth:
            depth_limit = max_depth if max_depth > 0 else len(new_sizes) + growth_amount_depth
            slots = max(depth_limit - len(new_sizes), 0)
            to_add = min(growth_amount_depth, slots)
            base_width = new_sizes[-1]
            if max_width > 0:
                base_width = min(base_width, max_width)
            for _ in range(to_add):
                new_sizes.append(base_width)

        if new_sizes == old_sizes:
            return {
                "grew": False,
                "reason": "at_capacity_or_zero_growth",
                "old_hidden_sizes": old_sizes,
                "new_hidden_sizes": new_sizes,
            }

        self._rebuild_network(new_sizes, preserve=True)
        return {
            "grew": True,
            "old_hidden_sizes": old_sizes,
            "new_hidden_sizes": list(new_sizes),
            "old_depth": len(old_sizes),
            "new_depth": len(new_sizes),
            "old_width": max(old_sizes),
            "new_width": max(new_sizes),
        }

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
