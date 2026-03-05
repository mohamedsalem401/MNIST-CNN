from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ForwardTrace:
    """Container for forward-pass activations used by the visualizer."""

    activations: List[torch.Tensor]
    probabilities: torch.Tensor


class MNISTMLP(nn.Module):
    def __init__(self, hidden_sizes: Sequence[int] = (128, 64), num_classes: int = 10) -> None:
        super().__init__()
        self.hidden_sizes = tuple(hidden_sizes)
        layer_sizes = [28 * 28, *self.hidden_sizes, num_classes]
        self.layers = nn.ModuleList(
            [nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)

    @torch.no_grad()
    def forward_with_trace(self, x: torch.Tensor) -> ForwardTrace:
        x = x.view(x.size(0), -1)
        activations: List[torch.Tensor] = [x.detach().clone()]

        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
            activations.append(x.detach().clone())

        logits = self.layers[-1](x)
        probabilities = F.softmax(logits, dim=1)
        activations.append(probabilities.detach().clone())

        return ForwardTrace(activations=activations, probabilities=probabilities)
