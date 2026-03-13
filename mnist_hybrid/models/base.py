from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import nn


InterventionFn = Callable[[str, torch.Tensor], torch.Tensor]


@dataclass
class ForwardResult:
    logits: torch.Tensor
    activations: Dict[str, torch.Tensor]
    intervention_log: Dict[str, Dict[str, torch.Tensor]]


class IntervenableModel(nn.Module):
    def candidate_layers(self) -> List[str]:
        raise NotImplementedError

    def forward_intervenable(
        self,
        x: torch.Tensor,
        intervention_fn: Optional[InterventionFn] = None,
        capture_activations: bool = False,
    ) -> ForwardResult:
        raise NotImplementedError

    def supports_growth(self) -> bool:
        return False

    def growth_state(self) -> Dict[str, int]:
        return {}

    def grow(
        self,
        growth_mode: str,
        growth_amount_width: int,
        growth_amount_depth: int,
        max_width: int,
        max_depth: int,
    ) -> Dict[str, object]:
        return {"grew": False}

    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_intervenable(x, intervention_fn=None, capture_activations=False).logits
