from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class QueryResult:
    values: torch.Tensor
    indices: torch.Tensor
    distances: torch.Tensor
    weights: torch.Tensor
    labels: Optional[torch.Tensor]
    purity: torch.Tensor


@dataclass
class MemoryStats:
    size: int
    avg_age: float
    avg_retrieval_count: float
    avg_usefulness: float
    avg_freshness: float
    class_histogram: dict
