from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def maybe_normalize(x: torch.Tensor, enabled: bool) -> torch.Tensor:
    if not enabled:
        return x
    return F.normalize(x, p=2, dim=-1, eps=1e-8)


def pairwise_distance(
    queries: torch.Tensor,
    keys: torch.Tensor,
    metric: str,
) -> torch.Tensor:
    if metric == "euclidean":
        return torch.cdist(queries, keys, p=2)
    if metric == "cosine":
        similarity = F.normalize(queries, dim=-1) @ F.normalize(keys, dim=-1).T
        return 1.0 - similarity
    raise ValueError(f"Unsupported metric: {metric}")


def topk_retrieval(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    k: int,
    metric: str,
    weighting: str,
    temperature: float,
    labels: Optional[torch.Tensor] = None,
    class_filter: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if keys.numel() == 0:
        raise ValueError("Cannot retrieve from empty memory.")

    distances = pairwise_distance(queries, keys, metric=metric)

    if class_filter is not None and labels is not None:
        # Class-conditional masking: keep keys matching class_filter per query.
        match = labels.unsqueeze(0) == class_filter.unsqueeze(1)
        invalid = ~match
        distances = distances.masked_fill(invalid, float("inf"))

    k = min(k, keys.size(0))
    topk_dist, topk_idx = torch.topk(distances, k=k, largest=False, dim=1)
    topk_values = values[topk_idx]

    if weighting == "uniform":
        weights = torch.full_like(topk_dist, 1.0 / k)
    elif weighting == "inverse_distance":
        inv = 1.0 / (topk_dist + 1e-8)
        weights = inv / inv.sum(dim=1, keepdim=True)
    elif weighting == "softmax":
        weights = torch.softmax(-topk_dist / max(temperature, 1e-6), dim=1)
    else:
        raise ValueError(f"Unsupported weighting mode: {weighting}")

    weighted = (topk_values * weights.unsqueeze(-1)).sum(dim=1)
    topk_labels = labels[topk_idx] if labels is not None else None
    return weighted, topk_idx, topk_dist, weights, topk_labels
