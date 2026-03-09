from __future__ import annotations

from typing import Optional

import torch


class ReservoirState:
    def __init__(self) -> None:
        self.seen = 0


def ttl_alive_mask(insert_step: torch.Tensor, current_step: int, ttl_steps: int) -> torch.Tensor:
    if ttl_steps <= 0:
        return torch.ones_like(insert_step, dtype=torch.bool)
    age = current_step - insert_step
    return age <= ttl_steps


def select_eviction_indices(
    policy: str,
    num_to_evict: int,
    insert_step: torch.Tensor,
    retrieval_count: torch.Tensor,
    usefulness: torch.Tensor,
    helpfulness_age_weight: float,
    current_step: int,
) -> torch.Tensor:
    if num_to_evict <= 0:
        return torch.empty(0, dtype=torch.long, device=insert_step.device)

    policy = policy.lower()
    if policy in {"fifo", "ttl", "none"}:
        # FIFO default removes oldest entries first.
        order = torch.argsort(insert_step)
        return order[:num_to_evict]

    if policy == "usage":
        order = torch.argsort(retrieval_count)
        return order[:num_to_evict]

    if policy == "helpfulness":
        order = torch.argsort(usefulness)
        return order[:num_to_evict]

    if policy == "helpfulness_age":
        age = (current_step - insert_step).float()
        age = age / (age.max() + 1e-6)
        usefulness_norm = usefulness - usefulness.min()
        usefulness_norm = usefulness_norm / (usefulness_norm.max() + 1e-6)
        score = (1.0 - helpfulness_age_weight) * usefulness_norm - helpfulness_age_weight * age
        order = torch.argsort(score)
        return order[:num_to_evict]

    raise ValueError(f"Unsupported eviction policy: {policy}")


def reservoir_target_slot(
    reservoir_state: ReservoirState,
    memory_size: int,
    device: torch.device,
) -> Optional[int]:
    reservoir_state.seen += 1
    seen = reservoir_state.seen

    if seen <= memory_size:
        return seen - 1

    j = int(torch.randint(low=0, high=seen, size=(1,), device=device).item())
    if j < memory_size:
        return j
    return None
