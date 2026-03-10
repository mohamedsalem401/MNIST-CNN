from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from mnist_hybrid.config import ForgettingConfig, RefreshConfig, RetrievalConfig
from mnist_hybrid.memory.policies import ReservoirState, reservoir_target_slot, select_eviction_indices, ttl_alive_mask
from mnist_hybrid.memory.retrieval import maybe_normalize, pairwise_distance, topk_retrieval
from mnist_hybrid.memory.types import MemoryStats, QueryResult


@dataclass
class RetrievalMetadata:
    topk_indices: torch.Tensor
    topk_distances: torch.Tensor
    topk_weights: torch.Tensor
    purity: torch.Tensor


class MemoryBank:
    def __init__(
        self,
        key_dim: int,
        value_dim: int,
        retrieval_cfg: RetrievalConfig,
        forgetting_cfg: ForgettingConfig,
        refresh_cfg: RefreshConfig,
        random_memory_control: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.retrieval_cfg = retrieval_cfg
        self.forgetting_cfg = forgetting_cfg
        self.refresh_cfg = refresh_cfg
        self.random_memory_control = random_memory_control

        self.keys = torch.empty((0, key_dim), dtype=torch.float32, device=self.device)
        self.values = torch.empty((0, value_dim), dtype=torch.float32, device=self.device)
        self.labels = torch.empty((0,), dtype=torch.long, device=self.device)
        self.insert_step = torch.empty((0,), dtype=torch.long, device=self.device)
        self.retrieval_count = torch.empty((0,), dtype=torch.float32, device=self.device)
        self.usefulness = torch.empty((0,), dtype=torch.float32, device=self.device)
        self.freshness = torch.empty((0,), dtype=torch.float32, device=self.device)

        self._reservoir_state = ReservoirState()

    @property
    def size(self) -> int:
        return int(self.keys.size(0))

    def _normalize_inputs(self, keys: torch.Tensor, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        keys = maybe_normalize(keys, self.retrieval_cfg.normalize_keys)
        values = maybe_normalize(values, self.retrieval_cfg.normalize_values)
        return keys, values

    def _apply_ttl(self, current_step: int) -> None:
        if self.forgetting_cfg.policy != "ttl" or self.forgetting_cfg.ttl_steps <= 0 or self.size == 0:
            return
        alive = ttl_alive_mask(self.insert_step, current_step=current_step, ttl_steps=self.forgetting_cfg.ttl_steps)
        if bool(alive.all()):
            return
        self._keep_mask(alive)

    def _keep_mask(self, mask: torch.Tensor) -> None:
        self.keys = self.keys[mask]
        self.values = self.values[mask]
        self.labels = self.labels[mask]
        self.insert_step = self.insert_step[mask]
        self.retrieval_count = self.retrieval_count[mask]
        self.usefulness = self.usefulness[mask]
        self.freshness = self.freshness[mask]

    def _drop_indices(self, indices: torch.Tensor) -> None:
        if indices.numel() == 0:
            return
        mask = torch.ones(self.size, dtype=torch.bool, device=self.device)
        mask[indices] = False
        self._keep_mask(mask)

    def _append(self, keys: torch.Tensor, values: torch.Tensor, labels: torch.Tensor, step: int) -> None:
        n = keys.size(0)
        self.keys = torch.cat([self.keys, keys], dim=0)
        self.values = torch.cat([self.values, values], dim=0)
        self.labels = torch.cat([self.labels, labels], dim=0)
        self.insert_step = torch.cat(
            [self.insert_step, torch.full((n,), step, dtype=torch.long, device=self.device)],
            dim=0,
        )
        self.retrieval_count = torch.cat(
            [self.retrieval_count, torch.zeros((n,), dtype=torch.float32, device=self.device)],
            dim=0,
        )
        self.usefulness = torch.cat(
            [self.usefulness, torch.zeros((n,), dtype=torch.float32, device=self.device)],
            dim=0,
        )
        self.freshness = torch.cat(
            [self.freshness, torch.ones((n,), dtype=torch.float32, device=self.device)],
            dim=0,
        )

    def _maybe_refresh(self, key: torch.Tensor, value: torch.Tensor, label: int) -> bool:
        if not self.refresh_cfg.enabled or self.size == 0:
            return False
        if float(torch.rand(1, device=self.device).item()) > self.refresh_cfg.rate:
            return False

        distances = pairwise_distance(key.unsqueeze(0), self.keys, metric=self.retrieval_cfg.metric).squeeze(0)
        best_dist, best_idx = torch.min(distances, dim=0)
        if best_dist.item() > self.refresh_cfg.match_distance_threshold:
            return False

        idx = int(best_idx.item())
        self.keys[idx] = self.refresh_cfg.ema_key * self.keys[idx] + (1.0 - self.refresh_cfg.ema_key) * key
        self.values[idx] = self.refresh_cfg.ema_value * self.values[idx] + (1.0 - self.refresh_cfg.ema_value) * value
        self.labels[idx] = int(label)
        self.freshness[idx] = 1.0
        return True

    def _enforce_capacity(self, current_step: int) -> None:
        capacity = self.forgetting_cfg.memory_size
        if capacity <= 0:
            return
        if self.size <= capacity:
            return

        to_evict = self.size - capacity
        indices = select_eviction_indices(
            policy=self.forgetting_cfg.policy,
            num_to_evict=to_evict,
            insert_step=self.insert_step,
            retrieval_count=self.retrieval_count,
            usefulness=self.usefulness,
            helpfulness_age_weight=self.forgetting_cfg.helpfulness_age_weight,
            current_step=current_step,
        )
        self._drop_indices(indices)

    def insert(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        labels: torch.Tensor,
        step: int,
        insertion_policy: str = "always",
        losses: Optional[torch.Tensor] = None,
    ) -> int:
        if keys.numel() == 0:
            return 0

        self._apply_ttl(step)

        keys = keys.detach().to(self.device)
        values = values.detach().to(self.device)
        labels = labels.detach().to(self.device).long()
        keys, values = self._normalize_inputs(keys, values)

        if insertion_policy == "loss_above_mean" and losses is not None:
            losses = losses.detach().to(keys.device)
            keep = losses > losses.mean()
            keys = keys[keep]
            values = values[keep]
            labels = labels[keep]
            if keys.numel() == 0:
                return 0

        inserted = 0
        if self.forgetting_cfg.policy == "reservoir" and self.forgetting_cfg.memory_size > 0:
            for key, value, label in zip(keys, values, labels):
                slot = reservoir_target_slot(
                    reservoir_state=self._reservoir_state,
                    memory_size=self.forgetting_cfg.memory_size,
                    device=self.device,
                )
                if slot is None:
                    continue
                if slot >= self.size:
                    self._append(key.unsqueeze(0), value.unsqueeze(0), label.unsqueeze(0), step)
                else:
                    self.keys[slot] = key
                    self.values[slot] = value
                    self.labels[slot] = label
                    self.insert_step[slot] = step
                    self.retrieval_count[slot] = 0.0
                    self.usefulness[slot] = 0.0
                    self.freshness[slot] = 1.0
                inserted += 1
            return inserted

        for key, value, label in zip(keys, values, labels):
            if self._maybe_refresh(key, value, int(label.item())):
                inserted += 1
                continue
            self._append(key.unsqueeze(0), value.unsqueeze(0), label.unsqueeze(0), step)
            inserted += 1

        self._enforce_capacity(current_step=step)
        return inserted

    def query(
        self,
        queries: torch.Tensor,
        query_labels: Optional[torch.Tensor] = None,
    ) -> QueryResult:
        if self.size == 0:
            batch = queries.size(0)
            zeros_values = torch.zeros((batch, self.value_dim), device=queries.device)
            empty_idx = torch.zeros((batch, 0), dtype=torch.long, device=queries.device)
            empty_dist = torch.zeros((batch, 0), device=queries.device)
            empty_weights = torch.zeros((batch, 0), device=queries.device)
            purity = torch.zeros((batch,), device=queries.device)
            return QueryResult(
                values=zeros_values,
                indices=empty_idx,
                distances=empty_dist,
                weights=empty_weights,
                labels=None,
                purity=purity,
            )

        queries = queries.detach().to(self.device)
        queries = maybe_normalize(queries, self.retrieval_cfg.normalize_queries)

        keys = self.keys
        values = self.values
        labels = self.labels

        if self.random_memory_control:
            k = min(self.retrieval_cfg.k, self.size)
            rand_idx = torch.randint(0, self.size, size=(queries.size(0), k), device=self.device)
            rand_values = values[rand_idx]
            weights = torch.full((queries.size(0), k), 1.0 / k, device=self.device)
            aggregated = (rand_values * weights.unsqueeze(-1)).sum(dim=1)
            topk_labels = labels[rand_idx]
            purity = torch.zeros((queries.size(0),), device=self.device)
            if query_labels is not None:
                purity = (topk_labels == query_labels.unsqueeze(1).to(self.device)).float().mean(dim=1)
            return QueryResult(
                values=aggregated,
                indices=rand_idx,
                distances=torch.zeros_like(weights),
                weights=weights,
                labels=topk_labels,
                purity=purity,
            )

        class_filter = None
        if self.retrieval_cfg.class_conditional and query_labels is not None:
            class_filter = query_labels.to(self.device)

        aggregated, topk_idx, topk_dist, topk_weights, topk_labels = topk_retrieval(
            queries=queries,
            keys=keys,
            values=values,
            k=self.retrieval_cfg.k,
            metric=self.retrieval_cfg.metric,
            weighting=self.retrieval_cfg.weighting,
            temperature=self.retrieval_cfg.temperature,
            labels=labels,
            class_filter=class_filter,
        )

        flat_indices = topk_idx.reshape(-1)
        self.retrieval_count[flat_indices] += 1.0

        purity = torch.zeros((queries.size(0),), device=self.device)
        if topk_labels is not None and query_labels is not None:
            purity = (topk_labels == query_labels.unsqueeze(1).to(self.device)).float().mean(dim=1)

        return QueryResult(
            values=aggregated,
            indices=topk_idx,
            distances=topk_dist,
            weights=topk_weights,
            labels=topk_labels,
            purity=purity,
        )

    def update_usefulness(
        self,
        retrieved_indices: torch.Tensor,
        reward: torch.Tensor,
    ) -> None:
        if retrieved_indices.numel() == 0:
            return
        reward = reward.detach().to(self.device)
        if reward.dim() == 1:
            reward = reward.unsqueeze(1)
        idx_flat = retrieved_indices.reshape(-1)
        reward_flat = reward.repeat_interleave(retrieved_indices.size(1), dim=1).reshape(-1)
        self.usefulness.index_add_(0, idx_flat, reward_flat)

    def decay_freshness(self, amount: float = 0.01) -> None:
        if self.size == 0:
            return
        self.freshness = torch.clamp(self.freshness - amount, min=0.0)

    def memory_footprint_bytes(self) -> int:
        tensors = [
            self.keys,
            self.values,
            self.labels,
            self.insert_step,
            self.retrieval_count,
            self.usefulness,
            self.freshness,
        ]
        return int(sum(t.numel() * t.element_size() for t in tensors))

    def stats(self, current_step: int) -> MemoryStats:
        if self.size == 0:
            return MemoryStats(
                size=0,
                avg_age=0.0,
                avg_retrieval_count=0.0,
                avg_usefulness=0.0,
                avg_freshness=0.0,
                class_histogram={},
            )

        age = (current_step - self.insert_step).float()
        labels, counts = torch.unique(self.labels, return_counts=True)
        histogram = {int(l.item()): int(c.item()) for l, c in zip(labels, counts)}

        return MemoryStats(
            size=self.size,
            avg_age=float(age.mean().item()),
            avg_retrieval_count=float(self.retrieval_count.mean().item()),
            avg_usefulness=float(self.usefulness.mean().item()),
            avg_freshness=float(self.freshness.mean().item()),
            class_histogram=histogram,
        )

    def export_state(self) -> Dict[str, torch.Tensor]:
        return {
            "keys": self.keys.detach().cpu(),
            "values": self.values.detach().cpu(),
            "labels": self.labels.detach().cpu(),
            "insert_step": self.insert_step.detach().cpu(),
            "retrieval_count": self.retrieval_count.detach().cpu(),
            "usefulness": self.usefulness.detach().cpu(),
            "freshness": self.freshness.detach().cpu(),
        }
