from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from mnist_hybrid.config import MemoryConfig
from mnist_hybrid.memory.datastore import MemoryBank
from mnist_hybrid.memory.targets import TargetBuilder
from mnist_hybrid.memory.types import QueryResult


@dataclass
class BatchInterventionState:
    enabled_for_forward: bool
    layer_name: str
    hidden_pre: Optional[torch.Tensor] = None
    hidden_post: Optional[torch.Tensor] = None
    query_tensor: Optional[torch.Tensor] = None
    query_result: Optional[QueryResult] = None
    affected_indices: Optional[torch.Tensor] = None
    intervention_magnitude: Optional[torch.Tensor] = None
    gate_values: Optional[torch.Tensor] = None
    retrieval_top1_agreement: Optional[torch.Tensor] = None


class HybridInterventionEngine:
    def __init__(self, memory_config: MemoryConfig, num_classes: int = 10) -> None:
        self.cfg = memory_config
        self.num_classes = num_classes

        self.memory_bank: Optional[MemoryBank] = None
        self.target_builder: Optional[TargetBuilder] = None

        self.hidden_dim: Optional[int] = None
        self.query_dim: Optional[int] = None
        self.affected_indices: Optional[torch.Tensor] = None
        self.query_projection_indices: Optional[torch.Tensor] = None

        self.training = True
        self.current_epoch = 0
        self.current_step = 0
        self.current_labels: Optional[torch.Tensor] = None
        self.state = BatchInterventionState(
            enabled_for_forward=False,
            layer_name=self.cfg.intervention.layer,
        )

    def reset_dynamic_state(self) -> None:
        self.memory_bank = None
        self.target_builder = None
        self.hidden_dim = None
        self.query_dim = None
        self.affected_indices = None
        self.query_projection_indices = None

    def set_mode(self, training: bool) -> None:
        self.training = training

    def begin_batch(self, labels: torch.Tensor, epoch: int, step: int, training: bool) -> None:
        self.current_labels = labels
        self.current_epoch = epoch
        self.current_step = step
        self.training = training
        self.state = BatchInterventionState(
            enabled_for_forward=self._memory_enabled_for_phase(epoch=epoch, training=training),
            layer_name=self.cfg.intervention.layer,
        )

    def _memory_enabled_for_phase(self, epoch: int, training: bool) -> bool:
        if not self.cfg.enabled or not self.cfg.intervention.enabled:
            return False

        warmup_passed = epoch >= self.cfg.intervention.warmup_epochs
        if training:
            if not self.cfg.intervention.training_use:
                return False
        else:
            if not self.cfg.intervention.inference_use:
                return False

        return warmup_passed

    def _should_populate(self, epoch: int) -> bool:
        warmup_passed = epoch >= self.cfg.intervention.warmup_epochs
        if warmup_passed:
            return True
        return self.cfg.intervention.populate_during_warmup

    def _init_layer_state(self, hidden: torch.Tensor) -> None:
        if self.hidden_dim is not None:
            return

        device = hidden.device
        hidden_dim = hidden.size(1)
        self.hidden_dim = hidden_dim

        affected = min(self.cfg.intervention.affected_dims, hidden_dim)
        if self.cfg.intervention.affected_selection == "random":
            generator = torch.Generator(device=device)
            generator.manual_seed(17)
            self.affected_indices = torch.randperm(hidden_dim, generator=generator, device=device)[:affected]
        else:
            self.affected_indices = torch.arange(affected, device=device)

        untouched = torch.tensor(
            [i for i in range(hidden_dim) if i not in set(self.affected_indices.tolist())],
            device=device,
            dtype=torch.long,
        )

        if self.cfg.intervention.query_mode == "full":
            query_dim = hidden_dim
        elif self.cfg.intervention.query_mode == "untouched":
            query_dim = max(int(untouched.numel()), 1)
        elif self.cfg.intervention.query_mode == "projection":
            proj_dims = min(self.cfg.intervention.query_projection_dims, hidden_dim)
            generator = torch.Generator(device=device)
            generator.manual_seed(23)
            self.query_projection_indices = torch.randperm(hidden_dim, generator=generator, device=device)[:proj_dims]
            query_dim = int(proj_dims)
        else:
            raise ValueError(f"Unsupported query mode: {self.cfg.intervention.query_mode}")

        self.query_dim = query_dim

        self.memory_bank = MemoryBank(
            key_dim=query_dim,
            value_dim=hidden_dim,
            retrieval_cfg=self.cfg.retrieval,
            forgetting_cfg=self.cfg.forgetting,
            refresh_cfg=self.cfg.intervention.refresh,
            random_memory_control=self.cfg.random_memory_control,
            device=device,
        )
        self.target_builder = TargetBuilder(
            config=self.cfg.target,
            feature_dim=hidden_dim,
            num_classes=self.num_classes,
            device=device,
        )

    def _build_query(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.cfg.intervention.query_mode == "full":
            return hidden

        if self.cfg.intervention.query_mode == "untouched":
            mask = torch.ones(hidden.size(1), dtype=torch.bool, device=hidden.device)
            mask[self.affected_indices] = False
            query = hidden[:, mask]
            if query.size(1) == 0:
                return hidden[:, :1]
            return query

        if self.cfg.intervention.query_mode == "projection":
            return hidden[:, self.query_projection_indices]

        raise ValueError(f"Unsupported query mode: {self.cfg.intervention.query_mode}")

    def _gate_alpha(self, epoch: int) -> float:
        schedule = self.cfg.intervention.gate_schedule
        base_alpha = self.cfg.intervention.gate_alpha
        if schedule == "constant":
            return base_alpha
        if schedule == "linear_warmup":
            warmup = max(self.cfg.intervention.warmup_epochs, 1)
            frac = min(max(epoch / warmup, 0.0), 1.0)
            return base_alpha * frac
        raise ValueError(f"Unsupported gate schedule: {schedule}")

    def _apply_mode(
        self,
        hidden: torch.Tensor,
        retrieved: torch.Tensor,
        epoch: int,
        retrieval_distances: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        idx = self.affected_indices
        pre = hidden

        strength = max(float(self.cfg.intervention.intervention_strength), 0.0)
        if self.cfg.target.value_type == "absolute":
            overwrite_target = pre + strength * (retrieved - pre)
        else:
            overwrite_target = pre + strength * retrieved

        mode = self.cfg.intervention.mode
        post = pre.clone()
        gate_values = torch.zeros((pre.size(0),), device=pre.device)

        if mode == "overwrite":
            post[:, idx] = overwrite_target[:, idx]
            if self.cfg.intervention.intervention_clip > 0:
                clip = float(self.cfg.intervention.intervention_clip)
                delta = torch.clamp(post[:, idx] - pre[:, idx], min=-clip, max=clip)
                post[:, idx] = pre[:, idx] + delta
            return post, gate_values

        if mode == "residual":
            post[:, idx] = pre[:, idx] + strength * retrieved[:, idx]
            if self.cfg.intervention.intervention_clip > 0:
                clip = float(self.cfg.intervention.intervention_clip)
                delta = torch.clamp(post[:, idx] - pre[:, idx], min=-clip, max=clip)
                post[:, idx] = pre[:, idx] + delta
            return post, gate_values

        if mode == "gated":
            base_alpha = self._gate_alpha(epoch)
            if self.cfg.intervention.uncertainty_aware_gating and retrieval_distances is not None and retrieval_distances.numel() > 0:
                dist_mean = retrieval_distances.mean(dim=1)
                logits = -dist_mean / max(float(self.cfg.intervention.gate_temperature), 1e-6)
                sample_alpha = torch.sigmoid(logits)
            else:
                sample_alpha = torch.ones((pre.size(0),), device=pre.device)

            alpha = base_alpha * sample_alpha + float(self.cfg.intervention.gate_bias_init)
            alpha = torch.clamp(alpha, min=0.0, max=1.0)
            gate_values = alpha.detach()
            post[:, idx] = (1.0 - alpha.unsqueeze(1)) * pre[:, idx] + alpha.unsqueeze(1) * overwrite_target[:, idx]
            if self.cfg.intervention.intervention_clip > 0:
                clip = float(self.cfg.intervention.intervention_clip)
                delta = torch.clamp(post[:, idx] - pre[:, idx], min=-clip, max=clip)
                post[:, idx] = pre[:, idx] + delta
            return post, gate_values

        raise ValueError(f"Unsupported intervention mode: {mode}")

    def intervention_fn(self, layer_name: str, hidden: torch.Tensor) -> torch.Tensor:
        if layer_name != self.cfg.intervention.layer:
            return hidden

        if hidden.dim() != 2:
            raise ValueError(
                f"Intervention expects 2D hidden vectors, got shape {tuple(hidden.shape)} at layer {layer_name}"
            )

        self._init_layer_state(hidden)

        self.state.hidden_pre = hidden
        if hidden.requires_grad:
            hidden.retain_grad()

        query = self._build_query(hidden.detach())
        self.state.query_tensor = query.detach()
        self.state.affected_indices = self.affected_indices.detach().clone()

        if not self.state.enabled_for_forward or self.memory_bank is None or self.memory_bank.size == 0:
            self.state.hidden_post = hidden
            self.state.intervention_magnitude = torch.zeros(hidden.size(0), device=hidden.device)
            self.state.gate_values = torch.zeros(hidden.size(0), device=hidden.device)
            self.state.retrieval_top1_agreement = torch.zeros(hidden.size(0), device=hidden.device)
            return hidden

        query_labels = self.current_labels.detach() if self.current_labels is not None else None
        retrieval = self.memory_bank.query(query, query_labels=query_labels)
        retrieved = retrieval.values.to(hidden.device)
        post, gate_values = self._apply_mode(
            hidden,
            retrieved,
            epoch=self.current_epoch,
            retrieval_distances=retrieval.distances,
        )

        self.state.query_result = retrieval
        self.state.hidden_post = post
        self.state.intervention_magnitude = (post - hidden).norm(dim=1).detach()
        self.state.gate_values = gate_values
        if retrieval.labels is not None and query_labels is not None and retrieval.labels.size(1) > 0:
            top1 = retrieval.labels[:, 0]
            self.state.retrieval_top1_agreement = (top1 == query_labels.to(top1.device)).float()
        else:
            self.state.retrieval_top1_agreement = torch.zeros(hidden.size(0), device=hidden.device)
        return post

    def populate_memory(
        self,
        labels: torch.Tensor,
        losses: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        if self.memory_bank is None or self.target_builder is None:
            return {"inserted": 0.0}
        if self.state.hidden_pre is None or self.state.query_tensor is None:
            return {"inserted": 0.0}
        if not self._should_populate(epoch=self.current_epoch):
            return {"inserted": 0.0}

        hidden_pre = self.state.hidden_pre
        hidden_grad = hidden_pre.grad if hidden_pre.grad is not None else None

        targets = self.target_builder.build(
            hidden_pre=hidden_pre,
            labels=labels,
            hidden_grad=hidden_grad,
        )

        inserted = self.memory_bank.insert(
            keys=self.state.query_tensor,
            values=targets.values,
            labels=labels,
            step=self.current_step,
            insertion_policy=self.cfg.insertion_policy,
            losses=losses,
        )

        self.memory_bank.decay_freshness(amount=0.005)
        return {"inserted": float(inserted)}

    def update_usefulness(self, reward: torch.Tensor) -> None:
        if self.memory_bank is None:
            return
        if self.state.query_result is None:
            return
        self.memory_bank.update_usefulness(self.state.query_result.indices, reward)

    def snapshot_metrics(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if self.memory_bank is None:
            metrics.update(
                {
                    "memory_size": 0.0,
                    "memory_avg_age": 0.0,
                    "memory_avg_retrieval_count": 0.0,
                    "memory_avg_usefulness": 0.0,
                    "memory_avg_freshness": 0.0,
                    "memory_footprint_mb": 0.0,
                }
            )
            return metrics

        stats = self.memory_bank.stats(current_step=self.current_step)
        metrics.update(
            {
                "memory_size": float(stats.size),
                "memory_avg_age": stats.avg_age,
                "memory_avg_retrieval_count": stats.avg_retrieval_count,
                "memory_avg_usefulness": stats.avg_usefulness,
                "memory_avg_freshness": stats.avg_freshness,
                "memory_footprint_mb": self.memory_bank.memory_footprint_bytes() / (1024.0**2),
            }
        )

        if self.state.intervention_magnitude is not None:
            metrics["intervention_magnitude_mean"] = float(self.state.intervention_magnitude.mean().item())
        if self.state.gate_values is not None:
            metrics["gate_value_mean"] = float(self.state.gate_values.mean().item())
            metrics["gate_value_std"] = float(self.state.gate_values.std(unbiased=False).item())
        if self.state.query_result is not None:
            metrics["retrieval_purity_mean"] = float(self.state.query_result.purity.float().mean().item())
            metrics["retrieval_distance_mean"] = float(self.state.query_result.distances.float().mean().item())
        if self.state.retrieval_top1_agreement is not None:
            metrics["retrieval_top1_agreement_mean"] = float(self.state.retrieval_top1_agreement.mean().item())
        return metrics

    def export_memory_state(self) -> Dict[str, torch.Tensor]:
        if self.memory_bank is None:
            return {}
        return self.memory_bank.export_state()
