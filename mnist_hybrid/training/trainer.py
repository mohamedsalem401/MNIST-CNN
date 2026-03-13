from __future__ import annotations

import copy
import csv
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from mnist_hybrid.config import ExperimentConfig
from mnist_hybrid.data import DataBundle, build_dataloaders
from mnist_hybrid.evaluation.embedding_knn import EmbeddingKNNModel
from mnist_hybrid.evaluation.metrics import summarize_logits
from mnist_hybrid.memory.intervention import HybridInterventionEngine
from mnist_hybrid.models.base import IntervenableModel
from mnist_hybrid.models.factory import build_model
from mnist_hybrid.utils.common import ensure_dir, get_env_info, save_json, set_seed, resolve_device


def _build_optimizer(config: ExperimentConfig, model: nn.Module) -> torch.optim.Optimizer:
    if config.optim.optimizer.lower() == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.optim.lr,
            weight_decay=config.optim.weight_decay,
        )
    if config.optim.optimizer.lower() == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.optim.lr,
            weight_decay=config.optim.weight_decay,
            momentum=config.optim.momentum,
        )
    raise ValueError(f"Unsupported optimizer: {config.optim.optimizer}")


def _resolve_layer_name(requested: str, candidates: List[str]) -> str:
    if requested in candidates:
        return requested

    if requested == "early":
        return candidates[0]
    if requested == "middle":
        return candidates[len(candidates) // 2]
    if requested == "late":
        return candidates[max(len(candidates) - 2, 0)]
    if requested == "penultimate" and "penultimate" in candidates:
        return "penultimate"

    raise ValueError(f"Intervention layer '{requested}' not in candidates: {candidates}")


def _normalize_hyperparams(config: ExperimentConfig) -> ExperimentConfig:
    cfg = copy.deepcopy(config)

    if cfg.data.train_subset_size > 0:
        cfg.data.max_train_samples = int(cfg.data.train_subset_size)
    if cfg.data.val_subset_size > 0:
        cfg.data.max_val_samples = int(cfg.data.val_subset_size)
    if cfg.data.test_subset_size > 0:
        cfg.data.max_test_samples = int(cfg.data.test_subset_size)

    if cfg.model.architecture == "mlp":
        if cfg.model.growth_enabled and cfg.model.initial_width > 0 and cfg.model.initial_depth > 0:
            cfg.model.hidden_sizes = [int(cfg.model.initial_width)] * int(cfg.model.initial_depth)

        if (not cfg.model.growth_enabled) and cfg.model.capacity_match_baseline:
            if cfg.model.max_width > 0 and cfg.model.max_depth > 0:
                cfg.model.hidden_sizes = [int(cfg.model.max_width)] * int(cfg.model.max_depth)

        if cfg.model.hidden_sizes:
            if cfg.model.initial_width <= 0:
                cfg.model.initial_width = int(cfg.model.hidden_sizes[0])
            if cfg.model.initial_depth <= 0:
                cfg.model.initial_depth = int(len(cfg.model.hidden_sizes))
            if cfg.model.max_width <= 0:
                cfg.model.max_width = int(max(cfg.model.hidden_sizes))
            if cfg.model.max_depth <= 0:
                cfg.model.max_depth = int(len(cfg.model.hidden_sizes))

    mem = cfg.memory
    if mem.memory_enabled is not None:
        mem.enabled = bool(mem.memory_enabled)
    if mem.memory_layer:
        mem.intervention.layer = str(mem.memory_layer)
    if mem.memory_k is not None:
        mem.retrieval.k = int(mem.memory_k)
    if mem.memory_size is not None:
        mem.forgetting.memory_size = int(mem.memory_size)
    if mem.memory_update_policy:
        mem.insertion_policy = str(mem.memory_update_policy)
    if mem.memory_forgetting_policy:
        mem.forgetting.policy = str(mem.memory_forgetting_policy)
    if mem.memory_distance:
        distance = str(mem.memory_distance).lower()
        if distance == "l2":
            distance = "euclidean"
        mem.retrieval.metric = distance
    if mem.memory_query_source:
        mem.intervention.query_mode = str(mem.memory_query_source)
    if mem.memory_value_target:
        mem.target.value_type = str(mem.memory_value_target)
    if mem.memory_intervention_mode:
        mem.intervention.mode = str(mem.memory_intervention_mode)
    if mem.memory_train_enabled is not None:
        mem.intervention.training_use = bool(mem.memory_train_enabled)
    if mem.memory_inference_enabled is not None:
        mem.intervention.inference_use = bool(mem.memory_inference_enabled)

    mem.memory_enabled = bool(mem.enabled)
    mem.memory_layer = str(mem.intervention.layer)
    mem.memory_k = int(mem.retrieval.k)
    mem.memory_size = int(mem.forgetting.memory_size)
    mem.memory_update_policy = str(mem.insertion_policy)
    mem.memory_forgetting_policy = str(mem.forgetting.policy)
    mem.memory_distance = "cosine" if mem.retrieval.metric == "cosine" else "l2"
    mem.memory_query_source = str(mem.intervention.query_mode)
    mem.memory_value_target = str(mem.target.value_type)
    mem.memory_intervention_mode = str(mem.intervention.mode)
    mem.memory_train_enabled = bool(mem.intervention.training_use)
    mem.memory_inference_enabled = bool(mem.intervention.inference_use)

    if cfg.evaluation.n_seeds <= 0:
        cfg.evaluation.n_seeds = len(cfg.seeds) if cfg.seeds else 1

    return cfg


def _maybe_adjust_method(config: ExperimentConfig) -> ExperimentConfig:
    cfg = _normalize_hyperparams(config)

    method = cfg.method
    if method == "nn":
        cfg.memory.enabled = False
    elif method == "nn_large":
        cfg.memory.enabled = False
        if cfg.model.architecture == "mlp":
            cfg.model.hidden_sizes = [int(v * 2) for v in cfg.model.hidden_sizes]
        else:
            cfg.model.latent_dim = int(cfg.model.latent_dim * 2)
            cfg.model.cnn_channels = [int(v * 2) for v in cfg.model.cnn_channels]
    elif method == "nn_extra_layer":
        cfg.memory.enabled = False
        cfg.model.extra_parametric_layer = True
    elif method == "embedding_knn":
        cfg.memory.enabled = False
    elif method == "hybrid":
        cfg.memory.enabled = True
    elif method == "hybrid_inactive":
        cfg.memory.enabled = True
        cfg.memory.intervention.training_use = False
        cfg.memory.intervention.inference_use = False
    elif method == "random_memory":
        cfg.memory.enabled = True
        cfg.memory.random_memory_control = True
    else:
        raise ValueError(f"Unknown method: {method}")

    cfg.memory.memory_enabled = bool(cfg.memory.enabled)
    cfg.memory.memory_train_enabled = bool(cfg.memory.intervention.training_use)
    cfg.memory.memory_inference_enabled = bool(cfg.memory.intervention.inference_use)
    return cfg


def _collect_penultimate_embeddings(
    model: IntervenableModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    embeddings = []
    labels_all = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            out = model.forward_intervenable(images, intervention_fn=None, capture_activations=True)
            emb = out.activations["penultimate"]
            embeddings.append(emb.detach().cpu())
            labels_all.append(labels.detach().cpu())
    return torch.cat(embeddings, dim=0), torch.cat(labels_all, dim=0)


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = _maybe_adjust_method(config)
        self.device = resolve_device(self.config.device)

        set_seed(self.config.seed, deterministic=self.config.deterministic)

        self.output_dir = ensure_dir(Path(self.config.logging.output_root) / self.config.logging.experiment_name)
        self.checkpoint_dir = ensure_dir(self.output_dir / "checkpoints")
        self.analysis_dir = ensure_dir(self.output_dir / "analysis")

        self.data_bundle = build_dataloaders(self.config.data, seed=self.config.seed)
        self.model = build_model(self.config.model).to(self.device)
        self.optimizer = _build_optimizer(self.config, self.model)
        self.criterion = nn.CrossEntropyLoss(reduction="none")

        self.memory_layer_request = str(self.config.memory.intervention.layer)
        self.engine: Optional[HybridInterventionEngine] = None
        self._refresh_memory_layer_resolution()

        if self.config.memory.enabled:
            self.engine = HybridInterventionEngine(
                memory_config=self.config.memory,
                num_classes=self.config.model.num_classes,
            )

        self.global_step = 0
        self.epoch_logs: List[Dict[str, float]] = []

        self.total_train_seconds = 0.0
        self.growth_events: List[Dict[str, object]] = []
        self.memory_resets_due_to_growth = 0
        self.parameter_count_timeline: List[Dict[str, float]] = [
            {
                "epoch": 0.0,
                "global_step": 0.0,
                "parameter_count": float(self._current_param_count()),
            }
        ]
        self.last_growth_epoch = 0
        self.last_growth_step = 0
        self.best_val_for_growth = -float("inf")
        self.epochs_since_growth_improvement = 0

        self.best_val_for_early_stop = -float("inf")
        self.early_stop_bad_epochs = 0

    def _refresh_memory_layer_resolution(self) -> None:
        candidates = self.model.candidate_layers()
        resolved = _resolve_layer_name(self.memory_layer_request, candidates)
        self.config.memory.intervention.layer = resolved
        self.config.memory.memory_layer = resolved
        if self.engine is not None:
            self.engine.cfg.intervention.layer = resolved

    def _intervention_fn(self):
        if not self.engine:
            return None
        return self.engine.intervention_fn

    def _current_param_count(self) -> int:
        return int(sum(p.numel() for p in self.model.parameters()))

    def _growth_allowed(self, epoch: int) -> bool:
        if not self.config.model.growth_enabled:
            return False
        if not self.model.supports_growth():
            return False
        if self.config.model.growth_warmup_epochs > 0 and epoch < self.config.model.growth_warmup_epochs:
            return False
        if self.config.model.growth_stop_epoch > 0 and epoch >= self.config.model.growth_stop_epoch:
            return False
        return True

    def _update_growth_plateau_tracker(self, val_accuracy: float) -> None:
        min_delta = float(self.config.model.growth_min_delta)
        if val_accuracy > self.best_val_for_growth + min_delta:
            self.best_val_for_growth = val_accuracy
            self.epochs_since_growth_improvement = 0
        else:
            self.epochs_since_growth_improvement += 1

    def _should_trigger_growth(self, epoch: int, val_accuracy: float) -> Tuple[bool, str]:
        if not self._growth_allowed(epoch):
            return False, "growth_not_allowed"

        interval = max(int(self.config.model.growth_interval), 1)
        schedule = self.config.model.growth_schedule

        if schedule == "epoch_based":
            return (epoch - self.last_growth_epoch) >= interval, "epoch_interval"

        if schedule == "fixed_step":
            return (self.global_step - self.last_growth_step) >= interval, "step_interval"

        if schedule == "plateau_based":
            return self.epochs_since_growth_improvement >= interval, "val_plateau"

        if schedule == "performance_triggered":
            threshold = float(self.config.model.growth_performance_threshold)
            if val_accuracy >= threshold and (epoch - self.last_growth_epoch) >= interval:
                return True, "val_threshold"
            return False, "val_threshold_not_met"

        raise ValueError(f"Unsupported growth schedule: {schedule}")

    def _maybe_grow(self, epoch: int, val_accuracy: float) -> Optional[Dict[str, object]]:
        should_grow, trigger_reason = self._should_trigger_growth(epoch, val_accuracy)
        if not should_grow:
            return None

        before_params = self._current_param_count()
        growth = self.model.grow(
            growth_mode=self.config.model.growth_mode,
            growth_amount_width=int(self.config.model.growth_amount_width),
            growth_amount_depth=int(self.config.model.growth_amount_depth),
            max_width=int(self.config.model.max_width),
            max_depth=int(self.config.model.max_depth),
        )
        if not growth.get("grew", False):
            return None

        self.optimizer = _build_optimizer(self.config, self.model)
        self.last_growth_epoch = int(epoch)
        self.last_growth_step = int(self.global_step)

        if self.engine is not None:
            self.engine.reset_dynamic_state()
            self.memory_resets_due_to_growth += 1

        self._refresh_memory_layer_resolution()

        after_params = self._current_param_count()
        event = {
            "epoch": int(epoch),
            "global_step": int(self.global_step),
            "trigger_reason": trigger_reason,
            "val_accuracy_at_trigger": float(val_accuracy),
            "param_count_before": int(before_params),
            "param_count_after": int(after_params),
            **growth,
        }
        self.growth_events.append(event)
        self.parameter_count_timeline.append(
            {
                "epoch": float(epoch),
                "global_step": float(self.global_step),
                "parameter_count": float(after_params),
            }
        )
        return event

    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        start = time.perf_counter()
        total_loss = 0.0
        total_correct = 0.0
        total_samples = 0

        memory_inserted = 0.0
        running_magnitude = 0.0
        running_purity = 0.0
        running_distance = 0.0
        running_gate_mean = 0.0
        running_top1_agreement = 0.0
        running_gate_reg_loss = 0.0
        running_intervention_count = 0

        for images, labels in self.data_bundle.train:
            images = images.to(self.device)
            labels = labels.to(self.device)

            if self.engine:
                self.engine.begin_batch(labels=labels, epoch=epoch, step=self.global_step, training=True)

            self.optimizer.zero_grad(set_to_none=True)
            out = self.model.forward_intervenable(
                images,
                intervention_fn=self._intervention_fn(),
                capture_activations=False,
            )
            logits = out.logits
            losses = self.criterion(logits, labels)
            base_loss = losses.mean()
            gate_reg_loss = torch.tensor(0.0, device=logits.device)

            if self.engine and self.engine.state.gate_values is not None:
                reg = float(self.config.memory.intervention.gate_regularization)
                if reg > 0:
                    gate_reg_loss = reg * self.engine.state.gate_values.pow(2).mean()

            loss = base_loss + gate_reg_loss
            loss.backward()

            if self.config.optim.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.optim.grad_clip_norm)

            self.optimizer.step()

            preds = logits.argmax(dim=1)
            reward = torch.where(preds == labels, 1.0, -1.0).to(torch.float32)

            if self.engine:
                self.engine.update_usefulness(reward)
                memory_info = self.engine.populate_memory(labels=labels, losses=losses.detach())
                memory_inserted += memory_info.get("inserted", 0.0)

                if self.engine.state.intervention_magnitude is not None:
                    running_magnitude += float(self.engine.state.intervention_magnitude.mean().item())
                    running_intervention_count += 1
                if self.engine.state.query_result is not None:
                    running_purity += float(self.engine.state.query_result.purity.mean().item())
                    running_distance += float(self.engine.state.query_result.distances.mean().item())
                if self.engine.state.gate_values is not None:
                    running_gate_mean += float(self.engine.state.gate_values.mean().item())
                if self.engine.state.retrieval_top1_agreement is not None:
                    running_top1_agreement += float(self.engine.state.retrieval_top1_agreement.mean().item())
                running_gate_reg_loss += float(gate_reg_loss.item())

            batch_size = labels.size(0)
            total_samples += batch_size
            total_loss += float(losses.sum().item())
            total_correct += float((preds == labels).sum().item())
            self.global_step += 1

        elapsed = time.perf_counter() - start
        avg_loss = total_loss / max(total_samples, 1)
        avg_acc = total_correct / max(total_samples, 1)
        throughput = total_samples / max(elapsed, 1e-6)

        metrics: Dict[str, float] = {
            "train_loss": avg_loss,
            "train_accuracy": avg_acc,
            "train_samples": float(total_samples),
            "train_epoch_seconds": elapsed,
            "train_throughput_samples_per_sec": throughput,
            "memory_inserted": memory_inserted,
            "parameter_count": float(self._current_param_count()),
        }

        if running_intervention_count > 0:
            metrics["intervention_magnitude_mean"] = running_magnitude / running_intervention_count
        if self.engine and running_intervention_count > 0:
            metrics["retrieval_purity_mean"] = running_purity / running_intervention_count
            metrics["retrieval_distance_mean"] = running_distance / running_intervention_count
            metrics["gate_value_mean"] = running_gate_mean / running_intervention_count
            metrics["retrieval_top1_agreement_mean"] = running_top1_agreement / running_intervention_count
            metrics["gate_regularization_loss_mean"] = running_gate_reg_loss / running_intervention_count
            metrics.update(self.engine.snapshot_metrics())

        return metrics

    @torch.no_grad()
    def _evaluate_loader(
        self,
        loader: torch.utils.data.DataLoader,
        split_name: str,
        epoch: int,
        memory_use_override: Optional[bool] = None,
        collect_sample_details: bool = False,
    ) -> Tuple[Dict[str, float], Dict[str, object]]:
        self.model.eval()

        original_inference_use = None
        if self.engine:
            original_inference_use = self.engine.cfg.intervention.inference_use
            if memory_use_override is not None:
                self.engine.cfg.intervention.inference_use = memory_use_override

        all_logits = []
        all_labels = []
        all_losses = []
        per_sample: Dict[str, object] = {
            "loss": [],
            "pred": [],
            "label": [],
            "confidence": [],
            "memory_intervention_magnitude": [],
            "retrieval_distance": [],
            "retrieval_purity": [],
            "retrieval_top1_agreement": [],
            "gate_value": [],
        }

        gate_values: List[float] = []
        agreement_values: List[float] = []

        start = time.perf_counter()
        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            if self.engine:
                self.engine.begin_batch(labels=labels, epoch=epoch, step=self.global_step, training=False)

            out = self.model.forward_intervenable(
                images,
                intervention_fn=self._intervention_fn(),
                capture_activations=False,
            )
            logits = out.logits
            losses = self.criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            confidence = probs.max(dim=1).values

            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())
            all_losses.append(losses.detach().cpu())

            if self.engine and self.engine.state.gate_values is not None:
                gate_values.extend(self.engine.state.gate_values.detach().cpu().tolist())
            if self.engine and self.engine.state.retrieval_top1_agreement is not None:
                agreement_values.extend(self.engine.state.retrieval_top1_agreement.detach().cpu().tolist())

            if collect_sample_details:
                per_sample["loss"].extend(losses.detach().cpu().tolist())
                per_sample["pred"].extend(preds.detach().cpu().tolist())
                per_sample["label"].extend(labels.detach().cpu().tolist())
                per_sample["confidence"].extend(confidence.detach().cpu().tolist())

                if self.engine and self.engine.state.intervention_magnitude is not None:
                    per_sample["memory_intervention_magnitude"].extend(
                        self.engine.state.intervention_magnitude.detach().cpu().tolist()
                    )
                else:
                    per_sample["memory_intervention_magnitude"].extend([0.0] * labels.size(0))

                if self.engine and self.engine.state.query_result is not None:
                    q = self.engine.state.query_result
                    dist = q.distances.mean(dim=1).detach().cpu().tolist()
                    purity = q.purity.detach().cpu().tolist()
                    per_sample["retrieval_distance"].extend(dist)
                    per_sample["retrieval_purity"].extend(purity)
                else:
                    per_sample["retrieval_distance"].extend([0.0] * labels.size(0))
                    per_sample["retrieval_purity"].extend([0.0] * labels.size(0))

                if self.engine and self.engine.state.retrieval_top1_agreement is not None:
                    per_sample["retrieval_top1_agreement"].extend(
                        self.engine.state.retrieval_top1_agreement.detach().cpu().tolist()
                    )
                else:
                    per_sample["retrieval_top1_agreement"].extend([0.0] * labels.size(0))

                if self.engine and self.engine.state.gate_values is not None:
                    per_sample["gate_value"].extend(self.engine.state.gate_values.detach().cpu().tolist())
                else:
                    per_sample["gate_value"].extend([0.0] * labels.size(0))

        elapsed = time.perf_counter() - start

        logits_tensor = torch.cat(all_logits, dim=0)
        labels_tensor = torch.cat(all_labels, dim=0)
        losses_tensor = torch.cat(all_losses, dim=0)

        summary, details = summarize_logits(
            logits=logits_tensor,
            labels=labels_tensor,
            losses=losses_tensor,
            ece_bins=self.config.evaluation.ece_bins,
            with_per_class=self.config.evaluation.per_class_metrics,
            with_confusion=self.config.evaluation.confusion_matrix,
        )

        metrics = {
            f"{split_name}_loss": summary["loss"],
            f"{split_name}_accuracy": summary["accuracy"],
            f"{split_name}_ece": summary["ece"],
            f"{split_name}_samples": float(labels_tensor.size(0)),
            f"{split_name}_seconds": elapsed,
            f"{split_name}_throughput_samples_per_sec": labels_tensor.size(0) / max(elapsed, 1e-6),
        }

        if gate_values:
            gate_tensor = torch.tensor(gate_values)
            metrics[f"{split_name}_gate_value_mean"] = float(gate_tensor.mean().item())
            metrics[f"{split_name}_gate_value_std"] = float(gate_tensor.std(unbiased=False).item())

        if agreement_values:
            agreement_tensor = torch.tensor(agreement_values)
            metrics[f"{split_name}_retrieval_top1_agreement_mean"] = float(agreement_tensor.mean().item())

        output_details: Dict[str, object] = {
            "summary": summary,
            "details": details,
            "per_sample": per_sample if collect_sample_details else {},
        }

        if self.engine:
            metrics.update({f"{split_name}_{k}": v for k, v in self.engine.snapshot_metrics().items()})
            if memory_use_override is not None and original_inference_use is not None:
                self.engine.cfg.intervention.inference_use = original_inference_use

        return metrics, output_details

    def _evaluate_embedding_knn(self, epoch: int) -> Dict[str, float]:
        train_emb, train_labels = _collect_penultimate_embeddings(
            model=self.model,
            loader=self.data_bundle.train,
            device=self.device,
        )
        knn = EmbeddingKNNModel(
            embeddings=train_emb,
            labels=train_labels,
            k=self.config.memory.retrieval.k,
            metric=self.config.memory.retrieval.metric,
        )

        metrics: Dict[str, float] = {}
        for split_name, loader in [
            ("val", self.data_bundle.val),
            ("test", self.data_bundle.test),
        ]:
            emb, labels = _collect_penultimate_embeddings(self.model, loader, self.device)
            preds = knn.predict(emb.to(self.device)).cpu()
            acc = (preds == labels).float().mean().item()
            metrics[f"{split_name}_accuracy"] = float(acc)
            metrics[f"{split_name}_loss"] = float("nan")
            metrics[f"{split_name}_ece"] = float("nan")

        metrics["model_parameter_count"] = float(self._current_param_count())
        return metrics

    def _build_growth_performance_deltas(self) -> List[Dict[str, float]]:
        if not self.growth_events:
            return []

        by_epoch = {int(row["epoch"]): row for row in self.epoch_logs}
        deltas: List[Dict[str, float]] = []
        for event in self.growth_events:
            epoch = int(event["epoch"])
            pre = float(by_epoch.get(epoch, {}).get("val_accuracy", float("nan")))
            post = float(by_epoch.get(epoch + 1, {}).get("val_accuracy", float("nan")))
            delta = float("nan")
            if not math.isnan(pre) and not math.isnan(post):
                delta = post - pre
            deltas.append(
                {
                    "growth_epoch": float(epoch),
                    "val_accuracy_pre": pre,
                    "val_accuracy_post": post,
                    "val_accuracy_delta": delta,
                }
            )
        return deltas

    def run(self) -> Dict[str, object]:
        final_epoch = self.config.optim.epochs

        for epoch in range(1, self.config.optim.epochs + 1):
            train_metrics = self._train_one_epoch(epoch)
            self.total_train_seconds += float(train_metrics.get("train_epoch_seconds", 0.0))
            val_metrics, _ = self._evaluate_loader(self.data_bundle.val, split_name="val", epoch=epoch)

            row = {
                "epoch": float(epoch),
                **train_metrics,
                **val_metrics,
                "parameter_count": float(self._current_param_count()),
                "growth_triggered": 0.0,
            }
            self.epoch_logs.append(row)

            self._update_growth_plateau_tracker(float(row["val_accuracy"]))
            growth_event = self._maybe_grow(epoch=epoch, val_accuracy=float(row["val_accuracy"]))
            if growth_event is not None:
                row["growth_triggered"] = 1.0
                row["parameter_count"] = float(growth_event["param_count_after"])

            print(
                f"Epoch {epoch:03d}/{self.config.optim.epochs} "
                f"train_acc={row['train_accuracy'] * 100:.2f}% "
                f"val_acc={row['val_accuracy'] * 100:.2f}% "
                f"params={int(row['parameter_count'])} "
                f"memory_size={row.get('memory_size', 0):.0f}"
            )

            if self.config.optim.early_stopping:
                current_val = float(row["val_accuracy"])
                if current_val > self.best_val_for_early_stop + 1e-6:
                    self.best_val_for_early_stop = current_val
                    self.early_stop_bad_epochs = 0
                else:
                    self.early_stop_bad_epochs += 1

                patience = max(int(self.config.optim.early_stopping_patience), 1)
                if self.early_stop_bad_epochs >= patience:
                    final_epoch = epoch
                    print(f"Early stopping at epoch {epoch} (patience={patience}).")
                    break

            if self.config.logging.save_checkpoints:
                ckpt_path = self.checkpoint_dir / f"epoch_{epoch:03d}.pt"
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "config": asdict(self.config),
                        "epoch": epoch,
                        "global_step": self.global_step,
                    },
                    ckpt_path,
                )

        if self.config.method == "embedding_knn":
            final_metrics = self._evaluate_embedding_knn(epoch=final_epoch)
            final_details = {}
        else:
            val_metrics, val_details = self._evaluate_loader(
                self.data_bundle.val,
                split_name="val",
                epoch=final_epoch,
                collect_sample_details=True,
            )
            test_metrics, test_details = self._evaluate_loader(
                self.data_bundle.test,
                split_name="test",
                epoch=final_epoch,
                collect_sample_details=True,
            )
            final_metrics = {**val_metrics, **test_metrics}
            final_details = {
                "val": val_details,
                "test": test_details,
            }

        toggle_eval = {}
        if self.engine and self.config.evaluation.run_memory_toggle_eval:
            with_mem, with_details = self._evaluate_loader(
                self.data_bundle.test,
                split_name="test_with_memory",
                epoch=final_epoch,
                memory_use_override=True,
                collect_sample_details=True,
            )
            no_mem, no_details = self._evaluate_loader(
                self.data_bundle.test,
                split_name="test_without_memory",
                epoch=final_epoch,
                memory_use_override=False,
                collect_sample_details=True,
            )

            losses_with = torch.tensor(with_details["per_sample"]["loss"])
            losses_without = torch.tensor(no_details["per_sample"]["loss"])
            loss_delta = losses_with - losses_without
            threshold = abs(float(self.config.evaluation.strong_effect_threshold))

            toggle_eval = {
                **with_mem,
                **no_mem,
                "test_helped_fraction": float((loss_delta < 0).float().mean().item()),
                "test_harmed_fraction": float((loss_delta > 0).float().mean().item()),
                "test_neutral_fraction": float((loss_delta == 0).float().mean().item()),
                "test_strong_help_fraction": float((loss_delta < -threshold).float().mean().item()),
                "test_strong_harm_fraction": float((loss_delta > threshold).float().mean().item()),
                "test_loss_delta_mean": float(loss_delta.mean().item()),
                "test_loss_delta_std": float(loss_delta.std(unbiased=False).item()),
            }

            final_details["toggle_eval"] = {
                "with_memory": with_details,
                "without_memory": no_details,
                "loss_delta": loss_delta.tolist(),
            }

        final_metrics["model_parameter_count"] = float(self._current_param_count())
        final_metrics["growth_event_count"] = float(len(self.growth_events))
        final_metrics["training_seconds_total"] = float(self.total_train_seconds)
        final_metrics["memory_resets_due_to_growth"] = float(self.memory_resets_due_to_growth)
        if self.model.supports_growth():
            for key, value in self.model.growth_state().items():
                final_metrics[f"model_{key}"] = float(value)

        if self.engine and self.config.logging.save_memory_snapshots:
            torch.save(self.engine.export_memory_state(), self.analysis_dir / "memory_state.pt")

        growth_summary = {
            "enabled": bool(self.config.model.growth_enabled),
            "events": self.growth_events,
            "parameter_count_timeline": self.parameter_count_timeline,
            "performance_around_events": self._build_growth_performance_deltas(),
            "memory_resets_due_to_growth": int(self.memory_resets_due_to_growth),
        }

        results = {
            "config": asdict(self.config),
            "env": get_env_info(),
            "epoch_logs": self.epoch_logs,
            "final_metrics": final_metrics,
            "toggle_eval": toggle_eval,
            "growth": growth_summary,
        }

        save_json(results, self.output_dir / "metrics.json")
        self._write_epoch_csv(self.output_dir / "epoch_logs.csv")
        self._write_details(final_details, self.output_dir / "details.pt")

        return {
            "results": results,
            "details": final_details,
            "output_dir": str(self.output_dir),
        }

    def _write_epoch_csv(self, path: Path) -> None:
        if not self.epoch_logs:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = sorted({k for row in self.epoch_logs for k in row.keys()})
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for row in self.epoch_logs:
                writer.writerow(row)

    def _write_details(self, details: Dict[str, object], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(details, path)
