from __future__ import annotations

import copy
import csv
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from mnist_hybrid.config import ExperimentConfig
from mnist_hybrid.data import DataBundle, build_dataloaders
from mnist_hybrid.evaluation.embedding_knn import EmbeddingKNNModel
from mnist_hybrid.evaluation.metrics import cross_entropy_per_sample, summarize_logits
from mnist_hybrid.memory.intervention import HybridInterventionEngine
from mnist_hybrid.models.base import IntervenableModel
from mnist_hybrid.models.factory import build_model
from mnist_hybrid.utils.common import ensure_dir, get_env_info, resolve_device, save_json, set_seed


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


def _maybe_adjust_method(config: ExperimentConfig) -> ExperimentConfig:
    cfg = copy.deepcopy(config)

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

        candidates = self.model.candidate_layers()
        self.config.memory.intervention.layer = _resolve_layer_name(
            self.config.memory.intervention.layer,
            candidates,
        )

        self.engine: Optional[HybridInterventionEngine] = None
        if self.config.memory.enabled:
            self.engine = HybridInterventionEngine(
                memory_config=self.config.memory,
                num_classes=self.config.model.num_classes,
            )

        self.global_step = 0
        self.epoch_logs: List[Dict[str, float]] = []

    def _intervention_fn(self):
        if not self.engine:
            return None
        return self.engine.intervention_fn

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
            loss = losses.mean()
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
        }

        if running_intervention_count > 0:
            metrics["intervention_magnitude_mean"] = running_magnitude / running_intervention_count
        if self.engine and running_intervention_count > 0:
            metrics["retrieval_purity_mean"] = running_purity / running_intervention_count
            metrics["retrieval_distance_mean"] = running_distance / running_intervention_count
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
        }

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

        output_details: Dict[str, object] = {
            "summary": summary,
            "details": details,
            "per_sample": per_sample if collect_sample_details else {},
        }

        if self.engine:
            metrics.update({f"{split_name}_{k}": v for k, v in self.engine.snapshot_metrics().items()})
            if memory_use_override is not None:
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

        return metrics

    def run(self) -> Dict[str, object]:
        for epoch in range(1, self.config.optim.epochs + 1):
            train_metrics = self._train_one_epoch(epoch)
            val_metrics, _ = self._evaluate_loader(self.data_bundle.val, split_name="val", epoch=epoch)

            row = {
                "epoch": float(epoch),
                **train_metrics,
                **val_metrics,
            }
            self.epoch_logs.append(row)

            print(
                f"Epoch {epoch:03d}/{self.config.optim.epochs} "
                f"train_acc={row['train_accuracy'] * 100:.2f}% "
                f"val_acc={row['val_accuracy'] * 100:.2f}% "
                f"memory_size={row.get('memory_size', 0):.0f}"
            )

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
            final_metrics = self._evaluate_embedding_knn(epoch=self.config.optim.epochs)
            final_details = {}
        else:
            val_metrics, val_details = self._evaluate_loader(
                self.data_bundle.val,
                split_name="val",
                epoch=self.config.optim.epochs,
                collect_sample_details=True,
            )
            test_metrics, test_details = self._evaluate_loader(
                self.data_bundle.test,
                split_name="test",
                epoch=self.config.optim.epochs,
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
                epoch=self.config.optim.epochs,
                memory_use_override=True,
                collect_sample_details=True,
            )
            no_mem, no_details = self._evaluate_loader(
                self.data_bundle.test,
                split_name="test_without_memory",
                epoch=self.config.optim.epochs,
                memory_use_override=False,
                collect_sample_details=True,
            )

            losses_with = torch.tensor(with_details["per_sample"]["loss"])
            losses_without = torch.tensor(no_details["per_sample"]["loss"])
            loss_delta = losses_with - losses_without

            toggle_eval = {
                **with_mem,
                **no_mem,
                "test_helped_fraction": float((loss_delta < 0).float().mean().item()),
                "test_harmed_fraction": float((loss_delta > 0).float().mean().item()),
                "test_neutral_fraction": float((loss_delta == 0).float().mean().item()),
                "test_loss_delta_mean": float(loss_delta.mean().item()),
                "test_loss_delta_std": float(loss_delta.std(unbiased=False).item()),
            }

            final_details["toggle_eval"] = {
                "with_memory": with_details,
                "without_memory": no_details,
                "loss_delta": loss_delta.tolist(),
            }

        if self.engine and self.config.logging.save_memory_snapshots:
            torch.save(self.engine.export_memory_state(), self.analysis_dir / "memory_state.pt")

        results = {
            "config": asdict(self.config),
            "env": get_env_info(),
            "epoch_logs": self.epoch_logs,
            "final_metrics": final_metrics,
            "toggle_eval": toggle_eval,
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
