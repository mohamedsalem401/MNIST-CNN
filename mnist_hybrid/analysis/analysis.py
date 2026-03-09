from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def read_epoch_logs(path: str | Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v)
                except (TypeError, ValueError):
                    continue
            rows.append(parsed)
    return rows


def plot_training_curves(epoch_logs_path: str | Path, output_dir: str | Path) -> Path:
    rows = read_epoch_logs(epoch_logs_path)
    epochs = [int(r.get("epoch", idx + 1)) for idx, r in enumerate(rows)]
    train_acc = [r.get("train_accuracy", np.nan) for r in rows]
    val_acc = [r.get("val_accuracy", np.nan) for r in rows]
    train_loss = [r.get("train_loss", np.nan) for r in rows]
    val_loss = [r.get("val_loss", np.nan) for r in rows]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "training_curves.png"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, train_acc, label="train_acc")
    axes[0].plot(epochs, val_acc, label="val_acc")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy Curves")
    axes[0].legend()

    axes[1].plot(epochs, train_loss, label="train_loss")
    axes[1].plot(epochs, val_loss, label="val_loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss Curves")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _extract_toggle_loss_delta(details: Dict[str, object]) -> np.ndarray:
    toggle = details.get("toggle_eval", {})
    loss_delta = toggle.get("loss_delta", [])
    return np.array(loss_delta, dtype=np.float64)


def plot_help_harm_hist(details_path: str | Path, output_dir: str | Path) -> Path:
    details = torch.load(details_path, map_location="cpu")
    deltas = _extract_toggle_loss_delta(details)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "loss_delta_histogram.png"

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(deltas, bins=40, color="#2F4F4F", alpha=0.85)
    ax.axvline(0.0, color="#B22222", linestyle="--", linewidth=1)
    ax.set_title("Per-sample Loss Delta (with memory - without memory)")
    ax.set_xlabel("Loss Delta")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_retrieval_distance_vs_benefit(details_path: str | Path, output_dir: str | Path) -> Path:
    details = torch.load(details_path, map_location="cpu")
    toggle = details.get("toggle_eval", {})
    with_memory = toggle.get("with_memory", {}).get("per_sample", {})
    deltas = np.array(toggle.get("loss_delta", []), dtype=np.float64)
    distance = np.array(with_memory.get("retrieval_distance", []), dtype=np.float64)

    if deltas.size == 0 or distance.size == 0:
        raise ValueError("Missing toggle eval or retrieval distance data for plotting")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "retrieval_distance_vs_benefit.png"

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(distance, -deltas, s=10, alpha=0.35)
    ax.set_xlabel("Average Neighbor Distance")
    ax.set_ylabel("Benefit (negative loss delta)")
    ax.set_title("Retrieval Distance vs Benefit")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_memory_state(memory_state_path: str | Path, output_dir: str | Path) -> Tuple[Path, Path]:
    state = torch.load(memory_state_path, map_location="cpu")
    insert_step = state.get("insert_step", torch.tensor([])).float()
    usefulness = state.get("usefulness", torch.tensor([])).float()
    retrieval_count = state.get("retrieval_count", torch.tensor([])).float()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    path_age = out / "memory_insert_step_hist.png"
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(insert_step.numpy(), bins=40, color="#4682B4", alpha=0.85)
    ax.set_title("Memory Insert Step Distribution")
    ax.set_xlabel("Insert Step")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(path_age, dpi=200)
    plt.close(fig)

    path_usefulness = out / "memory_usefulness_vs_usage.png"
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(retrieval_count.numpy(), usefulness.numpy(), s=14, alpha=0.45)
    ax.set_xlabel("Retrieval Count")
    ax.set_ylabel("Usefulness Score")
    ax.set_title("Memory Usage vs Usefulness")
    fig.tight_layout()
    fig.savefig(path_usefulness, dpi=200)
    plt.close(fig)

    return path_age, path_usefulness


def plot_latent_projection(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_dir: str | Path,
    method: str = "pca",
    max_samples: int = 2000,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if embeddings.shape[0] > max_samples:
        indices = np.random.RandomState(0).choice(embeddings.shape[0], size=max_samples, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]

    if method == "pca":
        proj = PCA(n_components=2, random_state=0).fit_transform(embeddings)
        title = "PCA"
        fname = "latent_pca.png"
    elif method == "tsne":
        proj = TSNE(n_components=2, random_state=0, init="pca", learning_rate="auto").fit_transform(embeddings)
        title = "t-SNE"
        fname = "latent_tsne.png"
    else:
        raise ValueError(f"Unsupported projection method: {method}")

    path = out / fname
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=labels, cmap="tab10", s=8, alpha=0.8)
    ax.set_title(f"Latent Space ({title})")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path
