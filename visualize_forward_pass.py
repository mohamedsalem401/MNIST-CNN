from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from torchvision import datasets, transforms

from mnist_nn.model import MNISTMLP

DEFAULT_MEAN = 0.1307
DEFAULT_STD = 0.3081


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize NN node/edge activity for one MNIST input.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/mnist_mlp.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--index", type=int, default=0, help="Sample index from the selected split")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--gif-name", type=str, default="mnist_forward_pass.gif")
    parser.add_argument("--final-image-name", type=str, default="mnist_forward_final.png")
    parser.add_argument("--input-image-name", type=str, default="mnist_input.png")
    parser.add_argument("--input-nodes", type=int, default=48)
    parser.add_argument("--hidden-nodes", type=int, default=24)
    parser.add_argument("--edge-quantile", type=float, default=0.75)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args()


def pick_top_indices(values: np.ndarray, k: int, by_absolute: bool = True) -> np.ndarray:
    k = min(max(k, 1), values.size)
    score = np.abs(values) if by_absolute else values
    if k == values.size:
        selected = np.arange(values.size)
    else:
        selected = np.argpartition(score, -k)[-k:]

    # Keep visual ordering deterministic from top to bottom.
    return np.sort(selected.astype(int))


def normalize(values: np.ndarray) -> np.ndarray:
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if np.isclose(vmin, vmax):
        return np.zeros_like(values, dtype=np.float64)
    return (values - vmin) / (vmax - vmin)


def load_model_and_sample(args: argparse.Namespace):
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    hidden_sizes = checkpoint.get("hidden_sizes", [128, 64])
    mean = float(checkpoint.get("normalize_mean", DEFAULT_MEAN))
    std = float(checkpoint.get("normalize_std", DEFAULT_STD))

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = MNISTMLP(hidden_sizes=hidden_sizes)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    is_train = args.split == "train"
    raw_dataset = datasets.MNIST(
        root=args.data_dir,
        train=is_train,
        download=True,
        transform=transforms.ToTensor(),
    )
    normalized_dataset = datasets.MNIST(
        root=args.data_dir,
        train=is_train,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((mean,), (std,)),
            ]
        ),
    )

    if args.index < 0 or args.index >= len(raw_dataset):
        raise IndexError(f"Index {args.index} is out of range for {args.split} dataset")

    input_raw, target = raw_dataset[args.index]
    input_norm, _ = normalized_dataset[args.index]

    with torch.no_grad():
        trace = model.forward_with_trace(input_norm.unsqueeze(0).to(device))

    activations = [layer.squeeze(0).detach().cpu().numpy() for layer in trace.activations]
    probabilities = trace.probabilities.squeeze(0).detach().cpu().numpy()
    prediction = int(np.argmax(probabilities))

    weights = [layer.weight.detach().cpu().numpy() for layer in model.layers]

    return {
        "model": model,
        "weights": weights,
        "activations": activations,
        "probabilities": probabilities,
        "prediction": prediction,
        "target": int(target),
        "input_raw": input_raw.squeeze(0).numpy(),
    }


def prepare_layer_indices(
    activations: Sequence[np.ndarray],
    input_raw: np.ndarray,
    input_nodes: int,
    hidden_nodes: int,
) -> List[np.ndarray]:
    layer_indices: List[np.ndarray] = []

    # Input layer uses most-intense pixels so the selected nodes align with what the image shows.
    input_flat = input_raw.reshape(-1)
    layer_indices.append(pick_top_indices(input_flat, input_nodes, by_absolute=False))

    for hidden_activation in activations[1:-1]:
        layer_indices.append(pick_top_indices(hidden_activation, hidden_nodes, by_absolute=True))

    # Output layer has 10 digits; display all classes.
    layer_indices.append(np.arange(activations[-1].size))

    return layer_indices


def build_positions(layer_indices: Sequence[np.ndarray]) -> Dict[Tuple[int, int], Tuple[float, float]]:
    positions: Dict[Tuple[int, int], Tuple[float, float]] = {}
    x_positions = np.linspace(0.08, 0.92, len(layer_indices))

    for layer_idx, node_ids in enumerate(layer_indices):
        y_positions = np.linspace(0.08, 0.92, len(node_ids))
        for y, node_id in zip(y_positions, node_ids):
            positions[(layer_idx, int(node_id))] = (float(x_positions[layer_idx]), float(y))

    return positions


def build_edges(
    weights: Sequence[np.ndarray],
    activations: Sequence[np.ndarray],
    layer_indices: Sequence[np.ndarray],
    edge_quantile: float,
) -> List[List[Tuple[int, int, float]]]:
    all_edges: List[List[Tuple[int, int, float]]] = []
    edge_quantile = float(np.clip(edge_quantile, 0.0, 1.0))

    for layer_idx, weight_matrix in enumerate(weights):
        src_ids = layer_indices[layer_idx]
        dst_ids = layer_indices[layer_idx + 1]

        raw_edges: List[Tuple[int, int, float]] = []
        for dst in dst_ids:
            contributions = weight_matrix[dst, src_ids] * activations[layer_idx][src_ids]
            for src, contrib in zip(src_ids, contributions):
                raw_edges.append((int(src), int(dst), float(contrib)))

        magnitudes = np.array([abs(edge[2]) for edge in raw_edges], dtype=np.float64)
        if magnitudes.size == 0:
            all_edges.append([])
            continue

        cutoff = float(np.quantile(magnitudes, edge_quantile))
        filtered = [edge for edge in raw_edges if abs(edge[2]) >= cutoff]
        all_edges.append(filtered if filtered else raw_edges)

    return all_edges


def render_visualization(args: argparse.Namespace) -> None:
    loaded = load_model_and_sample(args)

    activations = loaded["activations"]
    weights = loaded["weights"]
    input_raw = loaded["input_raw"]
    prediction = loaded["prediction"]
    target = loaded["target"]
    probabilities = loaded["probabilities"]

    layer_indices = prepare_layer_indices(
        activations=activations,
        input_raw=input_raw,
        input_nodes=args.input_nodes,
        hidden_nodes=args.hidden_nodes,
    )
    positions = build_positions(layer_indices)
    edges_by_layer = build_edges(
        weights=weights,
        activations=activations,
        layer_indices=layer_indices,
        edge_quantile=args.edge_quantile,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_image_path = args.output_dir / args.input_image_name
    gif_path = args.output_dir / args.gif_name
    final_image_path = args.output_dir / args.final_image_name

    plt.imsave(input_image_path, input_raw, cmap="gray")

    num_layers = len(layer_indices)
    layer_names = ["Input"] + [f"Hidden {i}" for i in range(1, num_layers - 1)] + ["Output"]

    fig, (ax_img, ax_net) = plt.subplots(
        1,
        2,
        figsize=(13, 6),
        gridspec_kw={"width_ratios": [1.0, 2.6]},
    )

    input_selected = layer_indices[0]
    input_rows = input_selected // 28
    input_cols = input_selected % 28

    def draw_stage(stage: int) -> None:
        ax_img.clear()
        ax_img.imshow(input_raw, cmap="gray", vmin=0.0, vmax=1.0)
        ax_img.scatter(
            input_cols,
            input_rows,
            s=22,
            facecolors="none",
            edgecolors="red",
            linewidths=0.7,
        )
        ax_img.set_title(f"Input Image\nLabel={target}")
        ax_img.set_xticks([])
        ax_img.set_yticks([])

        ax_net.clear()
        ax_net.set_xlim(0.0, 1.0)
        ax_net.set_ylim(0.0, 1.0)
        ax_net.axis("off")

        for edge_layer in range(stage):
            edges = edges_by_layer[edge_layer]
            if not edges:
                continue

            mags = np.array([abs(edge[2]) for edge in edges], dtype=np.float64)
            mags_norm = normalize(mags)

            for edge, mag_norm in zip(edges, mags_norm):
                src, dst, contrib = edge
                x1, y1 = positions[(edge_layer, src)]
                x2, y2 = positions[(edge_layer + 1, dst)]
                color = "#D1495B" if contrib >= 0 else "#2B59C3"
                alpha = 0.12 + 0.85 * float(mag_norm)
                width = 0.5 + 2.4 * float(mag_norm)
                ax_net.plot([x1, x2], [y1, y2], color=color, linewidth=width, alpha=alpha)

        for layer_idx, node_ids in enumerate(layer_indices):
            xs = []
            ys = []
            for node_id in node_ids:
                x, y = positions[(layer_idx, int(node_id))]
                xs.append(x)
                ys.append(y)

            if layer_idx <= stage:
                layer_values = activations[layer_idx][node_ids]
                node_colors = plt.cm.YlOrRd(normalize(layer_values))
            else:
                node_colors = ["#CFCFCF"] * len(node_ids)

            ax_net.scatter(xs, ys, s=160, c=node_colors, edgecolors="#202020", linewidths=0.35, zorder=3)
            ax_net.text(
                xs[0],
                0.985,
                layer_names[layer_idx],
                ha="center",
                va="top",
                fontsize=10,
                fontweight="bold",
            )

            if layer_idx == num_layers - 1:
                for y, digit, prob in zip(ys, node_ids, probabilities[node_ids]):
                    ax_net.text(0.94, y, f"{digit}: {prob:.2f}", fontsize=8, va="center")

        fig.suptitle(
            f"MNIST Forward Pass (stage {stage}/{num_layers - 1}) | "
            f"pred={prediction} prob={probabilities[prediction]:.3f}",
            fontsize=13,
            fontweight="bold",
        )

    frame_count = num_layers
    animation = FuncAnimation(fig, draw_stage, frames=frame_count, interval=1000, repeat=True)
    animation.save(gif_path, writer=PillowWriter(fps=args.fps))

    draw_stage(frame_count - 1)
    fig.savefig(final_image_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved input image:   {input_image_path}")
    print(f"Saved animated gif: {gif_path}")
    print(f"Saved final frame:  {final_image_path}")
    print(f"True label: {target} | Predicted: {prediction}")


if __name__ == "__main__":
    render_visualization(parse_args())
