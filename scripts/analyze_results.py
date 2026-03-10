from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mnist_hybrid.analysis.analysis import (
    plot_help_harm_hist,
    plot_memory_state,
    plot_retrieval_distance_vs_benefit,
    plot_training_curves,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate analysis plots from experiment outputs")
    parser.add_argument("--experiment-dir", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp_dir = Path(args.experiment_dir)
    analysis_dir = exp_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    epoch_logs = exp_dir / "epoch_logs.csv"
    details = exp_dir / "details.pt"
    memory_state = analysis_dir / "memory_state.pt"

    generated = []

    if epoch_logs.exists():
        generated.append(plot_training_curves(epoch_logs, analysis_dir))
    if details.exists():
        generated.append(plot_help_harm_hist(details, analysis_dir))
        try:
            generated.append(plot_retrieval_distance_vs_benefit(details, analysis_dir))
        except Exception:
            pass
    if memory_state.exists():
        generated.extend(plot_memory_state(memory_state, analysis_dir))

    print("Generated files:")
    for file_path in generated:
        print(f"- {file_path}")


if __name__ == "__main__":
    main()
