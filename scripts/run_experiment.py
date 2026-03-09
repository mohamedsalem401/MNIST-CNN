from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Dict, List
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mnist_hybrid.config import ExperimentConfig, load_config
from mnist_hybrid.training.trainer import ExperimentRunner
from mnist_hybrid.utils.common import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MNIST hybrid NN + KNN-memory experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=[],
        help="Dotlist overrides: key=value key2=value2",
    )
    return parser.parse_args()


def aggregate_metrics(runs: List[Dict[str, object]]) -> Dict[str, object]:
    metric_keys = set()
    for run in runs:
        metric_keys.update(run["results"]["final_metrics"].keys())
        metric_keys.update(run["results"].get("toggle_eval", {}).keys())

    agg: Dict[str, object] = {}
    for key in sorted(metric_keys):
        vals = []
        for run in runs:
            metrics = {}
            metrics.update(run["results"].get("final_metrics", {}))
            metrics.update(run["results"].get("toggle_eval", {}))
            if key in metrics:
                value = metrics[key]
                if isinstance(value, (int, float)):
                    vals.append(float(value))
        if vals:
            agg[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "n": len(vals),
            }
    return agg


def main() -> None:
    args = parse_args()
    config = load_config(args.config, overrides=args.override)

    seeds = config.seeds if config.seeds else [config.seed]
    runs = []

    for seed in seeds:
        run_cfg = copy.deepcopy(config)
        run_cfg.seed = int(seed)
        run_cfg.logging.experiment_name = f"{config.logging.experiment_name}_seed{seed}"

        runner = ExperimentRunner(run_cfg)
        output = runner.run()
        runs.append(output)

    aggregate = {
        "base_config": args.config,
        "overrides": args.override,
        "num_runs": len(runs),
        "aggregated_metrics": aggregate_metrics(runs),
        "run_output_dirs": [run["output_dir"] for run in runs],
    }

    base_out = ensure_dir(Path(config.logging.output_root) / config.logging.experiment_name)
    save_json(aggregate, base_out / "aggregate_metrics.json")

    print(f"Completed {len(runs)} run(s). Aggregate: {base_out / 'aggregate_metrics.json'}")


if __name__ == "__main__":
    main()
