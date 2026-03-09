from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Dict, List
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mnist_hybrid.config import load_config
from mnist_hybrid.training.trainer import ExperimentRunner
from mnist_hybrid.utils.common import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an experiment matrix from YAML")
    parser.add_argument("--base-config", type=str, required=True)
    parser.add_argument("--matrix", type=str, required=True)
    return parser.parse_args()


def flatten_overrides(overrides: Dict[str, object], prefix: str = "") -> List[str]:
    flat = []
    for key, value in overrides.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.extend(flatten_overrides(value, prefix=full_key))
        else:
            flat.append(f"{full_key}={value}")
    return flat


def main() -> None:
    args = parse_args()
    with open(args.matrix, "r", encoding="utf-8") as f:
        matrix_cfg = yaml.safe_load(f)

    runs = matrix_cfg.get("runs", [])
    if not runs:
        raise ValueError("Matrix file has no runs")

    summaries = []
    for run in runs:
        name = run["name"]
        overrides = flatten_overrides(run.get("overrides", {}))
        cfg = load_config(args.base_config, overrides=overrides)
        cfg.logging.experiment_name = name

        runner = ExperimentRunner(copy.deepcopy(cfg))
        output = runner.run()

        summaries.append(
            {
                "name": name,
                "overrides": overrides,
                "output_dir": output["output_dir"],
                "final_metrics": output["results"].get("final_metrics", {}),
                "toggle_eval": output["results"].get("toggle_eval", {}),
            }
        )

    out_root = ensure_dir(Path(matrix_cfg.get("output_root", "results/matrix")))
    save_json({"matrix": summaries}, out_root / "matrix_summary.json")
    print(f"Matrix complete. Summary: {out_root / 'matrix_summary.json'}")


if __name__ == "__main__":
    main()
