from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List, Optional
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Stage B promotion configs for growth study")
    parser.add_argument("--stage-a-summary", type=str, default="results/growth_analysis/stage_a_matrix/matrix_summary.json")
    parser.add_argument("--stage-a-matrix", type=str, default="configs/growth_matrix_stage_a.yaml")
    parser.add_argument("--stage-a-base", type=str, default="configs/growth_base_stage_a.yaml")
    parser.add_argument("--output-root", type=str, default="results/growth_analysis")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--stage-b-epochs", type=int, default=5)
    parser.add_argument("--stage-b-train-subset", type=int, default=12000)
    parser.add_argument("--stage-b-val-subset", type=int, default=3000)
    parser.add_argument("--stage-b-test-subset", type=int, default=10000)
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)


def load_yaml(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def best_by_val(candidates: List[str], val_map: Dict[str, float]) -> Optional[str]:
    available = [(name, val_map.get(name, float("nan"))) for name in candidates]
    available = [item for item in available if item[1] == item[1]]  # filter NaN
    if not available:
        return None
    available.sort(key=lambda x: x[1], reverse=True)
    return available[0][0]


def main() -> None:
    args = parse_args()

    stage_a_summary = load_json(Path(args.stage_a_summary))
    stage_a_matrix = load_yaml(Path(args.stage_a_matrix))
    stage_a_base = load_yaml(Path(args.stage_a_base))

    runs = stage_a_summary.get("matrix", [])
    val_map: Dict[str, float] = {}
    failures: List[Dict[str, str]] = []

    for run in runs:
        name = str(run.get("name", ""))
        status = str(run.get("status", "success"))
        if status != "success":
            failures.append({"name": name, "error": str(run.get("error", "unknown"))})
            continue
        val_acc = run.get("final_metrics", {}).get("val_accuracy", float("nan"))
        try:
            val_map[name] = float(val_acc)
        except (TypeError, ValueError):
            continue

    anchors = [
        "g4_growing_no_memory",
        "g5_growing_with_memory",
        "g6_growing_random_memory_control",
        "g7_growing_inactive_memory_control",
        "g8_prior_nn_large",
        "g9_prior_embedding_knn",
    ]
    fixed_no_memory_best = best_by_val(
        ["g0_fixed_small_no_memory", "g2_fixed_large_no_memory_capacity_matched"],
        val_map,
    )
    fixed_memory_best = best_by_val(
        ["g1_fixed_small_with_memory", "g3_fixed_large_with_memory_capacity_matched"],
        val_map,
    )
    optional_best = best_by_val(
        [
            "g10_growing_with_memory_overwrite",
            "g11_growing_with_memory_layer_early",
            "g12_growing_with_memory_uncertainty_gating",
        ],
        val_map,
    )

    promoted_ordered: List[str] = []
    for name in [*anchors, fixed_no_memory_best, fixed_memory_best, optional_best]:
        if not name:
            continue
        if name not in promoted_ordered:
            promoted_ordered.append(name)

    run_defs = {item["name"]: item for item in stage_a_matrix.get("runs", [])}
    promoted_runs = [copy.deepcopy(run_defs[name]) for name in promoted_ordered if name in run_defs]

    output_root = Path(args.output_root)
    policy = {
        "selection_principles": [
            "Promotion uses Stage A validation accuracy only.",
            "Keep anchors for growing model, controls, and prior strong baselines.",
            "Promote top fixed-no-memory and fixed-with-memory comparators.",
            "Promote top optional stress variant if available.",
        ],
        "stage_a_primary_metric": "val_accuracy",
        "anchors": anchors,
        "fixed_no_memory_candidates": ["g0_fixed_small_no_memory", "g2_fixed_large_no_memory_capacity_matched"],
        "fixed_memory_candidates": ["g1_fixed_small_with_memory", "g3_fixed_large_with_memory_capacity_matched"],
        "optional_candidates": [
            "g10_growing_with_memory_overwrite",
            "g11_growing_with_memory_layer_early",
            "g12_growing_with_memory_uncertainty_gating",
        ],
        "selected": promoted_ordered,
        "selected_val_accuracy": {k: val_map.get(k, float("nan")) for k in promoted_ordered},
        "failed_stage_a_runs": failures,
    }

    save_json(policy, output_root / "stage_policy.json")
    save_json({"promoted_runs": promoted_ordered}, output_root / "stage_b_promoted_runs.json")

    configs_root = output_root / "configs"
    for seed in args.seeds:
        base_cfg = copy.deepcopy(stage_a_base)
        base_cfg["seed"] = int(seed)
        base_cfg["seeds"] = [int(seed)]
        base_cfg.setdefault("optim", {})["epochs"] = int(args.stage_b_epochs)

        data_cfg = base_cfg.setdefault("data", {})
        data_cfg["train_subset_size"] = int(args.stage_b_train_subset)
        data_cfg["val_subset_size"] = int(args.stage_b_val_subset)
        data_cfg["test_subset_size"] = int(args.stage_b_test_subset)

        eval_cfg = base_cfg.setdefault("evaluation", {})
        eval_cfg["n_seeds"] = int(len(args.seeds))
        eval_cfg["compute_budget_tier"] = "stage_b_confirmation"

        logging_cfg = base_cfg.setdefault("logging", {})
        logging_cfg["output_root"] = f"results/growth_analysis/stage_b_runs/seed{seed}"
        logging_cfg["experiment_name"] = f"growth_stage_b_seed{seed}"

        base_path = configs_root / f"growth_base_stage_b_seed{seed}.yaml"
        save_yaml(base_cfg, base_path)

        matrix_payload = {
            "output_root": f"results/growth_analysis/stage_b_matrix/seed{seed}",
            "runs": promoted_runs,
        }
        matrix_path = configs_root / f"growth_matrix_stage_b_seed{seed}.yaml"
        save_yaml(matrix_payload, matrix_path)

    print(f"Prepared Stage B configs under: {configs_root}")
    print(f"Promoted runs ({len(promoted_ordered)}): {', '.join(promoted_ordered)}")


if __name__ == "__main__":
    main()
