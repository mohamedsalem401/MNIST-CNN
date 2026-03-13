from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "growth_analysis"
DOCS_DIR = ROOT / "docs"


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)


def to_float(x) -> float:
    if x is None:
        return float("nan")
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def fmt(v: float, digits: int = 4) -> str:
    if v is None:
        return "NA"
    if isinstance(v, float) and math.isnan(v):
        return "NA"
    return f"{float(v):.{digits}f}"


def ci95(values: List[float]) -> Tuple[float, float]:
    arr = np.array([v for v in values if not np.isnan(v)], dtype=np.float64)
    n = arr.size
    if n <= 1:
        return float("nan"), float("nan")
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1))
    se = sd / math.sqrt(n)
    tcrit = float(stats.t.ppf(0.975, df=n - 1))
    return mean - tcrit * se, mean + tcrit * se


def mean_std(values: List[float]) -> Tuple[float, float]:
    arr = np.array([v for v in values if not np.isnan(v)], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def md_table(headers: List[str], rows: List[List[str]]) -> str:
    line1 = "| " + " | ".join(headers) + " |"
    line2 = "|" + "|".join(["---" for _ in headers]) + "|"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([line1, line2, *body])


@dataclass
class RunRecord:
    stage: str
    seed: int
    run_name: str
    status: str
    method: str
    output_dir: str
    metrics_path: str
    val_accuracy: float
    test_accuracy: float
    val_loss: float
    test_loss: float
    val_ece: float
    test_ece: float
    test_helped_fraction: float
    test_harmed_fraction: float
    test_strong_help_fraction: float
    test_strong_harm_fraction: float
    test_loss_delta_mean: float
    test_loss_delta_std: float
    test_with_memory_accuracy: float
    test_without_memory_accuracy: float
    test_throughput_samples_per_sec: float
    test_with_memory_throughput: float
    test_without_memory_throughput: float
    throughput_overhead_ratio: float
    memory_footprint_mb: float
    memory_size: float
    retrieval_purity_mean: float
    retrieval_distance_mean: float
    retrieval_top1_agreement_mean: float
    intervention_magnitude_mean: float
    gate_value_mean: float
    model_parameter_count: float
    model_current_depth: float
    model_current_width: float
    growth_event_count: float
    growth_val_delta_mean: float
    training_seconds_total: float
    memory_resets_due_to_growth: float


def parse_summary_entry(stage: str, seed: int, entry: Dict[str, object]) -> RunRecord:
    name = str(entry.get("name", ""))
    status = str(entry.get("status", "success"))
    output_dir = Path(str(entry.get("output_dir", "")))

    if status != "success" or not output_dir:
        return RunRecord(
            stage=stage,
            seed=seed,
            run_name=name,
            status="failed",
            method="unknown",
            output_dir=str(output_dir),
            metrics_path="",
            val_accuracy=float("nan"),
            test_accuracy=float("nan"),
            val_loss=float("nan"),
            test_loss=float("nan"),
            val_ece=float("nan"),
            test_ece=float("nan"),
            test_helped_fraction=float("nan"),
            test_harmed_fraction=float("nan"),
            test_strong_help_fraction=float("nan"),
            test_strong_harm_fraction=float("nan"),
            test_loss_delta_mean=float("nan"),
            test_loss_delta_std=float("nan"),
            test_with_memory_accuracy=float("nan"),
            test_without_memory_accuracy=float("nan"),
            test_throughput_samples_per_sec=float("nan"),
            test_with_memory_throughput=float("nan"),
            test_without_memory_throughput=float("nan"),
            throughput_overhead_ratio=float("nan"),
            memory_footprint_mb=float("nan"),
            memory_size=float("nan"),
            retrieval_purity_mean=float("nan"),
            retrieval_distance_mean=float("nan"),
            retrieval_top1_agreement_mean=float("nan"),
            intervention_magnitude_mean=float("nan"),
            gate_value_mean=float("nan"),
            model_parameter_count=float("nan"),
            model_current_depth=float("nan"),
            model_current_width=float("nan"),
            growth_event_count=float("nan"),
            growth_val_delta_mean=float("nan"),
            training_seconds_total=float("nan"),
            memory_resets_due_to_growth=float("nan"),
        )

    metrics_path = output_dir / "metrics.json"
    if not metrics_path.exists():
        return RunRecord(
            stage=stage,
            seed=seed,
            run_name=name,
            status="failed",
            method="unknown",
            output_dir=str(output_dir),
            metrics_path=str(metrics_path),
            val_accuracy=float("nan"),
            test_accuracy=float("nan"),
            val_loss=float("nan"),
            test_loss=float("nan"),
            val_ece=float("nan"),
            test_ece=float("nan"),
            test_helped_fraction=float("nan"),
            test_harmed_fraction=float("nan"),
            test_strong_help_fraction=float("nan"),
            test_strong_harm_fraction=float("nan"),
            test_loss_delta_mean=float("nan"),
            test_loss_delta_std=float("nan"),
            test_with_memory_accuracy=float("nan"),
            test_without_memory_accuracy=float("nan"),
            test_throughput_samples_per_sec=float("nan"),
            test_with_memory_throughput=float("nan"),
            test_without_memory_throughput=float("nan"),
            throughput_overhead_ratio=float("nan"),
            memory_footprint_mb=float("nan"),
            memory_size=float("nan"),
            retrieval_purity_mean=float("nan"),
            retrieval_distance_mean=float("nan"),
            retrieval_top1_agreement_mean=float("nan"),
            intervention_magnitude_mean=float("nan"),
            gate_value_mean=float("nan"),
            model_parameter_count=float("nan"),
            model_current_depth=float("nan"),
            model_current_width=float("nan"),
            growth_event_count=float("nan"),
            growth_val_delta_mean=float("nan"),
            training_seconds_total=float("nan"),
            memory_resets_due_to_growth=float("nan"),
        )

    payload = load_json(metrics_path)
    config = payload.get("config", {})
    final = payload.get("final_metrics", {})
    toggle = payload.get("toggle_eval", {})
    growth = payload.get("growth", {})

    growth_deltas = []
    for item in growth.get("performance_around_events", []):
        delta = to_float(item.get("val_accuracy_delta"))
        if not math.isnan(delta):
            growth_deltas.append(delta)
    growth_delta_mean = float(np.mean(growth_deltas)) if growth_deltas else float("nan")

    test_with_thr = to_float(toggle.get("test_with_memory_throughput_samples_per_sec"))
    test_without_thr = to_float(toggle.get("test_without_memory_throughput_samples_per_sec"))
    overhead = float("nan")
    if not np.isnan(test_with_thr) and not np.isnan(test_without_thr) and test_without_thr > 0:
        overhead = test_with_thr / test_without_thr

    return RunRecord(
        stage=stage,
        seed=seed,
        run_name=name,
        status="success",
        method=str(config.get("method", "unknown")),
        output_dir=str(output_dir),
        metrics_path=str(metrics_path),
        val_accuracy=to_float(final.get("val_accuracy")),
        test_accuracy=to_float(final.get("test_accuracy")),
        val_loss=to_float(final.get("val_loss")),
        test_loss=to_float(final.get("test_loss")),
        val_ece=to_float(final.get("val_ece")),
        test_ece=to_float(final.get("test_ece")),
        test_helped_fraction=to_float(toggle.get("test_helped_fraction")),
        test_harmed_fraction=to_float(toggle.get("test_harmed_fraction")),
        test_strong_help_fraction=to_float(toggle.get("test_strong_help_fraction")),
        test_strong_harm_fraction=to_float(toggle.get("test_strong_harm_fraction")),
        test_loss_delta_mean=to_float(toggle.get("test_loss_delta_mean")),
        test_loss_delta_std=to_float(toggle.get("test_loss_delta_std")),
        test_with_memory_accuracy=to_float(toggle.get("test_with_memory_accuracy")),
        test_without_memory_accuracy=to_float(toggle.get("test_without_memory_accuracy")),
        test_throughput_samples_per_sec=to_float(final.get("test_throughput_samples_per_sec")),
        test_with_memory_throughput=test_with_thr,
        test_without_memory_throughput=test_without_thr,
        throughput_overhead_ratio=to_float(overhead),
        memory_footprint_mb=to_float(final.get("test_memory_footprint_mb")),
        memory_size=to_float(final.get("test_memory_size")),
        retrieval_purity_mean=to_float(final.get("test_retrieval_purity_mean")),
        retrieval_distance_mean=to_float(final.get("test_retrieval_distance_mean")),
        retrieval_top1_agreement_mean=to_float(final.get("test_retrieval_top1_agreement_mean")),
        intervention_magnitude_mean=to_float(final.get("test_intervention_magnitude_mean")),
        gate_value_mean=to_float(final.get("test_gate_value_mean")),
        model_parameter_count=to_float(final.get("model_parameter_count")),
        model_current_depth=to_float(final.get("model_current_depth")),
        model_current_width=to_float(final.get("model_current_width")),
        growth_event_count=to_float(final.get("growth_event_count")),
        growth_val_delta_mean=to_float(growth_delta_mean),
        training_seconds_total=to_float(final.get("training_seconds_total")),
        memory_resets_due_to_growth=to_float(final.get("memory_resets_due_to_growth")),
    )


def collect_records() -> List[RunRecord]:
    records: List[RunRecord] = []

    stage_a_summary_path = RESULTS_DIR / "stage_a_matrix" / "matrix_summary.json"
    stage_a_summary = load_json(stage_a_summary_path)
    for item in stage_a_summary.get("matrix", []):
        records.append(parse_summary_entry("stage_a", 11, item))

    for seed in [11, 22, 33]:
        path = RESULTS_DIR / "stage_b_matrix" / f"seed{seed}" / "matrix_summary.json"
        if not path.exists():
            continue
        summary = load_json(path)
        for item in summary.get("matrix", []):
            records.append(parse_summary_entry("stage_b", seed, item))

    return records


def write_consolidated(records: List[RunRecord]) -> None:
    fields = list(RunRecord.__dataclass_fields__.keys())
    out_path = RESULTS_DIR / "consolidated_metrics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({k: getattr(record, k) for k in fields})


def group_stage(records: List[RunRecord], stage: str) -> Dict[str, List[RunRecord]]:
    grouped: Dict[str, List[RunRecord]] = {}
    for r in records:
        if r.stage != stage or r.status != "success":
            continue
        grouped.setdefault(r.run_name, []).append(r)
    return grouped


def paired_compare(
    records: List[RunRecord],
    run_a: str,
    run_b: str,
    metric: str,
    question: str,
) -> Dict[str, object]:
    a = [r for r in records if r.stage == "stage_b" and r.status == "success" and r.run_name == run_a]
    b = [r for r in records if r.stage == "stage_b" and r.status == "success" and r.run_name == run_b]
    a_by_seed = {r.seed: r for r in a}
    b_by_seed = {r.seed: r for r in b}
    seeds = sorted(set(a_by_seed.keys()) & set(b_by_seed.keys()))

    vals_a: List[float] = []
    vals_b: List[float] = []
    for seed in seeds:
        va = to_float(getattr(a_by_seed[seed], metric))
        vb = to_float(getattr(b_by_seed[seed], metric))
        if np.isnan(va) or np.isnan(vb):
            continue
        vals_a.append(va)
        vals_b.append(vb)

    if not vals_a:
        return {
            "question": question,
            "metric": metric,
            "config_a": run_a,
            "config_b": run_b,
            "n": 0,
            "mean_a": float("nan"),
            "std_a": float("nan"),
            "mean_b": float("nan"),
            "std_b": float("nan"),
            "mean_diff_a_minus_b": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "p_value": float("nan"),
            "effect_size_d": float("nan"),
            "p_value_adjusted": float("nan"),
        }

    arr_a = np.array(vals_a)
    arr_b = np.array(vals_b)
    diff = arr_a - arr_b

    mean_a, std_a = mean_std(vals_a)
    mean_b, std_b = mean_std(vals_b)
    mean_diff = float(diff.mean())

    if len(diff) > 1:
        sd_diff = float(diff.std(ddof=1))
        se_diff = sd_diff / math.sqrt(len(diff))
        tcrit = float(stats.t.ppf(0.975, df=len(diff) - 1))
        ci_low = mean_diff - tcrit * se_diff
        ci_high = mean_diff + tcrit * se_diff
        t_res = stats.ttest_rel(arr_a, arr_b)
        p_value = float(t_res.pvalue)
        effect_size = mean_diff / sd_diff if sd_diff > 1e-12 else float("nan")
    else:
        ci_low = float("nan")
        ci_high = float("nan")
        p_value = float("nan")
        effect_size = float("nan")

    return {
        "question": question,
        "metric": metric,
        "config_a": run_a,
        "config_b": run_b,
        "n": int(len(vals_a)),
        "mean_a": mean_a,
        "std_a": std_a,
        "mean_b": mean_b,
        "std_b": std_b,
        "mean_diff_a_minus_b": mean_diff,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "p_value": p_value,
        "effect_size_d": effect_size,
        "p_value_adjusted": float("nan"),
    }


def adjust_pvalues(rows: List[Dict[str, object]], method: str) -> None:
    method = (method or "none").lower()
    valid_idx = [idx for idx, row in enumerate(rows) if not math.isnan(to_float(row.get("p_value")))]
    if not valid_idx:
        return

    pvals = np.array([to_float(rows[idx]["p_value"]) for idx in valid_idx], dtype=np.float64)

    if method in {"none", ""}:
        adjusted = pvals
    elif method == "bonferroni":
        adjusted = np.minimum(pvals * len(pvals), 1.0)
    elif method in {"fdr_bh", "bh", "benjamini-hochberg", "benjamini_hochberg"}:
        order = np.argsort(pvals)
        ranked = pvals[order]
        m = len(pvals)
        adjusted_ranked = np.empty_like(ranked)
        prev = 1.0
        for i in range(m - 1, -1, -1):
            rank = i + 1
            candidate = ranked[i] * m / rank
            prev = min(prev, candidate)
            adjusted_ranked[i] = min(prev, 1.0)
        adjusted = np.empty_like(adjusted_ranked)
        adjusted[order] = adjusted_ranked
    else:
        adjusted = pvals

    for local_idx, record_idx in enumerate(valid_idx):
        rows[record_idx]["p_value_adjusted"] = float(adjusted[local_idx])


def write_comparisons(rows: List[Dict[str, object]]) -> None:
    fields = [
        "question",
        "metric",
        "config_a",
        "config_b",
        "n",
        "mean_a",
        "std_a",
        "mean_b",
        "std_b",
        "mean_diff_a_minus_b",
        "ci95_low",
        "ci95_high",
        "p_value",
        "p_value_adjusted",
        "effect_size_d",
    ]
    path = RESULTS_DIR / "comparison_tests.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def stage_b_summary(records: List[RunRecord], run_names: List[str]) -> Dict[str, Dict[str, float]]:
    grouped = group_stage(records, "stage_b")
    out: Dict[str, Dict[str, float]] = {}
    for name in run_names:
        rows = grouped.get(name, [])
        if not rows:
            continue

        def agg(metric: str) -> Tuple[float, float, float, float, int]:
            vals = [to_float(getattr(r, metric)) for r in rows]
            vals = [v for v in vals if not np.isnan(v)]
            if not vals:
                return float("nan"), float("nan"), float("nan"), float("nan"), 0
            mean, std = mean_std(vals)
            lo, hi = ci95(vals)
            return mean, std, lo, hi, len(vals)

        test_mean, test_std, test_lo, test_hi, n = agg("test_accuracy")
        val_mean, _, _, _, _ = agg("val_accuracy")
        loss_mean, _, _, _, _ = agg("test_loss")
        ece_mean, _, _, _, _ = agg("test_ece")
        helped_mean, _, _, _, _ = agg("test_helped_fraction")
        harmed_mean, _, _, _, _ = agg("test_harmed_fraction")
        strong_help_mean, _, _, _, _ = agg("test_strong_help_fraction")
        strong_harm_mean, _, _, _, _ = agg("test_strong_harm_fraction")
        params_mean, _, _, _, _ = agg("model_parameter_count")
        growth_events_mean, _, _, _, _ = agg("growth_event_count")
        throughput_mean, _, _, _, _ = agg("throughput_overhead_ratio")
        memory_mb_mean, _, _, _, _ = agg("memory_footprint_mb")
        purity_mean, _, _, _, _ = agg("retrieval_purity_mean")
        agreement_mean, _, _, _, _ = agg("retrieval_top1_agreement_mean")
        gate_mean, _, _, _, _ = agg("gate_value_mean")

        out[name] = {
            "n": float(n),
            "val_accuracy_mean": val_mean,
            "test_accuracy_mean": test_mean,
            "test_accuracy_std": test_std,
            "test_accuracy_ci_low": test_lo,
            "test_accuracy_ci_high": test_hi,
            "test_loss_mean": loss_mean,
            "test_ece_mean": ece_mean,
            "helped_mean": helped_mean,
            "harmed_mean": harmed_mean,
            "strong_help_mean": strong_help_mean,
            "strong_harm_mean": strong_harm_mean,
            "params_mean": params_mean,
            "growth_events_mean": growth_events_mean,
            "throughput_ratio_mean": throughput_mean,
            "memory_mb_mean": memory_mb_mean,
            "retrieval_purity_mean": purity_mean,
            "retrieval_top1_agreement_mean": agreement_mean,
            "gate_value_mean": gate_mean,
        }
    return out


def mean_metric(records: List[RunRecord], run_name: str, metric: str) -> float:
    vals = [to_float(getattr(r, metric)) for r in records if r.stage == "stage_b" and r.status == "success" and r.run_name == run_name]
    vals = [v for v in vals if not np.isnan(v)]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def decide_status(comparisons: List[Dict[str, object]]) -> str:
    lookup = {row["question"]: row for row in comparisons}

    def is_positive(question: str) -> bool:
        row = lookup.get(question)
        if not row:
            return False
        diff = to_float(row.get("mean_diff_a_minus_b"))
        lo = to_float(row.get("ci95_low"))
        return (not np.isnan(diff)) and (not np.isnan(lo)) and diff > 0 and lo > 0

    beat_random = is_positive("growing_mem_vs_random_control")
    beat_inactive = is_positive("growing_mem_vs_inactive_control")
    beat_nn_large = is_positive("growing_mem_vs_prior_nn_large")
    beat_embedding = is_positive("growing_mem_vs_prior_embedding_knn")

    if beat_random and beat_inactive and beat_nn_large and beat_embedding:
        return "promising"
    if (not beat_random) or (not beat_inactive):
        return "currently unsupported"
    return "unresolved"


def build_report(
    records: List[RunRecord],
    comparisons: List[Dict[str, object]],
    promoted: List[str],
    correction_method: str,
) -> None:
    stage_a_success = sum(1 for r in records if r.stage == "stage_a" and r.status == "success")
    stage_a_fail = sum(1 for r in records if r.stage == "stage_a" and r.status != "success")
    stage_b_success = sum(1 for r in records if r.stage == "stage_b" and r.status == "success")
    stage_b_fail = sum(1 for r in records if r.stage == "stage_b" and r.status != "success")

    stage_b_stats = stage_b_summary(records, promoted)

    table_rows = []
    for run in promoted:
        stats_row = stage_b_stats.get(run)
        if not stats_row:
            continue
        table_rows.append(
            [
                run,
                fmt(stats_row["n"], 0),
                fmt(stats_row["val_accuracy_mean"]),
                f"{fmt(stats_row['test_accuracy_mean'])} +/- {fmt(stats_row['test_accuracy_std'])}",
                f"[{fmt(stats_row['test_accuracy_ci_low'])}, {fmt(stats_row['test_accuracy_ci_high'])}]",
                fmt(stats_row["params_mean"], 0),
                fmt(stats_row["growth_events_mean"], 2),
                fmt(stats_row["helped_mean"]),
                fmt(stats_row["harmed_mean"]),
                fmt(stats_row["strong_help_mean"]),
                fmt(stats_row["strong_harm_mean"]),
                fmt(stats_row["gate_value_mean"]),
                fmt(stats_row["retrieval_top1_agreement_mean"]),
                fmt(stats_row["throughput_ratio_mean"]),
                fmt(stats_row["memory_mb_mean"]),
            ]
        )

    comp_rows = []
    for row in comparisons:
        comp_rows.append(
            [
                str(row["question"]),
                str(row["config_a"]),
                str(row["config_b"]),
                fmt(to_float(row["n"]), 0),
                fmt(to_float(row["mean_diff_a_minus_b"])),
                f"[{fmt(to_float(row['ci95_low']))}, {fmt(to_float(row['ci95_high']))}]",
                fmt(to_float(row["p_value"]), 4),
                fmt(to_float(row["p_value_adjusted"]), 4),
            ]
        )

    fixed_no_mem = next((n for n in promoted if n in {"g0_fixed_small_no_memory", "g2_fixed_large_no_memory_capacity_matched"}), None)
    fixed_mem = next((n for n in promoted if n in {"g1_fixed_small_with_memory", "g3_fixed_large_with_memory_capacity_matched"}), None)

    status = decide_status(comparisons)

    h1_line = "insufficient promoted comparator"
    if fixed_no_mem:
        mean_g4 = mean_metric(records, "g4_growing_no_memory", "test_accuracy")
        mean_fx = mean_metric(records, fixed_no_mem, "test_accuracy")
        if not np.isnan(mean_g4) and not np.isnan(mean_fx):
            diff = mean_g4 - mean_fx
            h1_line = f"g4 - {fixed_no_mem} = {diff:.4f}"

    h2_line = "insufficient promoted comparator"
    if fixed_mem:
        gain_growing = mean_metric(records, "g5_growing_with_memory", "test_accuracy") - mean_metric(records, "g4_growing_no_memory", "test_accuracy")
        gain_fixed = mean_metric(records, fixed_mem, "test_accuracy") - mean_metric(records, fixed_no_mem, "test_accuracy") if fixed_no_mem else float("nan")
        if not np.isnan(gain_growing) and not np.isnan(gain_fixed):
            h2_line = f"memory_gain_growing={gain_growing:.4f}, memory_gain_fixed={gain_fixed:.4f}"

    report = []
    report.append("# Scientific Report: Growing NN + Hidden-State Memory (MNIST)")
    report.append("")
    report.append("## Motivation")
    report.append("This stage evaluates whether progressive network growth makes hidden-state retrieval interventions more causally useful than fixed-size alternatives.")
    report.append("")
    report.append("## Pre-registered hypotheses")
    report.append("- H1: Growing NN without memory can match/exceed fixed NN at similar effective capacity.")
    report.append("- H2: Growing NN with memory benefits more from retrieval than fixed NN with memory.")
    report.append("- H3: Memory usefulness depends on attachment layer in growing models.")
    report.append("- H4: Growth schedule and memory schedule interact; some combinations help, others add noise.")
    report.append("- H5: If growth improves memory usefulness, active memory should separate from random/inactive controls.")
    report.append("- H6: If not separated from controls, growth+memory is not yet a convincing causal improvement.")
    report.append("")
    report.append("## Experimental protocol")
    report.append("- Stage A screening: single-seed matrix over required groups + optional stress variants.")
    report.append("- Stage B confirmation: promoted subset with seeds 11/22/33.")
    report.append("- Promotion criterion: validation accuracy only (see `results/growth_analysis/stage_policy.json`).")
    report.append(f"- Multiple-comparison correction in pairwise claims: `{correction_method}`.")
    report.append("")
    report.append("## Run status")
    report.append(f"- Stage A successful runs: {stage_a_success}, failures: {stage_a_fail}")
    report.append(f"- Stage B successful runs: {stage_b_success}, failures: {stage_b_fail}")
    report.append("")
    report.append("## Stage B main table")
    report.append(
        md_table(
            [
                "Run",
                "n",
                "Val acc mean",
                "Test acc mean +/- std",
                "Test acc 95% CI",
                "Params mean",
                "Growth events",
                "Helped",
                "Harmed",
                "Strong help",
                "Strong harm",
                "Gate mean",
                "Retrieval top1",
                "Thr ratio",
                "Memory MB",
            ],
            table_rows,
        )
    )
    report.append("")
    report.append("## Critical causal/statistical checks")
    report.append(
        md_table(
            [
                "Question",
                "A",
                "B",
                "n",
                "Diff A-B",
                "95% CI",
                "p",
                "p adj",
            ],
            comp_rows,
        )
    )
    report.append("")
    report.append("## Hypothesis-by-hypothesis readout")
    report.append(f"- H1: {h1_line}")
    report.append(f"- H2: {h2_line}")
    report.append("- H3: Layer-placement evidence only from promoted optional variant if selected; otherwise unresolved.")
    report.append("- H4: Growth/memory interaction evidence is limited to compact schedule sweep in this compute tier.")
    report.append("- H5/H6: judged primarily by `g5` vs `g6` and `g7` pairwise tests.")
    report.append("")
    report.append("## Evidence-backed conclusions")
    report.append("- Claims are limited to Stage B paired comparisons and their confidence intervals.")
    report.append("- If CIs overlap zero, no decisive separation is claimed.")
    report.append("- Growth effects are interpreted jointly with parameter count and control runs.")
    report.append("")
    report.append("## Mechanistic diagnostics")
    report.append("- Reported diagnostics include helped/harmed and strong-help/strong-harm fractions, intervention magnitude, gate statistics, retrieval top-1 agreement, throughput ratio, training time, memory footprint, parameter count, growth events, and pre/post-growth validation deltas.")
    report.append("")
    report.append("## Limitations")
    report.append("- CPU-only budget and staged promotion limit matrix-wide multi-seed certainty.")
    report.append("- Some optional mechanism variants may remain single-seed if not promoted.")
    report.append("- Conclusions are benchmark-local to MNIST; no external validity claim is made.")
    report.append("")
    report.append("## Final recommendation")
    report.append(f"The growing-memory direction is **{status}** under the current evidence.")
    report.append("- Decision rule: no success claim unless growing+memory beats strong baselines and memory controls with uncertainty excluding zero.")
    report.append("- If controls are not separated, prioritize intervention/target redesign before scaling compute.")
    report.append("")
    report.append("## Artifact map")
    report.append("- Consolidated metrics: `results/growth_analysis/consolidated_metrics.csv`")
    report.append("- Pairwise tests: `results/growth_analysis/comparison_tests.csv`")
    report.append("- Promotion policy: `results/growth_analysis/stage_policy.json`")
    report.append("- Stage B promoted runs: `results/growth_analysis/stage_b_promoted_runs.json`")

    out_path = DOCS_DIR / "growing_scientific_report.md"
    out_path.write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    records = collect_records()
    write_consolidated(records)

    promoted_path = RESULTS_DIR / "stage_b_promoted_runs.json"
    promoted = []
    if promoted_path.exists():
        promoted = [str(v) for v in load_json(promoted_path).get("promoted_runs", [])]

    if not promoted:
        promoted = sorted({r.run_name for r in records if r.stage == "stage_b" and r.status == "success"})

    correction_method = "none"
    first_stage_b = next((r for r in records if r.stage == "stage_b" and r.status == "success"), None)
    if first_stage_b and first_stage_b.metrics_path:
        payload = load_json(Path(first_stage_b.metrics_path))
        correction_method = str(
            payload.get("config", {})
            .get("evaluation", {})
            .get("multiple_comparison_correction", "none")
        )

    fixed_no_mem = next((n for n in promoted if n in {"g0_fixed_small_no_memory", "g2_fixed_large_no_memory_capacity_matched"}), "")
    fixed_mem = next((n for n in promoted if n in {"g1_fixed_small_with_memory", "g3_fixed_large_with_memory_capacity_matched"}), "")

    comparisons: List[Dict[str, object]] = []
    base_pairs = [
        ("growing_mem_vs_growing_no_mem", "g5_growing_with_memory", "g4_growing_no_memory"),
        ("growing_mem_vs_random_control", "g5_growing_with_memory", "g6_growing_random_memory_control"),
        ("growing_mem_vs_inactive_control", "g5_growing_with_memory", "g7_growing_inactive_memory_control"),
        ("growing_mem_vs_prior_nn_large", "g5_growing_with_memory", "g8_prior_nn_large"),
        ("growing_mem_vs_prior_embedding_knn", "g5_growing_with_memory", "g9_prior_embedding_knn"),
    ]
    if fixed_mem:
        base_pairs.append(("growing_mem_vs_best_fixed_mem", "g5_growing_with_memory", fixed_mem))
    if fixed_no_mem:
        base_pairs.append(("growing_no_mem_vs_best_fixed_no_mem", "g4_growing_no_memory", fixed_no_mem))

    for question, run_a, run_b in base_pairs:
        comparisons.append(paired_compare(records, run_a, run_b, "test_accuracy", question))

    adjust_pvalues(comparisons, correction_method)
    write_comparisons(comparisons)

    build_report(records, comparisons, promoted, correction_method)

    save_json(
        {
            "num_records": len(records),
            "stage_a_success": sum(1 for r in records if r.stage == "stage_a" and r.status == "success"),
            "stage_b_success": sum(1 for r in records if r.stage == "stage_b" and r.status == "success"),
            "correction_method": correction_method,
        },
        RESULTS_DIR / "analysis_summary.json",
    )
    print("Wrote growth analysis artifacts.")


if __name__ == "__main__":
    main()
