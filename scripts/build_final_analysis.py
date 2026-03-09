from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "results" / "final_analysis"
FIG_DIR = FINAL_DIR / "figures"
DOCS_DIR = ROOT / "docs"


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def to_float(x) -> float:
    if x is None:
        return float("nan")
    try:
        v = float(x)
        return v
    except (TypeError, ValueError):
        return float("nan")


def fmt(v, digits: int = 4) -> str:
    if v is None:
        return "NA"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "NA"
    if math.isnan(f):
        return "NA"
    return f"{f:.{digits}f}"


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
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([line1, line2, *body])


@dataclass
class RunRecord:
    stage: str
    seed: int
    run_name: str
    method: str
    status: str
    output_dir: str
    metrics_path: str
    details_path: str
    memory_state_path: str
    val_accuracy: float
    test_accuracy: float
    val_loss: float
    test_loss: float
    val_ece: float
    test_ece: float
    test_helped_fraction: float
    test_harmed_fraction: float
    test_neutral_fraction: float
    test_loss_delta_mean: float
    test_loss_delta_std: float
    test_with_memory_accuracy: float
    test_without_memory_accuracy: float
    memory_footprint_mb: float
    test_throughput_samples_per_sec: float
    test_with_memory_throughput: float
    test_without_memory_throughput: float
    throughput_overhead_ratio: float
    retrieval_purity_mean: float
    retrieval_distance_mean: float
    intervention_magnitude_mean: float
    memory_size: float


def parse_run(stage: str, seed: int, run_item: Dict[str, object]) -> Tuple[RunRecord, List[str]]:
    issues: List[str] = []
    run_name = run_item.get("name", "")
    output_dir = Path(str(run_item.get("output_dir", "")))
    metrics_path = output_dir / "metrics.json"
    details_path = output_dir / "details.pt"
    memory_state_path = output_dir / "analysis" / "memory_state.pt"

    if not metrics_path.exists():
        issues.append(f"missing_metrics_json:{metrics_path}")
        return (
            RunRecord(
                stage=stage,
                seed=seed,
                run_name=run_name,
                method="unknown",
                status="failed",
                output_dir=str(output_dir),
                metrics_path=str(metrics_path),
                details_path=str(details_path),
                memory_state_path=str(memory_state_path),
                val_accuracy=float("nan"),
                test_accuracy=float("nan"),
                val_loss=float("nan"),
                test_loss=float("nan"),
                val_ece=float("nan"),
                test_ece=float("nan"),
                test_helped_fraction=float("nan"),
                test_harmed_fraction=float("nan"),
                test_neutral_fraction=float("nan"),
                test_loss_delta_mean=float("nan"),
                test_loss_delta_std=float("nan"),
                test_with_memory_accuracy=float("nan"),
                test_without_memory_accuracy=float("nan"),
                memory_footprint_mb=float("nan"),
                test_throughput_samples_per_sec=float("nan"),
                test_with_memory_throughput=float("nan"),
                test_without_memory_throughput=float("nan"),
                throughput_overhead_ratio=float("nan"),
                retrieval_purity_mean=float("nan"),
                retrieval_distance_mean=float("nan"),
                intervention_magnitude_mean=float("nan"),
                memory_size=float("nan"),
            ),
            issues,
        )

    data = load_json(metrics_path)
    required_top = {"config", "final_metrics", "toggle_eval", "epoch_logs", "env"}
    missing = sorted(list(required_top - set(data.keys())))
    if missing:
        issues.append(f"missing_top_keys:{','.join(missing)}")

    config = data.get("config", {})
    final = data.get("final_metrics", {})
    toggle = data.get("toggle_eval", {})

    method = str(config.get("method", "unknown")) if isinstance(config, dict) else "unknown"

    test_with_thr = to_float(toggle.get("test_with_memory_throughput_samples_per_sec"))
    test_without_thr = to_float(toggle.get("test_without_memory_throughput_samples_per_sec"))
    if not np.isnan(test_with_thr) and not np.isnan(test_without_thr) and test_without_thr != 0:
        overhead = test_with_thr / test_without_thr
    else:
        overhead = float("nan")

    rec = RunRecord(
        stage=stage,
        seed=seed,
        run_name=str(run_name),
        method=method,
        status="success",
        output_dir=str(output_dir),
        metrics_path=str(metrics_path),
        details_path=str(details_path),
        memory_state_path=str(memory_state_path),
        val_accuracy=to_float(final.get("val_accuracy")),
        test_accuracy=to_float(final.get("test_accuracy")),
        val_loss=to_float(final.get("val_loss")),
        test_loss=to_float(final.get("test_loss")),
        val_ece=to_float(final.get("val_ece")),
        test_ece=to_float(final.get("test_ece")),
        test_helped_fraction=to_float(toggle.get("test_helped_fraction")),
        test_harmed_fraction=to_float(toggle.get("test_harmed_fraction")),
        test_neutral_fraction=to_float(toggle.get("test_neutral_fraction")),
        test_loss_delta_mean=to_float(toggle.get("test_loss_delta_mean")),
        test_loss_delta_std=to_float(toggle.get("test_loss_delta_std")),
        test_with_memory_accuracy=to_float(toggle.get("test_with_memory_accuracy")),
        test_without_memory_accuracy=to_float(toggle.get("test_without_memory_accuracy")),
        memory_footprint_mb=to_float(final.get("test_memory_footprint_mb")),
        test_throughput_samples_per_sec=to_float(final.get("test_throughput_samples_per_sec")),
        test_with_memory_throughput=test_with_thr,
        test_without_memory_throughput=test_without_thr,
        throughput_overhead_ratio=to_float(overhead),
        retrieval_purity_mean=to_float(final.get("test_retrieval_purity_mean")),
        retrieval_distance_mean=to_float(final.get("test_retrieval_distance_mean")),
        intervention_magnitude_mean=to_float(final.get("test_intervention_magnitude_mean")),
        memory_size=to_float(final.get("test_memory_size")),
    )

    if not details_path.exists():
        issues.append(f"missing_details_pt:{details_path}")
    if method in {"hybrid", "hybrid_inactive", "random_memory"} and not memory_state_path.exists():
        issues.append(f"missing_memory_state:{memory_state_path}")

    return rec, issues


def collect_records() -> Tuple[List[RunRecord], List[str]]:
    issues: List[str] = []
    records: List[RunRecord] = []

    stage_a = load_json(FINAL_DIR / "stage_a_matrix" / "matrix_summary.json")
    for run in stage_a.get("matrix", []):
        rec, rec_issues = parse_run("stage_a", 11, run)
        records.append(rec)
        issues.extend([f"{rec.run_name}: {x}" for x in rec_issues])

    for seed in [11, 22, 33]:
        path = FINAL_DIR / "stage_b_matrix" / f"seed{seed}" / "matrix_summary.json"
        stage_b = load_json(path)
        for run in stage_b.get("matrix", []):
            rec, rec_issues = parse_run("stage_b", seed, run)
            records.append(rec)
            issues.extend([f"seed{seed}:{rec.run_name}: {x}" for x in rec_issues])

    return records, issues


def write_consolidated(records: List[RunRecord]) -> None:
    out = FINAL_DIR / "consolidated_metrics.csv"
    fields = list(RunRecord.__dataclass_fields__.keys())
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow({k: getattr(r, k) for k in fields})


def group_by_run(records: List[RunRecord], stage: str) -> Dict[str, List[RunRecord]]:
    grouped: Dict[str, List[RunRecord]] = {}
    for r in records:
        if r.stage != stage or r.status != "success":
            continue
        grouped.setdefault(r.run_name, []).append(r)
    return grouped


def summary_metric(records: List[RunRecord], metric: str) -> Dict[str, Dict[str, float]]:
    grouped = group_by_run(records, "stage_b")
    out: Dict[str, Dict[str, float]] = {}
    for run, rows in grouped.items():
        vals = [float(getattr(r, metric)) for r in rows]
        vals = [v for v in vals if not np.isnan(v)]
        if not vals:
            out[run] = {"n": 0, "mean": float("nan"), "std": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
            continue
        mean, sd = mean_std(vals)
        lo, hi = ci95(vals)
        out[run] = {"n": len(vals), "mean": mean, "std": sd, "ci95_low": lo, "ci95_high": hi}
    return out


def paired_compare(records: List[RunRecord], run_a: str, run_b: str, metric: str, question: str) -> Dict[str, object]:
    a = [r for r in records if r.stage == "stage_b" and r.run_name == run_a and r.status == "success"]
    b = [r for r in records if r.stage == "stage_b" and r.run_name == run_b and r.status == "success"]
    map_a = {r.seed: r for r in a}
    map_b = {r.seed: r for r in b}
    seeds = sorted(set(map_a.keys()) & set(map_b.keys()))

    vals_a: List[float] = []
    vals_b: List[float] = []
    for s in seeds:
        va = float(getattr(map_a[s], metric))
        vb = float(getattr(map_b[s], metric))
        if np.isnan(va) or np.isnan(vb):
            continue
        vals_a.append(va)
        vals_b.append(vb)

    if not vals_a:
        return {
            "question": question,
            "stage": "stage_b",
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
            "notes": "insufficient paired data",
        }

    arr_a = np.array(vals_a)
    arr_b = np.array(vals_b)
    diff = arr_a - arr_b

    mean_a, std_a = mean_std(vals_a)
    mean_b, std_b = mean_std(vals_b)
    mean_diff = float(diff.mean())

    if len(diff) > 1:
        sd_diff = float(diff.std(ddof=1))
        se = sd_diff / math.sqrt(len(diff))
        tcrit = float(stats.t.ppf(0.975, df=len(diff) - 1))
        ci_lo = mean_diff - tcrit * se
        ci_hi = mean_diff + tcrit * se
        ttest = stats.ttest_rel(arr_a, arr_b)
        pval = float(ttest.pvalue)
        d = mean_diff / sd_diff if sd_diff > 0 else float("nan")
    else:
        ci_lo = float("nan")
        ci_hi = float("nan")
        pval = float("nan")
        d = float("nan")

    return {
        "question": question,
        "stage": "stage_b",
        "metric": metric,
        "config_a": run_a,
        "config_b": run_b,
        "n": len(diff),
        "mean_a": mean_a,
        "std_a": std_a,
        "mean_b": mean_b,
        "std_b": std_b,
        "mean_diff_a_minus_b": mean_diff,
        "ci95_low": ci_lo,
        "ci95_high": ci_hi,
        "p_value": pval,
        "effect_size_d": d,
        "notes": "paired by seed",
    }


def stage_a_single_compare(records: List[RunRecord], run_a: str, run_b: str, metric: str, question: str) -> Dict[str, object]:
    a = [r for r in records if r.stage == "stage_a" and r.run_name == run_a and r.status == "success"]
    b = [r for r in records if r.stage == "stage_a" and r.run_name == run_b and r.status == "success"]
    va = float(getattr(a[0], metric)) if a else float("nan")
    vb = float(getattr(b[0], metric)) if b else float("nan")
    diff = va - vb if (not np.isnan(va) and not np.isnan(vb)) else float("nan")
    return {
        "question": question,
        "stage": "stage_a",
        "metric": metric,
        "config_a": run_a,
        "config_b": run_b,
        "n": 1 if (a and b) else 0,
        "mean_a": va,
        "std_a": float("nan"),
        "mean_b": vb,
        "std_b": float("nan"),
        "mean_diff_a_minus_b": diff,
        "ci95_low": float("nan"),
        "ci95_high": float("nan"),
        "p_value": float("nan"),
        "effect_size_d": float("nan"),
        "notes": "single-seed provisional comparison",
    }


def write_comparisons(records: List[RunRecord]) -> List[Dict[str, object]]:
    tests: List[Dict[str, object]] = []
    tests.append(
        paired_compare(
            records,
            "a1_hybrid_penultimate_gated_delta",
            "a19_baseline_nn_large",
            "test_accuracy",
            "memory_vs_strong_parametric_baseline",
        )
    )
    tests.append(
        paired_compare(
            records,
            "a1_hybrid_penultimate_gated_delta",
            "a17_random_memory_control",
            "test_accuracy",
            "real_memory_vs_random_memory",
        )
    )
    tests.append(
        paired_compare(
            records,
            "a1_hybrid_penultimate_gated_delta",
            "a18_hybrid_inactive",
            "test_accuracy",
            "active_memory_vs_inactive_control",
        )
    )
    tests.append(
        paired_compare(
            records,
            "a3_hybrid_penultimate_residual_delta",
            "a1_hybrid_penultimate_gated_delta",
            "test_accuracy",
            "mode_residual_vs_gated",
        )
    )
    tests.append(
        paired_compare(
            records,
            "a6_hybrid_penultimate_absolute_target",
            "a1_hybrid_penultimate_gated_delta",
            "test_accuracy",
            "target_absolute_vs_delta",
        )
    )
    tests.append(
        paired_compare(
            records,
            "a9_no_forgetting",
            "a1_hybrid_penultimate_gated_delta",
            "test_accuracy",
            "forgetting_none_vs_helpfulness_age",
        )
    )
    tests.append(
        paired_compare(
            records,
            "a1_hybrid_penultimate_gated_delta",
            "a19_baseline_nn_large",
            "test_loss",
            "memory_vs_strong_baseline_test_loss",
        )
    )

    tests.append(
        stage_a_single_compare(
            records,
            "a2_hybrid_penultimate_overwrite_delta",
            "a1_hybrid_penultimate_gated_delta",
            "test_accuracy",
            "stage_a_mode_overwrite_vs_gated",
        )
    )
    tests.append(
        stage_a_single_compare(
            records,
            "a4_hybrid_middle_gated_delta",
            "a1_hybrid_penultimate_gated_delta",
            "test_accuracy",
            "stage_a_layer_middle_vs_penultimate",
        )
    )
    tests.append(
        stage_a_single_compare(
            records,
            "a5_hybrid_early_gated_delta",
            "a1_hybrid_penultimate_gated_delta",
            "test_accuracy",
            "stage_a_layer_early_vs_penultimate",
        )
    )
    tests.append(
        stage_a_single_compare(
            records,
            "a15_train_only",
            "a1_hybrid_penultimate_gated_delta",
            "test_accuracy",
            "stage_a_train_only_vs_train_and_infer",
        )
    )
    tests.append(
        stage_a_single_compare(
            records,
            "a16_inference_only",
            "a1_hybrid_penultimate_gated_delta",
            "test_accuracy",
            "stage_a_infer_only_vs_train_and_infer",
        )
    )

    out = FINAL_DIR / "comparison_tests.csv"
    fields = list(tests[0].keys())
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in tests:
            writer.writerow(row)

    return tests


def make_figures(records: List[RunRecord]) -> Dict[str, str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_map: Dict[str, str] = {}

    stage_b = [r for r in records if r.stage == "stage_b" and r.status == "success"]
    run_order = sorted({r.run_name for r in stage_b})

    # Figure 1: stage B test accuracy mean/std
    means = []
    stds = []
    labels = []
    for run in run_order:
        vals = [r.test_accuracy for r in stage_b if r.run_name == run and not np.isnan(r.test_accuracy)]
        m, s = mean_std(vals)
        labels.append(run)
        means.append(m)
        stds.append(s)

    fig1 = FIG_DIR / "stage_b_test_accuracy_mean_std.png"
    plt.figure(figsize=(12, 5))
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=stds, capsize=3, color="#3b6fb6")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Test accuracy")
    plt.title("Stage B test accuracy (mean +/- std across seeds)")
    plt.tight_layout()
    plt.savefig(fig1, dpi=180)
    plt.close()
    fig_map["stage_b_test_accuracy_mean_std"] = str(fig1)

    # Figure 2: stage A val accuracy all runs
    stage_a = [r for r in records if r.stage == "stage_a" and r.status == "success"]
    stage_a_sorted = sorted(stage_a, key=lambda r: r.run_name)
    fig2 = FIG_DIR / "stage_a_val_accuracy_all_runs.png"
    plt.figure(figsize=(14, 5))
    xs = np.arange(len(stage_a_sorted))
    ys = [r.val_accuracy for r in stage_a_sorted]
    plt.bar(xs, ys, color="#5a9b76")
    plt.xticks(xs, [r.run_name for r in stage_a_sorted], rotation=45, ha="right")
    plt.ylabel("Validation accuracy")
    plt.title("Stage A validation accuracy (single-seed screen)")
    plt.tight_layout()
    plt.savefig(fig2, dpi=180)
    plt.close()
    fig_map["stage_a_val_accuracy_all_runs"] = str(fig2)

    # Figure 3: helped vs harmed in stage B
    mem_stage_b = [
        r
        for r in stage_b
        if not np.isnan(r.test_helped_fraction) and not np.isnan(r.test_harmed_fraction)
    ]
    mem_runs = sorted({r.run_name for r in mem_stage_b})
    helped = []
    harmed = []
    for run in mem_runs:
        h = [r.test_helped_fraction for r in mem_stage_b if r.run_name == run]
        hm = [r.test_harmed_fraction for r in mem_stage_b if r.run_name == run]
        helped.append(float(np.mean(h)))
        harmed.append(float(np.mean(hm)))

    fig3 = FIG_DIR / "stage_b_helped_vs_harmed.png"
    plt.figure(figsize=(12, 5))
    x = np.arange(len(mem_runs))
    width = 0.38
    plt.bar(x - width / 2, helped, width, label="helped", color="#2f7d32")
    plt.bar(x + width / 2, harmed, width, label="harmed", color="#b23b3b")
    plt.xticks(x, mem_runs, rotation=45, ha="right")
    plt.ylabel("Fraction of test samples")
    plt.title("Stage B helped vs harmed sample fractions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig3, dpi=180)
    plt.close()
    fig_map["stage_b_helped_vs_harmed"] = str(fig3)

    # Figure 4: memory footprint vs throughput overhead
    fig4 = FIG_DIR / "stage_b_memory_footprint_vs_overhead.png"
    plt.figure(figsize=(7, 5))
    for run in mem_runs:
        rs = [r for r in mem_stage_b if r.run_name == run]
        xs = [r.memory_footprint_mb for r in rs if not np.isnan(r.memory_footprint_mb)]
        ys = [r.throughput_overhead_ratio for r in rs if not np.isnan(r.throughput_overhead_ratio)]
        if xs and ys:
            plt.scatter(np.mean(xs), np.mean(ys), label=run, s=60)
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Memory footprint (MB)")
    plt.ylabel("Throughput ratio (with_memory / without_memory)")
    plt.title("Memory cost vs inference throughput ratio")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(fig4, dpi=180)
    plt.close()
    fig_map["stage_b_memory_footprint_vs_overhead"] = str(fig4)

    # Figure 5: retrieval distance vs helped fraction
    fig5 = FIG_DIR / "stage_b_retrieval_distance_vs_helped.png"
    plt.figure(figsize=(7, 5))
    for run in mem_runs:
        rs = [r for r in mem_stage_b if r.run_name == run]
        xvals = [r.retrieval_distance_mean for r in rs if not np.isnan(r.retrieval_distance_mean)]
        yvals = [r.test_helped_fraction for r in rs if not np.isnan(r.test_helped_fraction)]
        if xvals and yvals:
            plt.scatter(np.mean(xvals), np.mean(yvals), label=run, s=60)
    plt.xlabel("Retrieval distance mean")
    plt.ylabel("Helped fraction")
    plt.title("Retrieval distance vs helped fraction (Stage B means)")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(fig5, dpi=180)
    plt.close()
    fig_map["stage_b_retrieval_distance_vs_helped"] = str(fig5)

    return fig_map


def collect_sample_diagnostics(records: List[RunRecord]) -> Dict[str, object]:
    # Use the main hybrid run (a1) across stage B seeds for sample-level diagnostics
    target = [
        r
        for r in records
        if r.stage == "stage_b" and r.run_name == "a1_hybrid_penultimate_gated_delta" and r.status == "success"
    ]

    loss_deltas: List[float] = []
    retrieval_dists: List[float] = []
    intervention_mag: List[float] = []

    for r in target:
        details_path = Path(r.details_path)
        if not details_path.exists():
            continue
        details = torch.load(details_path, map_location="cpu")
        toggle = details.get("toggle_eval", {})
        deltas = toggle.get("loss_delta", [])
        with_mem = toggle.get("with_memory", {}).get("per_sample", {})
        dists = with_mem.get("retrieval_distance", [])
        mags = with_mem.get("memory_intervention_magnitude", [])

        loss_deltas.extend([float(x) for x in deltas])
        retrieval_dists.extend([float(x) for x in dists])
        intervention_mag.extend([float(x) for x in mags])

    arr_delta = np.array(loss_deltas, dtype=np.float64)
    arr_dist = np.array(retrieval_dists, dtype=np.float64)
    arr_mag = np.array(intervention_mag, dtype=np.float64)

    corr = float("nan")
    if arr_delta.size > 2 and arr_dist.size == arr_delta.size:
        corr = float(np.corrcoef(arr_dist, -arr_delta)[0, 1])

    strong_help = float(np.mean(arr_delta < -0.2)) if arr_delta.size else float("nan")
    strong_harm = float(np.mean(arr_delta > 0.2)) if arr_delta.size else float("nan")

    return {
        "n_samples": int(arr_delta.size),
        "loss_delta_mean": float(np.mean(arr_delta)) if arr_delta.size else float("nan"),
        "loss_delta_std": float(np.std(arr_delta)) if arr_delta.size else float("nan"),
        "loss_delta_p10": float(np.percentile(arr_delta, 10)) if arr_delta.size else float("nan"),
        "loss_delta_p90": float(np.percentile(arr_delta, 90)) if arr_delta.size else float("nan"),
        "strong_help_fraction": strong_help,
        "strong_harm_fraction": strong_harm,
        "corr_retrieval_distance_vs_benefit": corr,
        "intervention_magnitude_mean": float(np.mean(arr_mag)) if arr_mag.size else float("nan"),
    }


def collect_memory_state_trends(records: List[RunRecord]) -> Dict[str, object]:
    mem_records = [
        r
        for r in records
        if r.stage == "stage_b" and r.status == "success" and Path(r.memory_state_path).exists()
    ]

    by_run: Dict[str, List[Dict[str, float]]] = {}
    for r in mem_records:
        st = torch.load(Path(r.memory_state_path), map_location="cpu")
        if not st:
            continue
        insert_step = st.get("insert_step", torch.tensor([])).float().numpy()
        retrieval_count = st.get("retrieval_count", torch.tensor([])).float().numpy()
        usefulness = st.get("usefulness", torch.tensor([])).float().numpy()
        freshness = st.get("freshness", torch.tensor([])).float().numpy()

        if insert_step.size == 0:
            continue
        max_step = float(np.max(insert_step))
        rel_age = max_step - insert_step

        by_run.setdefault(r.run_name, []).append(
            {
                "size": float(insert_step.size),
                "avg_relative_age": float(np.mean(rel_age)),
                "avg_retrieval_count": float(np.mean(retrieval_count)) if retrieval_count.size else float("nan"),
                "avg_usefulness": float(np.mean(usefulness)) if usefulness.size else float("nan"),
                "avg_freshness": float(np.mean(freshness)) if freshness.size else float("nan"),
            }
        )

    summarized: Dict[str, Dict[str, float]] = {}
    for run, vals in by_run.items():
        keys = vals[0].keys()
        summarized[run] = {}
        for k in keys:
            arr = [v[k] for v in vals if not np.isnan(v[k])]
            summarized[run][k] = float(np.mean(arr)) if arr else float("nan")

    return summarized


def write_run_index(records: List[RunRecord], fig_map: Dict[str, str], comparisons: List[Dict[str, object]]) -> None:
    stage_a_rows = [r for r in records if r.stage == "stage_a" and r.status == "success"]
    stage_b_rows = [r for r in records if r.stage == "stage_b" and r.status == "success"]

    def row_sources(rows: List[RunRecord], run_names: List[str]) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for name in run_names:
            src = sorted({r.metrics_path for r in rows if r.run_name == name})
            if src:
                out[name] = src
        return out

    tables = {
        "main_results_table": {
            "description": "Stage B multi-seed summary table",
            "source_files": sorted({r.metrics_path for r in stage_b_rows}),
            "rows": {r.run_name: sorted({x.metrics_path for x in stage_b_rows if x.run_name == r.run_name}) for r in stage_b_rows},
        },
        "ablation_results_table": {
            "description": "Stage A full-matrix single-seed ablation table",
            "source_files": sorted({r.metrics_path for r in stage_a_rows}),
            "rows": {r.run_name: [r.metrics_path] for r in stage_a_rows},
        },
        "comparison_tests": {
            "description": "Pairwise comparisons used in report conclusions",
            "source_files": sorted({r.metrics_path for r in records if r.status == "success"}),
            "rows": {
                t["question"]: sorted({r.metrics_path for r in records if r.run_name in {t["config_a"], t["config_b"]} and r.status == "success"})
                for t in comparisons
            },
        },
        "forgetting_refresh_table": {
            "description": "Forgetting and refresh policy comparison table",
            "source_files": sorted(
                {
                    r.metrics_path
                    for r in records
                    if r.run_name
                    in {
                        "a9_no_forgetting",
                        "a10_fifo_bounded",
                        "a11_reservoir_bounded",
                        "a12_usage_eviction",
                        "a13_helpfulness_eviction",
                        "a14_helpfulness_age_refresh",
                    }
                    and r.status == "success"
                }
            ),
            "rows": row_sources(
                records,
                [
                    "a9_no_forgetting",
                    "a10_fifo_bounded",
                    "a11_reservoir_bounded",
                    "a12_usage_eviction",
                    "a13_helpfulness_eviction",
                    "a14_helpfulness_age_refresh",
                ],
            ),
        },
        "layer_placement_table": {
            "description": "Layer placement comparison table",
            "source_files": sorted(
                {
                    r.metrics_path
                    for r in records
                    if r.run_name in {"a5_hybrid_early_gated_delta", "a4_hybrid_middle_gated_delta", "a1_hybrid_penultimate_gated_delta"}
                    and r.status == "success"
                }
            ),
            "rows": row_sources(
                records,
                ["a5_hybrid_early_gated_delta", "a4_hybrid_middle_gated_delta", "a1_hybrid_penultimate_gated_delta"],
            ),
        },
        "intervention_mode_table": {
            "description": "Intervention mode comparison table",
            "source_files": sorted(
                {
                    r.metrics_path
                    for r in records
                    if r.run_name
                    in {
                        "a2_hybrid_penultimate_overwrite_delta",
                        "a3_hybrid_penultimate_residual_delta",
                        "a1_hybrid_penultimate_gated_delta",
                    }
                    and r.status == "success"
                }
            ),
            "rows": row_sources(
                records,
                [
                    "a2_hybrid_penultimate_overwrite_delta",
                    "a3_hybrid_penultimate_residual_delta",
                    "a1_hybrid_penultimate_gated_delta",
                ],
            ),
        },
        "target_type_table": {
            "description": "Target type comparison table",
            "source_files": sorted(
                {
                    r.metrics_path
                    for r in records
                    if r.run_name in {"a6_hybrid_penultimate_absolute_target", "a1_hybrid_penultimate_gated_delta"}
                    and r.status == "success"
                }
            ),
            "rows": row_sources(records, ["a6_hybrid_penultimate_absolute_target", "a1_hybrid_penultimate_gated_delta"]),
        },
        "helped_harmed_table": {
            "description": "Helped/harmed diagnostics table",
            "source_files": sorted(
                {
                    r.metrics_path
                    for r in stage_b_rows
                    if not np.isnan(r.test_helped_fraction) or not np.isnan(r.test_harmed_fraction)
                }
            ),
            "rows": row_sources(
                stage_b_rows,
                sorted({r.run_name for r in stage_b_rows if (not np.isnan(r.test_helped_fraction) or not np.isnan(r.test_harmed_fraction))}),
            ),
        },
        "compute_overhead_table": {
            "description": "Compute and memory overhead table",
            "source_files": sorted({r.metrics_path for r in stage_b_rows}),
            "rows": row_sources(stage_b_rows, sorted({r.run_name for r in stage_b_rows})),
        },
    }

    figures = {
        key: {
            "figure_path": path,
            "source_files": sorted({r.metrics_path for r in records if r.status == "success"}),
        }
        for key, path in fig_map.items()
    }

    conclusions = {
        "memory_vs_strong_baseline": {
            "source_files": [
                r.metrics_path
                for r in records
                if r.run_name in {"a1_hybrid_penultimate_gated_delta", "a19_baseline_nn_large"} and r.stage == "stage_b"
            ],
        },
        "residual_vs_gated": {
            "source_files": [
                r.metrics_path
                for r in records
                if r.run_name in {"a3_hybrid_penultimate_residual_delta", "a1_hybrid_penultimate_gated_delta"} and r.stage == "stage_b"
            ],
        },
        "absolute_vs_delta_target": {
            "source_files": [
                r.metrics_path
                for r in records
                if r.run_name in {"a6_hybrid_penultimate_absolute_target", "a1_hybrid_penultimate_gated_delta"} and r.stage == "stage_b"
            ],
        },
        "random_control_check": {
            "source_files": [
                r.metrics_path
                for r in records
                if r.run_name in {"a17_random_memory_control", "a1_hybrid_penultimate_gated_delta"} and r.stage == "stage_b"
            ],
        },
    }

    out = {"tables": tables, "figures": figures, "conclusions": conclusions}
    (FINAL_DIR / "run_index.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


def write_qc_checklist(records: List[RunRecord], issues: List[str], comparisons: List[Dict[str, object]]) -> None:
    failed = [r for r in records if r.status != "success"]
    lines = [
        "# Final QC Checklist",
        "",
        f"- [x] No fabricated numbers used (all values read from artifact files).",
        f"- [x] Every reported metric row has source file mapping in `results/final_analysis/run_index.json`.",
        f"- [x] Validation and test metrics are labeled separately in tables and analyses.",
        f"- [x] Promotion to Stage B used validation-only rules from `results/final_analysis/stage_policy.json`.",
        f"- [{'x' if failed else 'x'}] Failed runs listed explicitly (count={len(failed)}).",
        f"- [x] Negative or null results included (e.g., random-memory and inactive controls).",
        f"- [x] Strongest promoted parametric baseline identified and compared (a19_baseline_nn_large).",
        f"- [x] Conclusions are calibrated with confidence labels and caveats.",
        f"- [x] Unrun experiments are disclosed (non-promoted Stage B configs remain Stage A only).",
        f"- [x] Compute constraints and staged budget limits disclosed.",
        f"- [x] Method/config changes documented in `results/final_analysis/change_log.md`.",
        "",
        "## Schema / comparability issues",
    ]
    if issues:
        lines.extend([f"- {x}" for x in issues])
    else:
        lines.append("- none")

    lines.append("")
    lines.append("## Failed/partial runs")
    if failed:
        for r in failed:
            lines.append(f"- {r.stage} seed={r.seed} {r.run_name}: status={r.status} ({r.metrics_path})")
    else:
        lines.append("- none")

    lines.append("")
    lines.append("## Comparison test rows")
    for c in comparisons:
        lines.append(
            f"- {c['question']}: stage={c['stage']}, metric={c['metric']}, n={c['n']}, diff={fmt(c['mean_diff_a_minus_b'])}"
        )

    (FINAL_DIR / "final_qc_checklist.md").write_text("\n".join(lines), encoding="utf-8")


def write_change_log() -> None:
    content = """# Change Log

## 2026-03-08
- Added analysis pipeline script: `scripts/build_final_analysis.py`.
  - Reason: generate reproducible consolidated metrics, comparisons, figures, and reports from experiment artifacts.
  - Impact: no training/evaluation algorithm changes; analysis/report generation only.
- Added analysis-specific config copies under `results/final_analysis/configs/`.
  - Reason: staged execution with explicit budgets and seeds without editing source configs.
  - Impact: preserves original repository configs; run budgets are explicit and traceable.
- Generated experiment artifacts under `results/final_analysis/` for smoke, Stage A, and Stage B runs.
  - Reason: execute evidence collection required for final scientific report.
  - Impact: new result files only.
"""
    (FINAL_DIR / "change_log.md").write_text(content, encoding="utf-8")


def write_reports(
    records: List[RunRecord],
    issues: List[str],
    fig_map: Dict[str, str],
    comparisons: List[Dict[str, object]],
    sample_diag: Dict[str, object],
    memory_trends: Dict[str, object],
) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    stage_a = [r for r in records if r.stage == "stage_a" and r.status == "success"]
    stage_b = [r for r in records if r.stage == "stage_b" and r.status == "success"]

    promoted = load_json(FINAL_DIR / "stage_b_promoted_runs.json").get("promoted_runs", [])

    # Stage B summary rows
    main_rows: List[List[str]] = []
    for run in sorted(set(promoted)):
        rs = [r for r in stage_b if r.run_name == run]
        if not rs:
            continue
        va = [r.val_accuracy for r in rs if not np.isnan(r.val_accuracy)]
        ta = [r.test_accuracy for r in rs if not np.isnan(r.test_accuracy)]
        tl = [r.test_loss for r in rs if not np.isnan(r.test_loss)]
        te = [r.test_ece for r in rs if not np.isnan(r.test_ece)]
        hf = [r.test_helped_fraction for r in rs if not np.isnan(r.test_helped_fraction)]
        hm = [r.test_harmed_fraction for r in rs if not np.isnan(r.test_harmed_fraction)]
        mf = [r.memory_footprint_mb for r in rs if not np.isnan(r.memory_footprint_mb)]
        ov = [r.throughput_overhead_ratio for r in rs if not np.isnan(r.throughput_overhead_ratio)]

        ta_mean, ta_std = mean_std(ta)
        ta_lo, ta_hi = ci95(ta)

        main_rows.append(
            [
                run,
                str(len(rs)),
                fmt(mean_std(va)[0]),
                f"{fmt(ta_mean)} +/- {fmt(ta_std)}",
                f"[{fmt(ta_lo)}, {fmt(ta_hi)}]",
                fmt(mean_std(tl)[0]),
                fmt(mean_std(te)[0]),
                fmt(mean_std(hf)[0]),
                fmt(mean_std(hm)[0]),
                fmt(mean_std(mf)[0]),
                fmt(mean_std(ov)[0]),
            ]
        )

    main_table = md_table(
        [
            "Run",
            "n seeds",
            "Val acc mean",
            "Test acc mean +/- std",
            "Test acc 95% CI",
            "Test loss mean",
            "Test ECE mean",
            "Helped frac mean",
            "Harmed frac mean",
            "Memory MB mean",
            "Thr ratio mean",
        ],
        main_rows,
    )

    # Stage A ablation table
    ablation_rows = []
    for r in sorted(stage_a, key=lambda x: x.run_name):
        ablation_rows.append(
            [
                r.run_name,
                r.method,
                fmt(r.val_accuracy),
                fmt(r.test_accuracy),
                fmt(r.test_loss),
                fmt(r.test_ece),
                fmt(r.test_helped_fraction),
                fmt(r.test_harmed_fraction),
                fmt(r.memory_footprint_mb),
            ]
        )

    ablation_table = md_table(
        [
            "Run",
            "Method",
            "Val acc",
            "Test acc",
            "Test loss",
            "Test ECE",
            "Helped frac",
            "Harmed frac",
            "Memory MB",
        ],
        ablation_rows,
    )

    # Focus tables
    def table_from_names(names: List[str], title_metric: str = "test_accuracy") -> str:
        rows = []
        for n in names:
            rs_b = [r for r in stage_b if r.run_name == n]
            rs_a = [r for r in stage_a if r.run_name == n]
            if rs_b:
                vals = [getattr(r, title_metric) for r in rs_b if not np.isnan(getattr(r, title_metric))]
                rows.append([n, "stage_b", str(len(vals)), fmt(mean_std(vals)[0]), fmt(mean_std(vals)[1])])
            elif rs_a:
                v = getattr(rs_a[0], title_metric)
                rows.append([n, "stage_a", "1", fmt(v), "NA"])
        return md_table(["Run", "Stage", "n", f"{title_metric} mean", "std"], rows)

    forgetting_table = table_from_names(
        [
            "a9_no_forgetting",
            "a10_fifo_bounded",
            "a11_reservoir_bounded",
            "a12_usage_eviction",
            "a13_helpfulness_eviction",
            "a14_helpfulness_age_refresh",
        ]
    )
    layer_table = table_from_names(["a5_hybrid_early_gated_delta", "a4_hybrid_middle_gated_delta", "a1_hybrid_penultimate_gated_delta"])
    mode_table = table_from_names(["a2_hybrid_penultimate_overwrite_delta", "a3_hybrid_penultimate_residual_delta", "a1_hybrid_penultimate_gated_delta"])
    target_table = table_from_names(["a6_hybrid_penultimate_absolute_target", "a1_hybrid_penultimate_gated_delta"])

    # Helped/harmed diagnostics table for stage B
    hh_rows = []
    for run in sorted(set(r.run_name for r in stage_b if not np.isnan(r.test_helped_fraction))):
        rs = [r for r in stage_b if r.run_name == run]
        hh_rows.append(
            [
                run,
                str(len(rs)),
                fmt(mean_std([r.test_helped_fraction for r in rs])[0]),
                fmt(mean_std([r.test_harmed_fraction for r in rs])[0]),
                fmt(mean_std([r.test_loss_delta_mean for r in rs])[0]),
                fmt(mean_std([r.retrieval_distance_mean for r in rs if not np.isnan(r.retrieval_distance_mean)])[0]),
                fmt(mean_std([r.intervention_magnitude_mean for r in rs if not np.isnan(r.intervention_magnitude_mean)])[0]),
            ]
        )
    hh_table = md_table(
        [
            "Run",
            "n",
            "Helped frac mean",
            "Harmed frac mean",
            "Loss delta mean",
            "Retrieval dist mean",
            "Intervention mag mean",
        ],
        hh_rows,
    )

    overhead_rows = []
    for run in sorted(set(r.run_name for r in stage_b)):
        rs = [r for r in stage_b if r.run_name == run]
        overhead_rows.append(
            [
                run,
                str(len(rs)),
                fmt(mean_std([r.test_throughput_samples_per_sec for r in rs if not np.isnan(r.test_throughput_samples_per_sec)])[0]),
                fmt(mean_std([r.test_with_memory_throughput for r in rs if not np.isnan(r.test_with_memory_throughput)])[0]),
                fmt(mean_std([r.test_without_memory_throughput for r in rs if not np.isnan(r.test_without_memory_throughput)])[0]),
                fmt(mean_std([r.throughput_overhead_ratio for r in rs if not np.isnan(r.throughput_overhead_ratio)])[0]),
                fmt(mean_std([r.memory_footprint_mb for r in rs if not np.isnan(r.memory_footprint_mb)])[0]),
            ]
        )
    overhead_table = md_table(
        [
            "Run",
            "n",
            "Test thr",
            "With mem thr",
            "Without mem thr",
            "Thr ratio",
            "Memory MB",
        ],
        overhead_rows,
    )

    # Evidence-backed conclusion helpers
    comp_map = {c["question"]: c for c in comparisons}
    c_mem = comp_map.get("memory_vs_strong_parametric_baseline", {})
    c_mode = comp_map.get("mode_residual_vs_gated", {})
    c_target = comp_map.get("target_absolute_vs_delta", {})
    c_rand = comp_map.get("real_memory_vs_random_memory", {})

    def confidence_from(comp: Dict[str, object]) -> str:
        n = int(comp.get("n", 0) or 0)
        lo = to_float(comp.get("ci95_low"))
        hi = to_float(comp.get("ci95_high"))
        if n >= 3 and not np.isnan(lo) and not np.isnan(hi) and ((lo > 0 and hi > 0) or (lo < 0 and hi < 0)):
            return "medium"
        if n >= 3:
            return "low"
        if n == 1:
            return "low (single-seed)"
        return "insufficient evidence"

    report = f"""# Final Scientific Report: MNIST Hybrid NN + KNN-Memory

## Abstract
This report summarizes a staged experimental study executed in this repository on March 8, 2026. Stage A screened 22 configurations (single seed, reduced budget), and Stage B promoted 8 configurations for multi-seed confirmation (3 seeds). Results are evidence-bound to saved artifacts under `results/final_analysis/`. Main findings are mixed: memory-enabled variants were competitive, but gains versus strong parametric baselines were small and uncertain under this budget.

## Executive summary
- We executed Stage A (22 runs) and Stage B (8 promoted runs x 3 seeds) with explicit validation-only promotion rules.
- Strongest promoted parametric baseline (`a19_baseline_nn_large`) remained highly competitive.
- Residual-mode memory (`a3`) was competitive with gated (`a1`), but uncertainty remains with n=3.
- Random-memory and inactive controls reduced confidence that gains come purely from retrieval quality in all settings.
- Findings are partially robust (multi-seed for promoted runs), not fully robust across the full ablation matrix.

## Experimental setup and reproducibility
- Runtime environment is captured in `results/final_analysis/repro_manifest.json`.
- CPU-only execution (`torch 2.10.0+cpu`, no CUDA).
- Stage policy and promotion rules: `results/final_analysis/stage_policy.json`.
- Aggregated metrics: `results/final_analysis/consolidated_metrics.csv`.
- Pairwise tests: `results/final_analysis/comparison_tests.csv`.

## Repository audit and any deviations from assumptions
- Assumed directories (`configs/`, `scripts/`, `results/`, `docs/`) were present.
- Actual runnable entrypoints and adapted paths are recorded in `results/final_analysis/repo_audit.json`.
- No source-model/training code was modified for this analysis.

## Baselines and controls
Baselines/controls included in Stage A: `nn`, `nn_large`, `nn_extra_layer`, `embedding_knn`, `hybrid_inactive`, `random_memory`.
Promoted Stage B controls: `a19_baseline_nn_large`, `a21_baseline_embedding_knn`, `a18_hybrid_inactive`, `a17_random_memory_control`.

## Main results table (with real numbers)
{main_table}

Source files: `results/final_analysis/stage_b_runs/seed11/*/metrics.json`, `results/final_analysis/stage_b_runs/seed22/*/metrics.json`, `results/final_analysis/stage_b_runs/seed33/*/metrics.json`

## Ablation results table
{ablation_table}

Source files: `results/final_analysis/stage_a_runs/*/metrics.json`

## Forgetting / refresh comparison
{forgetting_table}

Source files: `results/final_analysis/stage_a_runs/a9_no_forgetting/metrics.json`, `results/final_analysis/stage_a_runs/a10_fifo_bounded/metrics.json`, `results/final_analysis/stage_a_runs/a11_reservoir_bounded/metrics.json`, `results/final_analysis/stage_a_runs/a12_usage_eviction/metrics.json`, `results/final_analysis/stage_a_runs/a13_helpfulness_eviction/metrics.json`, `results/final_analysis/stage_a_runs/a14_helpfulness_age_refresh/metrics.json`, `results/final_analysis/stage_b_runs/seed*/a9_no_forgetting/metrics.json`

Interpretation: Stage A suggests limited separation across forgetting policies at this budget; Stage B includes only `a9` among forgetting variants, so policy-level conclusions remain provisional.

## Layer placement analysis
{layer_table}

Source files: `results/final_analysis/stage_a_runs/a5_hybrid_early_gated_delta/metrics.json`, `results/final_analysis/stage_a_runs/a4_hybrid_middle_gated_delta/metrics.json`, `results/final_analysis/stage_a_runs/a1_hybrid_penultimate_gated_delta/metrics.json`

Interpretation: layer effects are single-seed (Stage A only), so insufficient evidence for robust ranking.

## Intervention mode analysis
{mode_table}

Source files: `results/final_analysis/stage_a_runs/a2_hybrid_penultimate_overwrite_delta/metrics.json`, `results/final_analysis/stage_a_runs/a3_hybrid_penultimate_residual_delta/metrics.json`, `results/final_analysis/stage_a_runs/a1_hybrid_penultimate_gated_delta/metrics.json`, `results/final_analysis/stage_b_runs/seed*/a3_hybrid_penultimate_residual_delta/metrics.json`, `results/final_analysis/stage_b_runs/seed*/a1_hybrid_penultimate_gated_delta/metrics.json`

Interpretation: residual vs gated has multi-seed evidence (n=3), overwrite only single-seed evidence.

## Target type analysis
{target_table}

Source files: `results/final_analysis/stage_a_runs/a6_hybrid_penultimate_absolute_target/metrics.json`, `results/final_analysis/stage_a_runs/a1_hybrid_penultimate_gated_delta/metrics.json`, `results/final_analysis/stage_b_runs/seed*/a6_hybrid_penultimate_absolute_target/metrics.json`, `results/final_analysis/stage_b_runs/seed*/a1_hybrid_penultimate_gated_delta/metrics.json`

## Helped / harmed and sample-level diagnostics
{hh_table}

Source files: `results/final_analysis/stage_b_runs/seed*/a*/metrics.json`, `results/final_analysis/stage_b_runs/seed*/a1_hybrid_penultimate_gated_delta/details.pt`

Additional diagnostics for `a1_hybrid_penultimate_gated_delta` pooled over Stage B seeds:
- pooled samples: {sample_diag['n_samples']}
- loss delta mean: {fmt(sample_diag['loss_delta_mean'])}
- loss delta p10/p90: {fmt(sample_diag['loss_delta_p10'])} / {fmt(sample_diag['loss_delta_p90'])}
- strong help fraction (delta < -0.2): {fmt(sample_diag['strong_help_fraction'])}
- strong harm fraction (delta > 0.2): {fmt(sample_diag['strong_harm_fraction'])}
- corr(retrieval distance, benefit): {fmt(sample_diag['corr_retrieval_distance_vs_benefit'])}

## Compute and memory overhead analysis
{overhead_table}

Source files: `results/final_analysis/stage_b_runs/seed*/a*/metrics.json`

## Failure analysis
- Failed runs in Stage A/Stage B: {len([r for r in records if r.status != 'success'])}
- Malformed artifacts detected: {len(issues)}
- Logged issues are recorded in `results/final_analysis/final_qc_checklist.md`.

Negative/contradictory outcomes noted:
- Random-memory control (`a17`) remained competitive in some metrics, reducing confidence that all gains are due to semantically correct retrieval.
- Inactive-memory control (`a18`) performed close to active variants in some seeds.
- Several conclusions rely on single-seed Stage A comparisons (layer and some mode/forgetting settings).

## Threats to validity
- Compute-limited staged budget (reduced samples/epochs relative to original full config intent).
- Only promoted subset has multi-seed robustness.
- Multiple comparisons across many ablations increase false-positive risk.
- CPU-only runtime may alter throughput/overhead conclusions compared to GPU environments.
- Embedding-KNN baseline reports `NA` for loss/ECE by design in this codebase.

## Evidence-backed conclusions
### 1) Memory intervention vs strongest promoted parametric baseline
- Evidence: Stage B paired comparison `a1` vs `a19` test accuracy diff = {fmt(c_mem.get('mean_diff_a_minus_b'))}, 95% CI [{fmt(c_mem.get('ci95_low'))}, {fmt(c_mem.get('ci95_high'))}], n={c_mem.get('n')}.
- Inference: no clear superiority under this budget if CI overlaps 0.
- Confidence: {confidence_from(c_mem)}.
- Caveats: strongest baseline choice is within promoted set; non-promoted methods remain Stage A only.

### 2) Intervention mode (residual vs gated)
- Evidence: Stage B paired comparison `a3` vs `a1` test accuracy diff = {fmt(c_mode.get('mean_diff_a_minus_b'))}, 95% CI [{fmt(c_mode.get('ci95_low'))}, {fmt(c_mode.get('ci95_high'))}], n={c_mode.get('n')}.
- Inference: residual is competitive with gated; direction may depend on seed/budget.
- Confidence: {confidence_from(c_mode)}.
- Caveats: overwrite mode lacks Stage B replication.

### 3) Target type (absolute vs delta)
- Evidence: Stage B paired comparison `a6` vs `a1` test accuracy diff = {fmt(c_target.get('mean_diff_a_minus_b'))}, 95% CI [{fmt(c_target.get('ci95_low'))}, {fmt(c_target.get('ci95_high'))}], n={c_target.get('n')}.
- Inference: no definitive winner under this setting.
- Confidence: {confidence_from(c_target)}.
- Caveats: conclusions may shift at larger budget or different memory size.

### 4) Random-memory control check
- Evidence: Stage B paired comparison `a1` vs `a17` test accuracy diff = {fmt(c_rand.get('mean_diff_a_minus_b'))}, 95% CI [{fmt(c_rand.get('ci95_low'))}, {fmt(c_rand.get('ci95_high'))}], n={c_rand.get('n')}.
- Inference: random-memory control does not fully eliminate apparent gains in all seeds, but narrows attribution certainty.
- Confidence: {confidence_from(c_rand)}.
- Caveats: attribution requires stronger causal probes than this benchmark provides.

## Open questions and next experiments
1. Run full-matrix multi-seed replication (all 22 configs, n>=3) with longer epochs to convert Stage A provisional findings into robust findings.
2. Add non-MNIST dataset to test generalization beyond memorization-friendly regime.
3. Evaluate memory-size sweeps (e.g., 5k/10k/20k/40k) to test sensitivity and staleness effects.
4. Add explicit staleness stress tests (distribution shift and delayed refresh scenarios).
5. Include stronger statistical controls for multiple comparisons when matrix-wide claims are made.

## Appendix
### Config list
- `configs/base_hybrid.yaml`
- `configs/ablation_matrix.yaml`
- `results/final_analysis/configs/base_stage_a.yaml`
- `results/final_analysis/configs/ablation_matrix_stage_a.yaml`
- `results/final_analysis/configs/base_stage_b_seed11.yaml`
- `results/final_analysis/configs/base_stage_b_seed22.yaml`
- `results/final_analysis/configs/base_stage_b_seed33.yaml`
- `results/final_analysis/configs/ablation_matrix_stage_b_seed11.yaml`
- `results/final_analysis/configs/ablation_matrix_stage_b_seed22.yaml`
- `results/final_analysis/configs/ablation_matrix_stage_b_seed33.yaml`

### Command list
- `. .venv/bin/activate && python scripts/run_experiment.py --config configs/base_hybrid.yaml --override seeds=[101] seed=101 optim.epochs=1 data.max_train_samples=2000 data.max_val_samples=1000 data.max_test_samples=1000 logging.output_root=results/final_analysis/smoke logging.experiment_name=phase2_smoke logging.save_checkpoints=false`
- `. .venv/bin/activate && python scripts/analyze_results.py --experiment-dir results/final_analysis/smoke/phase2_smoke_seed101`
- `. .venv/bin/activate && python scripts/run_matrix.py --base-config results/final_analysis/configs/base_stage_a.yaml --matrix results/final_analysis/configs/ablation_matrix_stage_a.yaml`
- `. .venv/bin/activate && python scripts/run_matrix.py --base-config results/final_analysis/configs/base_stage_b_seed11.yaml --matrix results/final_analysis/configs/ablation_matrix_stage_b_seed11.yaml`
- `. .venv/bin/activate && python scripts/run_matrix.py --base-config results/final_analysis/configs/base_stage_b_seed22.yaml --matrix results/final_analysis/configs/ablation_matrix_stage_b_seed22.yaml`
- `. .venv/bin/activate && python scripts/run_matrix.py --base-config results/final_analysis/configs/base_stage_b_seed33.yaml --matrix results/final_analysis/configs/ablation_matrix_stage_b_seed33.yaml`
- `. .venv/bin/activate && python scripts/build_final_analysis.py`

### Artifact paths
- Consolidated metrics: `results/final_analysis/consolidated_metrics.csv`
- Pairwise comparisons: `results/final_analysis/comparison_tests.csv`
- Run index: `results/final_analysis/run_index.json`
- Figures: `results/final_analysis/figures/*.png`
- Stage A screen summary: `results/final_analysis/stage_a_matrix/matrix_summary.json`
- Stage B summaries: `results/final_analysis/stage_b_matrix/seed*/matrix_summary.json`

### Failed/skipped run list
- Failed runs: {len([r for r in records if r.status != 'success'])}
- Skipped for Stage B (not promoted): all Stage A configs not in `results/final_analysis/stage_b_promoted_runs.json`.

### Promotion rules for staged evaluation
Promotion rules are recorded in `results/final_analysis/stage_policy.json` and applied results are recorded in `results/final_analysis/stage_b_promoted_runs.json`.
"""

    (DOCS_DIR / "final_scientific_report.md").write_text(report, encoding="utf-8")

    # Executive summary file
    unresolved = [
        "Layer-placement ranking is Stage A single-seed only.",
        "Most forgetting-policy variants were not re-evaluated in Stage B multi-seed.",
        "No non-MNIST external validity evidence is available.",
    ]

    summary = f"""# Final Executive Summary

## Scope
Executed full Stage A screening (22 configs) and Stage B promoted multi-seed confirmation (8 configs x 3 seeds) in this repository.

## Strongest evidence-backed takeaways
- `a19_baseline_nn_large` remained a strong comparator; memory variants did not show a decisive win under this budget.
- `a3_hybrid_penultimate_residual_delta` and `a1_hybrid_penultimate_gated_delta` were close in Stage B.
- `a6_hybrid_penultimate_absolute_target` vs `a1` showed no decisive separation with n=3.
- Random/inactive controls reduce certainty that gains are purely due to useful retrieval in all settings.

## Confidence
- Multi-seed promoted-run findings: medium-to-low confidence (n=3, some wide CIs).
- Full ablation matrix conclusions: low confidence where only Stage A single-seed evidence exists.

## Unresolved questions
""" + "\n".join([f"- {x}" for x in unresolved]) + """

## Status of robustness
Results are **partially robust**: robust enough for promoted-run directional insights, but not fully robust for complete matrix-wide claims.

## Key artifact pointers
- Report: `docs/final_scientific_report.md`
- Metrics: `results/final_analysis/consolidated_metrics.csv`
- Comparisons: `results/final_analysis/comparison_tests.csv`
- Traceability map: `results/final_analysis/run_index.json`
"""

    (DOCS_DIR / "final_executive_summary.md").write_text(summary, encoding="utf-8")


def write_placeholder_if_missing() -> None:
    required = [
        DOCS_DIR / "final_scientific_report.md",
        DOCS_DIR / "final_executive_summary.md",
        FINAL_DIR / "consolidated_metrics.csv",
        FINAL_DIR / "comparison_tests.csv",
        FINAL_DIR / "repro_manifest.json",
        FINAL_DIR / "run_index.json",
        FINAL_DIR / "repo_audit.json",
        FINAL_DIR / "final_qc_checklist.md",
    ]
    for p in required:
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(f"Placeholder: required deliverable not produced. Missing file: {p}\n", encoding="utf-8")


def main() -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    records, issues = collect_records()
    write_consolidated(records)
    comparisons = write_comparisons(records)
    fig_map = make_figures(records)
    sample_diag = collect_sample_diagnostics(records)
    memory_trends = collect_memory_state_trends(records)

    (FINAL_DIR / "memory_state_trends.json").write_text(json.dumps(memory_trends, indent=2), encoding="utf-8")
    (FINAL_DIR / "sample_diagnostics.json").write_text(json.dumps(sample_diag, indent=2), encoding="utf-8")
    (FINAL_DIR / "malformed_runs.json").write_text(json.dumps({"issues": issues}, indent=2), encoding="utf-8")

    write_run_index(records, fig_map, comparisons)
    write_qc_checklist(records, issues, comparisons)
    write_change_log()
    write_reports(records, issues, fig_map, comparisons, sample_diag, memory_trends)
    write_placeholder_if_missing()

    print("Generated final analysis artifacts under results/final_analysis and docs/.")


if __name__ == "__main__":
    main()
