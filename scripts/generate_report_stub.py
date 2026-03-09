from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a report stub markdown from experiment summaries")
    parser.add_argument("--output", type=str, default="docs/report_draft.md")
    parser.add_argument("--aggregate", type=str, default="")
    parser.add_argument("--matrix-summary", type=str, default="")
    return parser.parse_args()


def _load_json(path: str) -> Dict[str, object]:
    p = Path(path)
    if not path or not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _format_metric_block(metrics: Dict[str, object]) -> str:
    lines = []
    for key, value in sorted(metrics.items()):
        if isinstance(value, dict) and "mean" in value:
            lines.append(f"- {key}: mean={value['mean']:.6f}, std={value['std']:.6f}, n={value['n']}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    aggregate = _load_json(args.aggregate)
    matrix = _load_json(args.matrix_summary)

    metrics_text = _format_metric_block(aggregate.get("aggregated_metrics", {}))

    matrix_rows = []
    for run in matrix.get("matrix", []):
        name = run.get("name", "")
        final = run.get("final_metrics", {})
        acc = final.get("test_accuracy", "NA")
        loss = final.get("test_loss", "NA")
        matrix_rows.append(f"| {name} | {acc} | {loss} |")

    if not matrix_rows:
        matrix_rows = ["| run | test_accuracy | test_loss |", "|---|---:|---:|"]
    else:
        matrix_rows.insert(0, "| run | test_accuracy | test_loss |")
        matrix_rows.insert(1, "|---|---:|---:|")

    report = f"""# Draft Report: Hybrid NN + KNN-Memory on MNIST

## Abstract
[Fill after analyzing full results]

## Executive summary
[Fill concise answer, include caveats]

## Main aggregated metrics (from `aggregate_metrics.json`)
{metrics_text or '[No aggregate file provided]'}

## Matrix snapshot
{'\n'.join(matrix_rows)}

## Interpretation checklist
- Distinguish evidence vs speculation.
- Report negative results.
- Compare against strong baselines.
- Analyze helped vs harmed samples.
- Discuss memorization risk and staleness effects.

## Open questions
- Which target construction is most stable under drift?
- Does helpfulness-based eviction remain robust with larger memory sizes?
- Are gains preserved on non-MNIST datasets?
"""

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"Wrote report stub: {out}")


if __name__ == "__main__":
    main()
