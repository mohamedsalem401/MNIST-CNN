from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert matrix summary JSON into CSV table")
    parser.add_argument("--summary", type=str, required=True, help="Path to matrix_summary.json")
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary)
    with summary_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    metrics = set()
    for run in data.get("matrix", []):
        row = {"name": run.get("name", "")}
        final = run.get("final_metrics", {})
        toggle = run.get("toggle_eval", {})
        merged = {**final, **toggle}
        for k, v in merged.items():
            if isinstance(v, (int, float)):
                row[k] = v
                metrics.add(k)
        rows.append(row)

    output_path = Path(args.output) if args.output else summary_path.with_suffix(".csv")
    fields = ["name", *sorted(metrics)]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote CSV summary: {output_path}")


if __name__ == "__main__":
    main()
