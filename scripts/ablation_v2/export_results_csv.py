#!/usr/bin/env python3
"""Export the six component ablations across NYC/CHI/SIP to one CSV."""

from __future__ import annotations

import csv
from pathlib import Path

from export_results_xlsx import load_results


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "csv_files" / "BrainAI" / "ablation_v2_component_results.csv"

PARENT_MODULE = {
    "gradc": "ProgCA",
    "dac": "ProgCA",
    "datase": "EvoPoG",
    "reinedit": "EvoPoG",
    "mgo": "CoML",
    "arcon": "CoML",
}

FIELDNAMES = [
    "parent_module",
    "ablation",
    "setting",
    "dataset",
    "target",
    "time_range",
    "seed",
    "mae",
    "rmse",
    "mape",
    "status",
    "experiment_name",
    "summary_path",
    "log_path",
]


def main() -> None:
    results = load_results()
    rows = []
    for item in results:
        rows.append(
            {
                "parent_module": PARENT_MODULE[item["mode"]],
                "ablation": item["label"],
                "setting": item["detail"],
                "dataset": item["dataset"],
                "target": item["target"],
                "time_range": item["time"],
                "seed": item["seed"],
                "mae": item["mae"],
                "rmse": item["rmse"],
                "mape": item["mape"],
                "status": item["status"],
                "experiment_name": item["experiment"],
                "summary_path": item["summary_path"],
                "log_path": item["log_path"],
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(rows)} completed results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
