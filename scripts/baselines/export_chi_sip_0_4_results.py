#!/usr/bin/env python3
"""Combine CHI/SIP five-seed baseline results and compute mean/std."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = REPO_ROOT / "csv_files" / "BrainAI"
ALL_RUNS_PATH = RESULT_ROOT / "baseline_chi_sip_0_4_all_runs.csv"
SUMMARY_PATH = RESULT_ROOT / "baseline_chi_sip_0_4_mean_std.csv"

DATASETS = {
    "CHI": {"target": "RISK", "group": "baseline_repeats_CHI_RISK_0_4"},
    "SIP": {"target": "FLOW", "group": "baseline_repeats_SIP_FLOW_0_4"},
}
MODELS = ("STGODE", "CMuST", "STGCN")
SEEDS = (2025, 2026, 2027, 2028, 2029)
METRICS = ("mae", "rmse", "mape")


def load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset, info in DATASETS.items():
        source = RESULT_ROOT / dataset / info["group"] / "repeat_metrics.csv"
        if not source.is_file():
            raise FileNotFoundError(source)
        with source.open(newline="", encoding="utf-8") as handle:
            source_rows = list(csv.DictReader(handle))
        if len(source_rows) != len(MODELS) * len(SEEDS):
            raise ValueError(f"{source}: expected 15 rows, found {len(source_rows)}")
        for row in source_rows:
            rows.append(
                {
                    "dataset": dataset,
                    "target": info["target"],
                    "time_range": "00:00-04:00",
                    "model": row["model"],
                    "run": int(row["run"]),
                    "seed": int(row["seed"]),
                    "mae": float(row["mae"]),
                    "rmse": float(row["rmse"]),
                    "mape": float(row["mape"]),
                }
            )
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for dataset, info in DATASETS.items():
        for model in MODELS:
            selected = [
                row for row in rows
                if row["dataset"] == dataset and row["model"] == model
            ]
            if len(selected) != len(SEEDS):
                raise ValueError(f"{dataset}/{model}: expected 5 rows, found {len(selected)}")
            output: dict[str, object] = {
                "dataset": dataset,
                "target": info["target"],
                "time_range": "00:00-04:00",
                "model": model,
                "num_seeds": len(selected),
                "seeds": " ".join(str(seed) for seed in SEEDS),
            }
            for metric in METRICS:
                values = [float(row[metric]) for row in selected]
                output[f"{metric}_mean"] = round(statistics.mean(values), 6)
                output[f"{metric}_std"] = round(statistics.pstdev(values), 6)
            summary.append(output)
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = load_rows()
    summary = summarize(rows)
    write_csv(ALL_RUNS_PATH, rows)
    write_csv(SUMMARY_PATH, summary)
    print(f"All runs: {ALL_RUNS_PATH} ({len(rows)} rows)")
    print(f"Mean/std: {SUMMARY_PATH} ({len(summary)} rows)")


if __name__ == "__main__":
    main()
