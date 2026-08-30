#!/usr/bin/env python3
"""Extract final per-horizon test metrics from BrainAI baseline logs."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


HORIZON_RE = re.compile(
    r"Horizon\s+(?P<horizon>\d+),\s*"
    r"Test MAE:\s*(?P<mae>[0-9.+\-eE]+),\s*"
    r"Test RMSE:\s*(?P<rmse>[0-9.+\-eE]+),\s*"
    r"Test MAPE:\s*(?P<mape>[0-9.+\-eE]+)"
)


def latest_log(log_dir: Path) -> Path | None:
    logs = sorted(log_dir.glob("*.log"))
    return logs[-1] if logs else None


def final_horizon_block(text: str) -> list[re.Match[str]]:
    matches = list(HORIZON_RE.finditer(text))
    if not matches:
        return []

    block = []
    for match in reversed(matches):
        block.append(match)
        if int(match.group("horizon")) == 1:
            break
    return list(reversed(block))


def extract_rows(log_root: Path) -> list[dict[str, str]]:
    rows = []
    for dataset_dir in sorted(p for p in log_root.iterdir() if p.is_dir()):
        for baseline_dir in sorted(dataset_dir.glob("baseline_*")):
            if not baseline_dir.is_dir():
                continue
            log_file = latest_log(baseline_dir)
            if log_file is None:
                continue

            text = log_file.read_text(errors="ignore")
            baseline = baseline_dir.name.removeprefix("baseline_")
            for match in final_horizon_block(text):
                rows.append(
                    {
                        "dataset": dataset_dir.name,
                        "baseline": baseline,
                        "horizon": match.group("horizon"),
                        "test_mae": match.group("mae"),
                        "test_rmse": match.group("rmse"),
                        "test_mape": match.group("mape"),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", default="logs/BrainAI", type=Path)
    parser.add_argument(
        "--output",
        default="csv_files/BrainAI/baseline_horizon_test_metrics.csv",
        type=Path,
    )
    args = parser.parse_args()

    rows = extract_rows(args.log_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "baseline",
                "horizon",
                "test_mae",
                "test_rmse",
                "test_mape",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
