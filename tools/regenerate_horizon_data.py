#!/usr/bin/env python3
import argparse
import os
import shutil
from collections import Counter
from datetime import datetime

import numpy as np

DATASET_TASKS = {
    "NYC": ["CROWDIN", "CROWDOUT", "TAXIDROP", "TAXIPICK"],
    "CHI": ["RISK", "TAXIPICK", "TAXIDROP"],
    "SIP": ["FLOW", "SPEED"],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Regenerate windowed npz data with a new output horizon without touching originals."
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--input_len", type=int, default=12)
    parser.add_argument("--output_len", type=int, required=True)
    parser.add_argument("--src_root", type=str, default="data")
    parser.add_argument("--dst_root", type=str, required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--compress", dest="compress", action="store_true")
    parser.add_argument("--no_compress", dest="compress", action="store_false")
    parser.set_defaults(compress=True)
    return parser.parse_args()


def _timestamp_key(ts_row):
    return (
        int(ts_row[0]) * 10000000000
        + int(ts_row[1]) * 100000000
        + int(ts_row[2]) * 1000000
        + int(ts_row[3]) * 10000
        + int(ts_row[4]) * 100
        + int(ts_row[5])
    )


def _extract_time_keys(x):
    ts = x[:, -1, 0, 5:11]
    return np.array([_timestamp_key(row) for row in ts], dtype=np.int64)


def _summarize_step(x):
    ts = x[:, -1, 0, 5:11]
    times = [
        datetime(int(t[0]), int(t[1]), int(t[2]), int(t[3]), int(t[4]), int(t[5]))
        for t in ts
    ]
    if len(times) < 2:
        return None
    deltas = [times[i] - times[i - 1] for i in range(1, len(times))]
    return Counter(deltas).most_common(1)[0][0]


def build_windows(series, input_len, output_len, stride):
    total = series.shape[0]
    num = (total - input_len - output_len) // stride + 1
    if num <= 0:
        raise ValueError(
            f"Not enough timesteps: total={total}, input_len={input_len}, output_len={output_len}, stride={stride}"
        )
    x_out = np.empty((num, input_len) + series.shape[1:], dtype=series.dtype)
    y_out = np.empty((num, output_len) + series.shape[1:], dtype=series.dtype)
    for i in range(num):
        start = i * stride
        x_out[i] = series[start : start + input_len]
        y_out[i] = series[start + input_len : start + input_len + output_len]
    return x_out, y_out


def process_split(src_npz, input_len, output_len, stride):
    data = np.load(src_npz)
    x = data["x"]
    if x.shape[1] < input_len:
        raise ValueError(f"input_len={input_len} exceeds available history={x.shape[1]} in {src_npz}")

    series = x[:, -1, :, :]
    keys = _extract_time_keys(x)
    order = np.argsort(keys)
    if not np.all(order == np.arange(len(order))):
        series = series[order]

    x_out, y_out = build_windows(series, input_len, output_len, stride)
    return x_out, y_out


def process_task(src_dir, dst_dir, input_len, output_len, stride, overwrite, dry_run, compress):
    os.makedirs(dst_dir, exist_ok=True)
    prompt_path = os.path.join(src_dir, "prompt.pth")
    if os.path.exists(prompt_path):
        dst_prompt = os.path.join(dst_dir, "prompt.pth")
        if not os.path.exists(dst_prompt) or overwrite:
            shutil.copy2(prompt_path, dst_prompt)

    for split in ("train", "val", "test"):
        src_npz = os.path.join(src_dir, f"{split}.npz")
        dst_npz = os.path.join(dst_dir, f"{split}.npz")
        if not os.path.exists(src_npz):
            raise FileNotFoundError(src_npz)
        if os.path.exists(dst_npz) and not overwrite:
            raise FileExistsError(f"{dst_npz} exists; use --overwrite to replace")

        x_out, y_out = process_split(src_npz, input_len, output_len, stride)
        print(f"{src_npz} -> x {x_out.shape}, y {y_out.shape}")
        if not dry_run:
            save_fn = np.savez_compressed if compress else np.savez
            save_fn(dst_npz, x=x_out, y=y_out)


def main():
    args = parse_args()
    dataset = args.dataset
    tasks = args.tasks
    if tasks is None:
        if dataset in DATASET_TASKS:
            tasks = DATASET_TASKS[dataset]
        else:
            raise ValueError(f"Unknown dataset '{dataset}'. Use --tasks to specify subfolders.")

    for task in tasks:
        src_dir = os.path.join(args.src_root, dataset, task)
        dst_dir = os.path.join(args.dst_root, dataset, task)
        if not os.path.exists(src_dir):
            raise FileNotFoundError(src_dir)
        step = _summarize_step(np.load(os.path.join(src_dir, "train.npz"))["x"])
        if step is not None:
            print(f"{dataset}/{task}: most common step = {step}")
        process_task(
            src_dir,
            dst_dir,
            args.input_len,
            args.output_len,
            args.stride,
            args.overwrite,
            args.dry_run,
            args.compress,
        )


if __name__ == "__main__":
    main()
