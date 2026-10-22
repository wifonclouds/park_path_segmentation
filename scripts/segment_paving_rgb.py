#!/usr/bin/env python3
"""Inspect RGB values in a LAS/LAZ point cloud.

This is an exploratory baseline, not a final paving classifier. It prints
color statistics and writes a random RGB sample to CSV for plotting/inspection.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import laspy
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect RGB values in a terrain LAS/LAZ point cloud."
    )
    parser.add_argument("input", type=Path, help="Input LAS or LAZ file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for generated diagnostics (default: output)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100_000,
        help="Maximum number of points to sample for the CSV (default: 100000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input.is_file():
        print(f"ERROR: input file does not exist: {args.input}", file=sys.stderr)
        return 2
    if args.sample_size < 1:
        print("ERROR: --sample-size must be at least 1", file=sys.stderr)
        return 2

    try:
        cloud = laspy.read(args.input)
    except Exception as exc:
        print(f"ERROR: could not read point cloud: {exc}", file=sys.stderr)
        return 2

    dimensions = set(cloud.point_format.dimension_names)
    rgb_names = {"red", "green", "blue"}
    if not rgb_names.issubset(dimensions):
        print(
            "ERROR: this point cloud has no complete RGB dimensions. "
            f"Available dimensions: {', '.join(sorted(dimensions))}",
            file=sys.stderr,
        )
        return 2

    rgb_raw = np.column_stack((cloud.red, cloud.green, cloud.blue))
    if rgb_raw.size == 0:
        print("ERROR: point cloud contains no points.", file=sys.stderr)
        return 2

    # LAS RGB values are commonly stored as 16-bit values (0..65535), although
    # some datasets use 8-bit-like values. Keep raw values and report the range.
    rgb = rgb_raw.astype(np.float64)
    mins = rgb.min(axis=0)
    maxs = rgb.max(axis=0)
    means = rgb.mean(axis=0)
    medians = np.median(rgb, axis=0)

    print(f"Input: {args.input}")
    print(f"Points: {len(cloud.points):,}")
    print("RGB channel statistics (raw LAS values):")
    for i, name in enumerate(("red", "green", "blue")):
        print(
            f"  {name:>5}: min={mins[i]:.0f}, max={maxs[i]:.0f}, "
            f"mean={means[i]:.1f}, median={medians[i]:.1f}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    count = min(args.sample_size, len(rgb_raw))
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(rgb_raw), size=count, replace=False)

    sample_path = args.output_dir / f"{args.input.stem}_rgb_sample.csv"
    try:
        with sample_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("point_index", "red", "green", "blue"))
            for idx in indices:
                r, g, b = rgb_raw[idx]
                writer.writerow((int(idx), int(r), int(g), int(b)))
    except OSError as exc:
        print(f"ERROR: could not write sample CSV: {exc}", file=sys.stderr)
        return 2

    print(f"Saved {count:,} sampled RGB records to: {sample_path}")
    print(
        "\nNote: this script only inspects colors. "
        "It does not classify or remove points yet."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
