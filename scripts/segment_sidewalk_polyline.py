#!/usr/bin/env python3
"""Segment a sidewalk from a LAS point cloud using a 3D DXF polyline.

Default inputs: input/GP.las and poly.dxf
Outputs: output/sidewalk_segmented.las and output/remaining_points.las

Optional variable-width CSV columns:
    distance_m,half_width_m
Distances are cumulative 3D metres from the start of the polyline.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import ezdxf
import laspy
import numpy as np
from scipy.spatial import cKDTree


def read_polyline(path: Path) -> np.ndarray:
    doc = ezdxf.readfile(path)
    candidates = []
    for ent in doc.modelspace():
        kind = ent.dxftype()
        if kind == "POLYLINE":
            xyz = np.asarray([[v.dxf.location.x, v.dxf.location.y, v.dxf.location.z]
                              for v in ent.vertices], dtype=np.float64)
        elif kind == "LWPOLYLINE":
            z = float(ent.dxf.elevation.z)
            xyz = np.asarray([[p[0], p[1], z] for p in ent.get_points("xy")],
                             dtype=np.float64)
        else:
            continue
        if len(xyz) >= 2:
            keep = np.ones(len(xyz), dtype=bool)
            keep[1:] = np.linalg.norm(np.diff(xyz, axis=0), axis=1) > 1e-8
            xyz = xyz[keep]
            if len(xyz) >= 2:
                candidates.append(xyz)
    if not candidates:
        raise ValueError(f"No usable POLYLINE/LWPOLYLINE found in {path}")
    candidates.sort(key=len, reverse=True)
    if len(candidates) > 1:
        print(f"WARNING: found {len(candidates)} polylines; using the one with most vertices.")
    return candidates[0]


def read_width_profile(path: Path | None, total_length: float, default_half: float):
    if path is None:
        return np.array([0.0, total_length]), np.array([default_half, default_half])
    stations, widths = [], []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"distance_m", "half_width_m"}.issubset(
                {h.strip().lower() for h in reader.fieldnames}):
            raise ValueError("CSV needs columns distance_m,half_width_m")
        names = {h.strip().lower(): h for h in reader.fieldnames}
        for row in reader:
            stations.append(float(row[names["distance_m"]]))
            widths.append(float(row[names["half_width_m"]]))
    s, w = np.asarray(stations), np.asarray(widths)
    if len(s) < 2 or np.any(w <= 0):
        raise ValueError("Width profile needs at least two rows and positive widths.")
    order = np.argsort(s)
    s, w = s[order], w[order]
    if np.any(np.diff(s) <= 0):
        raise ValueError("distance_m values must be unique.")
    if s[0] > 0:
        s, w = np.insert(s, 0, 0.0), np.insert(w, 0, w[0])
    if s[-1] < total_length:
        s, w = np.append(s, total_length), np.append(w, w[-1])
    return s, w


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=Path("input/GP.las"))
    ap.add_argument("--dxf", type=Path, default=Path("poly.dxf"))
    ap.add_argument("--output-dir", type=Path, default=Path("output"))
    ap.add_argument("--half-width", type=float, default=1.0,
                    help="Default half-width in metres.")
    ap.add_argument("--vertical-tolerance", type=float, default=0.10,
                    help="Allowed vertical difference from polyline, in metres.")
    ap.add_argument("--spacing", type=float, default=0.25,
                    help="Sampling interval along the polyline, in metres.")
    ap.add_argument("--chunk-size", type=int, default=250_000)
    ap.add_argument("--width-profile", type=Path, default=None,
                    help="CSV with distance_m,half_width_m for variable width.")
    args = ap.parse_args()

    if args.half_width <= 0 or args.spacing <= 0 or args.vertical_tolerance < 0:
        ap.error("half-width and spacing must be > 0; vertical tolerance must be >= 0.")
    for p in (args.input, args.dxf):
        if not p.exists():
            ap.error(f"File not found: {p}")

    vertices = read_polyline(args.dxf)
    a, b = vertices[:-1], vertices[1:]
    v = b - a
    len2 = np.einsum("ij,ij->i", v, v)
    lengths = np.sqrt(len2)
    valid = lengths > 1e-8
    a, v, len2, lengths = a[valid], v[valid], len2[valid], lengths[valid]
    if not len(a):
        ap.error("DXF polyline has no non-zero-length segments.")
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    total_length = cumulative[-1]

    # Sample every segment; each sample remembers its original segment.
    sample_xyz, sample_seg = [], []
    for i, length in enumerate(lengths):
        n = max(1, int(np.ceil(length / args.spacing)))
        t = np.arange(n, dtype=float) / n
        if i == len(lengths) - 1:
            t = np.r_[t, 1.0]
        sample_xyz.append(a[i] + t[:, None] * v[i])
        sample_seg.append(np.full(len(t), i, dtype=np.int32))
    samples = np.vstack(sample_xyz)
    sample_seg = np.concatenate(sample_seg)
    tree = cKDTree(samples)
    width_s, width_vals = read_width_profile(args.width_profile, total_length, args.half_width)

    print(f"Polyline vertices: {len(vertices)}; length: {total_length:.2f} m")
    print(f"Samples: {len(samples):,}; vertical tolerance: ±{args.vertical_tolerance:.3f} m")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sidewalk_path = args.output_dir / "sidewalk_segmented.las"
    rest_path = args.output_dir / "remaining_points.las"
    total = selected = 0

    with laspy.open(args.input) as reader:
        header = reader.header.copy()
        with laspy.open(sidewalk_path, mode="w", header=header) as out_side, \
             laspy.open(rest_path, mode="w", header=header) as out_rest:
            for block_no, points in enumerate(reader.chunk_iterator(args.chunk_size), 1):
                xyz = np.column_stack((points.x, points.y, points.z)).astype(np.float64)
                k = min(8, len(samples))
                _, idx = tree.query(xyz, k=k, workers=-1)
                if k == 1:
                    idx = idx[:, None]
                seg_ids = sample_seg[idx]
                npts = len(xyz)
                best_d2 = np.full(npts, np.inf)
                best_s = np.zeros(npts)
                best_z = np.zeros(npts)

                # Exact projection to each candidate segment; pick nearest 3D projection.
                for sid in np.unique(seg_ids):
                    rows = np.any(seg_ids == sid, axis=1)
                    ids = np.flatnonzero(rows)
                    p = xyz[rows]
                    t = np.einsum("ij,j->i", p - a[sid], v[sid]) / len2[sid]
                    t = np.clip(t, 0.0, 1.0)
                    q = a[sid] + t[:, None] * v[sid]
                    d = p - q
                    d2 = np.einsum("ij,ij->i", d, d)
                    improve = d2 < best_d2[rows]
                    chosen = ids[improve]
                    best_d2[chosen] = d2[improve]
                    best_s[chosen] = cumulative[sid] + t[improve] * lengths[sid]
                    best_z[chosen] = q[improve, 2]

                dz = np.abs(xyz[:, 2] - best_z)
                # Remove vertical component from 3D distance to obtain XY lateral distance.
                lateral = np.sqrt(np.maximum(best_d2 - (xyz[:, 2] - best_z) ** 2, 0.0))
                half_width = np.interp(best_s, width_s, width_vals)
                mask = (lateral <= half_width) & (dz <= args.vertical_tolerance)
                out_side.write_points(points[mask])
                out_rest.write_points(points[~mask])
                total += npts
                selected += int(mask.sum())
                print(f"\rBlock {block_no}: {total:,} processed; {selected:,} selected",
                      end="", flush=True)

    print(f"\nDone. Selected {selected:,}/{total:,} points "
          f"({100*selected/max(total,1):.2f}%).")
    print(f"Sidewalk: {sidewalk_path}")
    print(f"Remaining: {rest_path}")


if __name__ == "__main__":
    main()
