#!/usr/bin/env python3
"""Segment sidewalk points near every 3D line in a DXF using GDAL/OGR.

Defaults from the repository root:
  input/ground.las, input/path.dxf
  output/sidewalk_segmented.las, output/remaining_points.las

Optional width-profile CSV columns:
  polyline_id,distance_m,half_width_m
polyline_id is 1-based in the order OGR returns line features. Distance is
cumulative 3D metres from that polyline's start. Lines without profile rows
use --half-width.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import laspy
import numpy as np
from scipy.spatial import cKDTree


def _append_line(coords, lines):
    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return
    if arr.shape[1] == 2:
        arr = np.column_stack((arr, np.zeros(len(arr), dtype=np.float64)))
    else:
        arr = arr[:, :3]
    keep = np.ones(len(arr), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(arr, axis=0), axis=1) > 1e-8
    arr = arr[keep]
    if len(arr) >= 2:
        lines.append(arr)


def _collect_geometry(geometry, lines):
    if not geometry:
        return
    kind = geometry.get("type")
    coords = geometry.get("coordinates", [])
    if kind == "LineString":
        _append_line(coords, lines)
    elif kind == "MultiLineString":
        for line in coords:
            _append_line(line, lines)
    elif kind == "GeometryCollection":
        for child in geometry.get("geometries", []):
            _collect_geometry(child, lines)


def read_dxf_lines(dxf_path: Path):
    ogr2ogr = shutil.which("ogr2ogr")
    if not ogr2ogr:
        raise RuntimeError("Nie znaleziono ogr2ogr. Zainstaluj GDAL: sudo apt install gdal-bin")
    with tempfile.TemporaryDirectory(prefix="sidewalk_dxf_") as temp_dir:
        geojson_path = Path(temp_dir) / "paths.geojson"
        result = subprocess.run(
            [ogr2ogr, "-f", "GeoJSON", str(geojson_path), str(dxf_path)],
            capture_output=True, text=True
        )
        if result.returncode != 0 or not geojson_path.exists():
            raise RuntimeError("GDAL nie odczytał DXF:\n" + (result.stderr.strip() or result.stdout.strip()))
        data = json.loads(geojson_path.read_text(encoding="utf-8"))
    lines = []
    for feature in data.get("features", []):
        _collect_geometry(feature.get("geometry"), lines)
    if not lines:
        raise ValueError(f"Nie znaleziono geometrii liniowych w {dxf_path}")
    return lines


def load_width_profiles(path, line_lengths):
    profiles = {}
    if path is None:
        return profiles
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        names = {name.strip().lower(): name for name in (reader.fieldnames or [])}
        required = {"polyline_id", "distance_m", "half_width_m"}
        if not required.issubset(names):
            raise ValueError("CSV musi mieć kolumny polyline_id,distance_m,half_width_m")
        rows = {}
        for row in reader:
            pid = int(row[names["polyline_id"]])
            distance = float(row[names["distance_m"]])
            width = float(row[names["half_width_m"]])
            if not 1 <= pid <= len(line_lengths):
                raise ValueError(f"polyline_id={pid} poza zakresem 1..{len(line_lengths)}")
            if distance < 0 or width <= 0:
                raise ValueError("distance_m musi być >= 0, half_width_m musi być > 0")
            rows.setdefault(pid - 1, []).append((distance, width))
    for pid, values in rows.items():
        values.sort()
        stations = np.asarray([v[0] for v in values], dtype=np.float64)
        widths = np.asarray([v[1] for v in values], dtype=np.float64)
        if np.any(np.diff(stations) <= 0):
            raise ValueError(f"Powtarzające się distance_m dla polyline_id={pid + 1}")
        profiles[pid] = (stations, widths)
    return profiles


def build_segments(lines, spacing):
    seg_a, seg_v, seg_lengths, seg_line, seg_start_s = [], [], [], [], []
    sample_xy, sample_seg, line_lengths = [], [], []
    for line_id, xyz in enumerate(lines):
        vec = xyz[1:] - xyz[:-1]
        lengths = np.linalg.norm(vec, axis=1)
        valid = lengths > 1e-8
        a_line, vec, lengths = xyz[:-1][valid], vec[valid], lengths[valid]
        if not len(lengths):
            line_lengths.append(0.0)
            continue
        cumulative = np.r_[0.0, np.cumsum(lengths)]
        line_lengths.append(float(cumulative[-1]))
        for local_idx, (a, v, length) in enumerate(zip(a_line, vec, lengths)):
            sid = len(seg_a)
            seg_a.append(a)
            seg_v.append(v)
            seg_lengths.append(float(length))
            seg_line.append(line_id)
            seg_start_s.append(float(cumulative[local_idx]))
            n = max(1, int(np.ceil(length / spacing)))
            t = np.linspace(0.0, 1.0, n + 1)
            pts = a[None, :] + t[:, None] * v[None, :]
            sample_xy.append(pts[:, :2])
            sample_seg.append(np.full(len(pts), sid, dtype=np.int32))
    if not seg_a:
        raise ValueError("DXF nie zawiera segmentów o dodatniej długości")
    return (np.asarray(seg_a), np.asarray(seg_v), np.asarray(seg_lengths),
            np.asarray(seg_line, dtype=np.int32), np.asarray(seg_start_s),
            np.vstack(sample_xy), np.concatenate(sample_seg), line_lengths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("input/ground.las"))
    parser.add_argument("--dxf", type=Path, default=Path("input/path.dxf"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--half-width", type=float, default=1.1)
    parser.add_argument("--vertical-tolerance", type=float, default=0.10)
    parser.add_argument("--spacing", type=float, default=0.25)
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--width-profile", type=Path, default=None,
                        help="CSV: polyline_id,distance_m,half_width_m")
    parser.add_argument("--neighbors", type=int, default=12,
                        help="Nearest XY samples checked per point")
    args = parser.parse_args()
    if args.half_width <= 0 or args.spacing <= 0 or args.vertical_tolerance < 0:
        parser.error("--half-width/--spacing must be > 0; vertical tolerance must be >= 0")
    if args.neighbors < 1:
        parser.error("--neighbors must be >= 1")
    for path in (args.input, args.dxf):
        if not path.is_file():
            parser.error(f"Nie znaleziono pliku: {path}")
    if args.width_profile and not args.width_profile.is_file():
        parser.error(f"Nie znaleziono profilu: {args.width_profile}")

    lines = read_dxf_lines(args.dxf)
    (seg_a, seg_v, seg_lengths, seg_line, seg_start_s,
     sample_xy, sample_seg, line_lengths) = build_segments(lines, args.spacing)
    profiles = load_width_profiles(args.width_profile, line_lengths)
    tree = cKDTree(sample_xy)
    k = min(args.neighbors, len(sample_xy))

    print(f"Polilinie odczytane z DXF: {len(lines)}")
    print(f"Segmenty: {len(seg_a):,}; próbki co maks. {args.spacing:.2f} m: {len(sample_xy):,}")
    print(f"Połowa szerokości: {args.half_width:.2f} m; tolerancja Z: ±{args.vertical_tolerance:.2f} m")
    if profiles:
        print(f"Polilinie z profilem zmiennej szerokości: {len(profiles)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sidewalk_path = args.output_dir / "sidewalk_segmented.las"
    rest_path = args.output_dir / "remaining_points.las"
    total = selected_total = 0
    with laspy.open(args.input) as reader:
        header = reader.header.copy()
        with laspy.open(sidewalk_path, mode="w", header=header) as out_side, \
             laspy.open(rest_path, mode="w", header=header) as out_rest:
            for block_no, points in enumerate(reader.chunk_iterator(args.chunk_size), 1):
                xyz = np.column_stack((points.x, points.y, points.z)).astype(np.float64)
                _, sample_ids = tree.query(xyz[:, :2], k=k, workers=-1)
                if k == 1:
                    sample_ids = sample_ids[:, None]
                candidate_seg = sample_seg[sample_ids]
                n_points = len(xyz)
                best_d2 = np.full(n_points, np.inf)
                best_z = np.zeros(n_points)
                best_s = np.zeros(n_points)
                best_line = np.zeros(n_points, dtype=np.int32)

                # Exact XY projection to each candidate line segment.
                for sid in np.unique(candidate_seg):
                    rows = np.any(candidate_seg == sid, axis=1)
                    ids = np.flatnonzero(rows)
                    p = xyz[rows]
                    a, v = seg_a[sid], seg_v[sid]
                    denom = max(float(np.dot(v[:2], v[:2])), 1e-12)
                    t = np.einsum("ij,j->i", p[:, :2] - a[:2], v[:2]) / denom
                    t = np.clip(t, 0.0, 1.0)
                    q = a[None, :] + t[:, None] * v[None, :]
                    dxy = p[:, :2] - q[:, :2]
                    d2 = np.einsum("ij,ij->i", dxy, dxy)
                    improve = d2 < best_d2[rows]
                    chosen = ids[improve]
                    best_d2[chosen] = d2[improve]
                    best_z[chosen] = q[improve, 2]
                    best_s[chosen] = seg_start_s[sid] + t[improve] * seg_lengths[sid]
                    best_line[chosen] = seg_line[sid]

                half_width = np.full(n_points, args.half_width, dtype=np.float64)
                for line_id, (stations, widths) in profiles.items():
                    rows = best_line == line_id
                    if np.any(rows):
                        half_width[rows] = np.interp(
                            best_s[rows], stations, widths,
                            left=widths[0], right=widths[-1]
                        )
                lateral = np.sqrt(best_d2)
                dz = np.abs(xyz[:, 2] - best_z)
                mask = (lateral <= half_width) & (dz <= args.vertical_tolerance)
                out_side.write_points(points[mask])
                out_rest.write_points(points[~mask])
                total += n_points
                selected_total += int(mask.sum())
                print(f"\rBlok {block_no}: {total:,} przetworzono; {selected_total:,} chodnik",
                      end="", flush=True)

    print(f"\nGotowe: {selected_total:,}/{total:,} punktów "
          f"({100 * selected_total / max(total, 1):.2f}%) przypisano do chodników.")
    print(f"Chodniki: {sidewalk_path}")
    print(f"Pozostałe: {rest_path}")


if __name__ == "__main__":
    main()
