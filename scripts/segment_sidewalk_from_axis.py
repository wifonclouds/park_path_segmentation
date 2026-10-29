#!/usr/bin/env python3
"""Estimate sidewalk edges along 3D DXF centerlines using local LAS geometry and RGB."""
import argparse
import csv
from pathlib import Path

import ezdxf
import laspy
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter1d
from scipy.spatial import cKDTree


def load_axes(path):
    doc = ezdxf.readfile(str(path))
    axes = []
    for ent in doc.modelspace():
        if ent.dxftype() == "POLYLINE":
            pts = np.array([[v.dxf.location.x, v.dxf.location.y, v.dxf.location.z]
                            for v in ent.vertices], dtype=float)
            if len(pts) >= 2:
                axes.append((ent.dxf.layer, pts))
        elif ent.dxftype() == "LWPOLYLINE":
            pts2 = np.array([p[:2] for p in ent.get_points()], dtype=float)
            if len(pts2) >= 2:
                pts = np.column_stack((pts2, np.zeros(len(pts2))))
                axes.append((ent.dxf.layer, pts))
    if not axes:
        raise ValueError("Nie znaleziono polilinii w DXF osi.")
    return axes


def sample_axis(points, spacing):
    samples = []
    for a, b in zip(points[:-1], points[1:]):
        d = b - a
        length = np.linalg.norm(d[:2])
        if length < 1e-8:
            continue
        tangent = d[:2] / length
        n = max(1, int(np.ceil(length / spacing)))
        for k in range(n):
            samples.append((a + d * (k / n), tangent))
    d = points[-1] - points[-2]
    length = np.linalg.norm(d[:2])
    if length > 1e-8:
        samples.append((points[-1], d[:2] / length))
    return samples


def edge_at_transect(lat, z, rgb, side, args):
    signed = lat if side == "left" else -lat
    best_score, best_offset = -1.0, np.nan
    for edge in np.arange(args.min_offset, args.max_offset + args.bin_size / 2, args.bin_size):
        inside = (signed >= edge - args.window) & (signed < edge - args.bin_size / 2)
        outside = (signed > edge + args.bin_size / 2) & (signed <= edge + args.window)
        ni, no = int(inside.sum()), int(outside.sum())
        if ni < args.min_points or no < args.min_points:
            continue
        score = 0.0
        if rgb is not None:
            ci = np.median(rgb[inside], axis=0)
            co = np.median(rgb[outside], axis=0)
            score += args.rgb_weight * min(1.0, np.linalg.norm(ci - co) / args.rgb_scale)
        dz = abs(float(np.median(z[inside]) - np.median(z[outside])))
        score += args.height_weight * min(1.0, dz / args.height_scale)
        di = ni / args.window
        do = no / args.window
        score += args.density_weight * min(1.0, abs(np.log((di + 1) / (do + 1))) / 2)
        if score > best_score:
            best_score, best_offset = score, edge
    if best_score < args.min_score:
        return np.nan, max(0.0, best_score)
    return best_offset, min(1.0, best_score)


def smooth_offsets(values, window):
    x = np.asarray(values, float)
    good = np.isfinite(x)
    if good.sum() < 3:
        return x
    ids = np.arange(len(x))
    filled = x.copy()
    filled[~good] = np.interp(ids[~good], ids[good], x[good])
    size = max(3, int(window) | 1)
    filtered = gaussian_filter1d(median_filter(filled, size=size, mode="nearest"),
                                 sigma=max(0.5, size / 5), mode="nearest")
    filtered[~good] = np.nan
    return filtered


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("las", type=Path)
    ap.add_argument("axis_dxf", type=Path)
    ap.add_argument("--out-dxf", type=Path, default=Path("output/sidewalk_boundaries_auto.dxf"))
    ap.add_argument("--csv", type=Path, default=Path("output/sidewalk_edges_auto.csv"))
    ap.add_argument("--spacing", type=float, default=0.25)
    ap.add_argument("--min-offset", type=float, default=0.45)
    ap.add_argument("--max-offset", type=float, default=2.25)
    ap.add_argument("--bin-size", type=float, default=0.10)
    ap.add_argument("--window", type=float, default=0.25)
    ap.add_argument("--min-points", type=int, default=5)
    ap.add_argument("--rgb-weight", type=float, default=1.0)
    ap.add_argument("--height-weight", type=float, default=0.8)
    ap.add_argument("--density-weight", type=float, default=0.25)
    ap.add_argument("--rgb-scale", type=float, default=0.18)
    ap.add_argument("--height-scale", type=float, default=0.08)
    ap.add_argument("--min-score", type=float, default=0.28)
    ap.add_argument("--smooth-window", type=int, default=9)
    args = ap.parse_args()
    if args.spacing <= 0 or args.bin_size <= 0 or args.max_offset <= args.min_offset:
        ap.error("Nieprawidłowe wartości spacing/offset.")

    args.out_dxf.parent.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    print(f"Wczytuję chmurę: {args.las}", flush=True)
    cloud = laspy.read(args.las)
    xy = np.column_stack((np.asarray(cloud.x), np.asarray(cloud.y)))
    z_all = np.asarray(cloud.z, dtype=float)
    rgb_all = None
    if all(hasattr(cloud, k) for k in ("red", "green", "blue")):
        rgb_all = np.column_stack((cloud.red, cloud.green, cloud.blue)).astype(float)
        maxv = max(float(np.max(rgb_all)), 1.0)
        rgb_all /= maxv
    tree = cKDTree(xy)
    axes = load_axes(args.axis_dxf)
    out_doc = ezdxf.new("R2010")
    msp = out_doc.modelspace()
    rows = []

    for line_no, (layer, axis) in enumerate(axes, 1):
        samples = sample_axis(axis, args.spacing)
        left_vals, right_vals = [], []
        left_conf, right_conf = [], []
        for station, (p, tangent) in enumerate(samples):
            normal = np.array([-tangent[1], tangent[0]])
            idx = np.asarray(tree.query_ball_point(p[:2], args.max_offset + args.window), dtype=int)
            if len(idx):
                delta = xy[idx] - p[:2]
                along = delta @ tangent
                lat = delta @ normal
                keep = np.abs(along) <= args.window
                idx, lat = idx[keep], lat[keep]
            if len(idx) < args.min_points * 2:
                lo, lc, ro, rc = np.nan, 0.0, np.nan, 0.0
            else:
                rgb = rgb_all[idx] if rgb_all is not None else None
                lo, lc = edge_at_transect(lat, z_all[idx], rgb, "left", args)
                ro, rc = edge_at_transect(lat, z_all[idx], rgb, "right", args)
            left_vals.append(lo); left_conf.append(lc)
            right_vals.append(ro); right_conf.append(rc)
            rows.append({"polyline": line_no, "layer": layer, "station": station,
                         "x_axis": p[0], "y_axis": p[1], "z_axis": p[2],
                         "left_offset_m": lo, "left_confidence": lc,
                         "right_offset_m": ro, "right_confidence": rc,
                         "points_in_window": len(idx)})

        left_vals = smooth_offsets(left_vals, args.smooth_window)
        right_vals = smooth_offsets(right_vals, args.smooth_window)
        left_xyz, right_xyz = [], []
        for (p, tangent), lo, ro in zip(samples, left_vals, right_vals):
            normal = np.array([-tangent[1], tangent[0]])
            left_xyz.append([p[0] + normal[0] * lo, p[1] + normal[1] * lo, p[2]]
                            if np.isfinite(lo) else [np.nan, np.nan, np.nan])
            right_xyz.append([p[0] - normal[0] * ro, p[1] - normal[1] * ro, p[2]]
                             if np.isfinite(ro) else [np.nan, np.nan, np.nan])
        for layer_name, coords, color in (("SIDEWALK_LEFT", left_xyz, 6),  # magenta, distinct from green axis
                                          ("SIDEWALK_RIGHT", right_xyz, 1)):  # red
            arr = np.asarray(coords)
            valid = np.all(np.isfinite(arr), axis=1)
            ids = np.where(valid)[0]
            if len(ids) < 2:
                continue
            for group in np.split(ids, np.where(np.diff(ids) > 1)[0] + 1):
                if len(group) >= 2:
                    msp.add_polyline3d([tuple(v) for v in arr[group]],
                                       dxfattribs={"layer": layer_name, "color": color})

    out_doc.saveas(args.out_dxf)
    with args.csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["polyline"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Zapisano DXF: {args.out_dxf}")
    print(f"Zapisano CSV: {args.csv}")
    print("Wynik jest heurystyczny; sprawdź granice nałożone na chmurę punktów.")


if __name__ == "__main__":
    main()
