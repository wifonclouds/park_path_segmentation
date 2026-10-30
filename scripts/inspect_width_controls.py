#!/usr/bin/env python3
"""Diagnose manual width-control sections against sidewalk axis polylines."""
import argparse
import csv
from pathlib import Path

import ezdxf
import numpy as np


def get_polyline_points(entity):
    kind = entity.dxftype()
    if kind == "POLYLINE":
        return np.array([[v.dxf.location.x, v.dxf.location.y, v.dxf.location.z]
                         for v in entity.vertices], dtype=float)
    if kind == "LWPOLYLINE":
        xy = np.array([p[:2] for p in entity.get_points()], dtype=float)
        return np.column_stack((xy, np.zeros(len(xy))))
    return None


def load_axes(path):
    doc = ezdxf.readfile(str(path))
    axes = []
    for entity in doc.modelspace():
        pts = get_polyline_points(entity)
        if pts is not None and len(pts) >= 2:
            axes.append({"layer": entity.dxf.layer, "points": pts})
    if not axes:
        raise ValueError(f"No axis polylines found in {path}")
    return axes


def load_controls(path):
    doc = ezdxf.readfile(str(path))
    controls = []
    for entity in doc.modelspace():
        kind = entity.dxftype()
        if kind == "LINE":
            a = np.array(entity.dxf.start, dtype=float)
            b = np.array(entity.dxf.end, dtype=float)
        elif kind == "POLYLINE":
            pts = np.array([[v.dxf.location.x, v.dxf.location.y, v.dxf.location.z]
                            for v in entity.vertices], dtype=float)
            if len(pts) != 2:
                continue
            a, b = pts
        elif kind == "LWPOLYLINE":
            pts = list(entity.get_points())
            if len(pts) != 2:
                continue
            a = np.array([pts[0][0], pts[0][1], 0.0])
            b = np.array([pts[1][0], pts[1][1], 0.0])
        else:
            continue
        if np.linalg.norm((b - a)[:2]) > 1e-6:
            controls.append({"layer": entity.dxf.layer, "a": a, "b": b})
    return controls


def project_to_axis(point_xy, axis_points):
    best = None
    station_base = 0.0
    for p0, p1 in zip(axis_points[:-1], axis_points[1:]):
        v = p1[:2] - p0[:2]
        length = float(np.linalg.norm(v))
        if length < 1e-9:
            continue
        t = float(np.clip(np.dot(point_xy - p0[:2], v) / (length * length), 0.0, 1.0))
        projected = p0[:2] + t * v
        distance = float(np.linalg.norm(point_xy - projected))
        if best is None or distance < best["distance"]:
            tangent = v / length
            best = {
                "distance": distance,
                "projected": projected,
                "tangent": tangent,
                "station": station_base + t * length,
            }
        station_base += length
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axes", type=Path, default=Path("input/path_repaired.dxf"))
    parser.add_argument("--controls", type=Path, default=Path("input/width_control_repaired.dxf"))
    parser.add_argument("--csv", type=Path, default=Path("output/width_control_report.csv"))
    parser.add_argument("--max-axis-distance", type=float, default=5.0)
    parser.add_argument("--min-perpendicularity", type=float, default=0.70,
                        help="Minimum absolute dot product with axis normal (1.0 = perfectly perpendicular).")
    args = parser.parse_args()

    axes = load_axes(args.axes)
    controls = load_controls(args.controls)
    if not controls:
        raise ValueError("No 2-vertex LINE/POLYLINE controls found.")

    rows = []
    for index, control in enumerate(controls, 1):
        midpoint = (control["a"][:2] + control["b"][:2]) / 2.0
        candidates = []
        for axis_index, axis in enumerate(axes, 1):
            projection = project_to_axis(midpoint, axis["points"])
            if projection is not None:
                candidates.append((projection["distance"], axis_index, axis, projection))
        if not candidates:
            continue
        _, axis_index, axis, projection = min(candidates, key=lambda item: item[0])

        normal = np.array([-projection["tangent"][1], projection["tangent"][0]])
        control_vec = control["b"][:2] - control["a"][:2]
        control_unit = control_vec / np.linalg.norm(control_vec)
        perpendicularity = abs(float(np.dot(control_unit, normal)))

        signed_a = float(np.dot(control["a"][:2] - projection["projected"], normal))
        signed_b = float(np.dot(control["b"][:2] - projection["projected"], normal))
        left_width = max(signed_a, signed_b, 0.0)
        right_width = max(-signed_a, -signed_b, 0.0)
        rows.append({
            "control_id": index,
            "control_layer": control["layer"],
            "axis_id": axis_index,
            "axis_layer": axis["layer"],
            "station_m": projection["station"],
            "axis_distance_m": projection["distance"],
            "perpendicularity": perpendicularity,
            "left_width_m": left_width,
            "right_width_m": right_width,
            "total_width_m": left_width + right_width,
            "accepted": projection["distance"] <= args.max_axis_distance
                        and perpendicularity >= args.min_perpendicularity,
        })

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["control_id"]
    with args.csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Axes: {len(axes)} | control sections: {len(controls)}")
    print(f"Report: {args.csv}")
    print("ID | axis | station[m] | left[m] | right[m] | axis_dist[m] | perp | accepted")
    for row in rows:
        print(f'{row["control_id"]:2d} | {row["axis_id"]:4d} | '
              f'{row["station_m"]:10.2f} | {row["left_width_m"]:7.2f} | '
              f'{row["right_width_m"]:8.2f} | {row["axis_distance_m"]:12.2f} | '
              f'{row["perpendicularity"]:4.2f} | {row["accepted"]}')


if __name__ == "__main__":
    main()
