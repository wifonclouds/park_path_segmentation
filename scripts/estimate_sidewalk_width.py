#!/usr/bin/env python3
"""Estimate sidewalk width from RGB transitions along DXF polylines.

Polyline Z and point Z are intentionally ignored: all analysis is performed in XY.
The result is a heuristic and should be visually checked in CloudCompare.
"""
import argparse
import csv
import sys
from pathlib import Path

import laspy
import numpy as np
from ezdxf import recover

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cloud", help="Input RGB LAS/LAZ point cloud")
    parser.add_argument("dxf", help="DXF containing 2D/3D POLYLINE entities")
    parser.add_argument("--csv", default="output/sidewalk_width.csv")
    parser.add_argument("--out-dxf", default="output/sidewalk_boundaries.dxf")
    parser.add_argument("--spacing", type=float, default=0.25)
    parser.add_argument("--max-width", type=float, default=2.0)
    parser.add_argument("--bin-size", type=float, default=0.05)
    parser.add_argument("--longitudinal-half", type=float, default=None)
    parser.add_argument("--core-inner", type=float, default=0.10)
    parser.add_argument("--core-outer", type=float, default=0.45)
    parser.add_argument("--min-points-bin", type=int, default=3)
    parser.add_argument("--min-transition", type=float, default=24.0)
    parser.add_argument("--min-consecutive-bins", type=int, default=2)
    parser.add_argument("--continuity-weight", type=float, default=3.0)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--width-mode", choices=("piecewise", "constant", "variable"), default="piecewise",
                        help="piecewise: stable widths with detected changes (default); constant: one width per polyline; variable: smoothed station widths")
    parser.add_argument("--width-change-threshold", type=float, default=0.25,
                        help="minimum width change in metres to consider a new segment")
    parser.add_argument("--min-width-segment", type=int, default=8,
                        help="minimum stations in each constant-width segment")
    parser.add_argument("--max-stations-per-polyline", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=1000000)
    parser.add_argument("--no-out-dxf", action="store_true")
    return parser.parse_args()


def read_polylines(filename):
    doc, auditor = recover.readfile(filename)
    if auditor.fixes:
        print("DXF recovery applied {} fixes.".format(len(auditor.fixes)))
    if auditor.errors:
        print("WARNING: {} DXF audit errors.".format(len(auditor.errors)))
    result = []
    for entity in doc.modelspace():
        kind = entity.dxftype()
        if kind == "LWPOLYLINE":
            points = [(float(p[0]), float(p[1])) for p in entity.get_points("xy")]
            closed = entity.closed
        elif kind == "POLYLINE":
            points = [(float(v.dxf.location.x), float(v.dxf.location.y))
                      for v in entity.vertices]
            closed = entity.is_closed
        else:
            continue
        if len(points) < 2:
            continue
        xy = np.asarray(points, dtype=float)
        keep = np.ones(len(xy), dtype=bool)
        keep[1:] = np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-8
        xy = xy[keep]
        if len(xy) < 2:
            continue
        if closed and np.linalg.norm(xy[-1] - xy[0]) > 1e-8:
            xy = np.vstack((xy, xy[0]))
        result.append({"id": len(result) + 1,
                       "handle": str(entity.dxf.get("handle", "no_handle")),
                       "xy": xy})
    if not result:
        raise ValueError("No usable POLYLINE or LWPOLYLINE entities found.")
    print("Found {} polylines.".format(len(result)))
    return result


def sample_line(polyline, spacing, limit):
    xy = polyline["xy"]
    segments = np.diff(xy, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    total = cumulative[-1]
    if total <= 0:
        return []
    stations = np.arange(0.0, total, spacing)
    if not len(stations) or total - stations[-1] > spacing * 0.25:
        stations = np.r_[stations, total]
    else:
        stations[-1] = total
    rows = []
    for distance in stations:
        i = min(np.searchsorted(cumulative, distance, side="right") - 1, len(lengths) - 1)
        t = np.clip((distance - cumulative[i]) / lengths[i], 0.0, 1.0)
        point = xy[i] + t * segments[i]
        tangent = segments[i] / lengths[i]
        normal = np.array([-tangent[1], tangent[0]])
        rows.append((distance, point[0], point[1], tangent[0], tangent[1],
                     normal[0], normal[1]))
    return rows[:limit] if limit > 0 else rows


def normalize_rgb(red, green, blue):
    rgb = np.column_stack((red, green, blue)).astype(np.float32)
    if rgb.size and np.nanmax(rgb) > 255:
        rgb /= 257.0
    return np.clip(rgb, 0, 255)


def read_corridor(cloud, polylines, max_width, longitudinal_half, chunk_size):
    all_xy = np.vstack([p["xy"] for p in polylines])
    margin = max_width + longitudinal_half + 1.0
    low, high = all_xy.min(axis=0) - margin, all_xy.max(axis=0) + margin
    xy_parts, rgb_parts = [], []
    with laspy.open(cloud) as reader:
        dimensions = set(reader.header.point_format.dimension_names)
        if not {"red", "green", "blue"}.issubset(dimensions):
            raise ValueError("Input LAS/LAZ must contain red, green and blue dimensions.")
        total = reader.header.point_count
        scanned = 0
        for points in reader.chunk_iterator(chunk_size):
            x, y = np.asarray(points.x), np.asarray(points.y)
            mask = ((x >= low[0]) & (x <= high[0]) &
                    (y >= low[1]) & (y <= high[1]))
            if mask.any():
                xy_parts.append(np.column_stack((x[mask], y[mask])).astype(np.float64))
                rgb_parts.append(normalize_rgb(np.asarray(points.red)[mask],
                                               np.asarray(points.green)[mask],
                                               np.asarray(points.blue)[mask]))
            scanned += len(points)
            if scanned % max(chunk_size * 5, 1) < len(points) or scanned == total:
                print("Scanned {:,}/{:,} points".format(scanned, total))
    if not xy_parts:
        raise ValueError("No points near DXF polylines; check coordinate systems.")
    xy, rgb = np.concatenate(xy_parts), np.concatenate(rgb_parts)
    print("Retained {:,} corridor points.".format(len(xy)))
    return xy, rgb


def rgb_profile(lateral, rgb, side, width, bin_size, min_points):
    edges = np.arange(0.0, width + bin_size, bin_size)
    if edges[-1] < width:
        edges = np.r_[edges, width]
    centers = (edges[:-1] + edges[1:]) / 2
    values = []
    signed = lateral * side
    for lo, hi in zip(edges[:-1], edges[1:]):
        selected = (signed >= lo) & (signed < hi)
        values.append(np.median(rgb[selected], axis=0)
                      if selected.sum() >= min_points else None)
    return centers, values


def find_edge(centers, values, core, threshold, consecutive, previous, continuity):
    """Find a sustained RGB boundary, not merely the strongest colour far from the core.

    A candidate is a transition from bins whose colour resembles the sidewalk core
    to consecutive bins whose colour differs from it. Scoring the adjacent-bin
    contrast helps avoid selecting isolated colour patches inside the pavement.
    """
    candidates = []
    if len(values) < consecutive + 1:
        return None, 0.0

    core_dist = [
        None if value is None else float(np.linalg.norm(value - core))
        for value in values
    ]

    for i in range(len(values) - consecutive):
        inside, outside = values[i], values[i + 1]
        if inside is None or outside is None:
            continue

        # The bin immediately inside the edge should still look like pavement.
        if core_dist[i] > threshold * 1.5:
            continue

        # Require a sustained change outside the edge to reject isolated RGB noise.
        run = 0
        outside_contrasts = []
        for j in range(i + 1, min(i + 1 + consecutive, len(values))):
            if values[j] is not None and core_dist[j] is not None and core_dist[j] >= threshold:
                run += 1
                outside_contrasts.append(core_dist[j])
            else:
                break
        if run < consecutive:
            continue

        transition = float(np.linalg.norm(outside - inside))
        if transition < threshold * 0.5:
            continue

        width = float((centers[i] + centers[i + 1]) / 2.0)
        outside_contrast = float(np.median(outside_contrasts))
        # Prefer a clear, sustained colour boundary. Penalize implausibly large
        # widths and sudden changes from the previous station, without forcing
        # every section to have the same width.
        score = transition + 0.35 * outside_contrast - 3.0 * width
        if previous is not None:
            score -= continuity * 5.0 * abs(width - previous)
        candidates.append((score, width, transition, outside_contrast))

    if not candidates:
        return None, 0.0

    _, width, transition, outside_contrast = max(candidates)
    confidence = float(np.clip(
        (min(transition, outside_contrast) - threshold * 0.5) /
        max(threshold * 1.5, 1.0), 0.0, 1.0
    ))
    return width, confidence


def smooth(values, window):
    if window <= 1:
        return values[:]
    window += (window % 2 == 0)
    radius, result = window // 2, []
    for i, value in enumerate(values):
        if value is None:
            result.append(None)
        else:
            neighbours = [v for v in values[max(0, i-radius):i+radius+1] if v is not None]
            result.append(float(np.median(neighbours)) if neighbours else value)
    return result


def robust_constant_width(values, confidences, min_confidence=0.25):
    """Estimate one stable offset for a polyline, rejecting local RGB outliers."""
    valid = [(float(v), float(c)) for v, c in zip(values, confidences) if v is not None]
    if not valid:
        return None

    reliable = [v for v, c in valid if c >= min_confidence]
    sample = np.asarray(reliable if len(reliable) >= 5 else [v for v, _ in valid], dtype=float)
    median = float(np.median(sample))
    mad = float(np.median(np.abs(sample - median)))
    # A 15 cm floor prevents tiny MAD values from rejecting ordinary bin noise.
    limit = max(0.15, 3.0 * 1.4826 * mad)
    inliers = sample[np.abs(sample - median) <= limit]
    if len(inliers) >= 3:
        median = float(np.median(inliers))
    return median


def piecewise_constant_widths(values, confidences, change_threshold=0.25, min_segment=8):
    """Fit stable widths along stretches while allowing genuine width changes."""
    n = len(values)
    valid_idx = np.array([i for i, value in enumerate(values) if value is not None], dtype=int)
    if not len(valid_idx):
        return [None] * n

    raw = np.asarray([float(values[i]) for i in valid_idx], dtype=float)
    conf = np.asarray([float(confidences[i]) for i in valid_idx], dtype=float)
    reliable = conf >= 0.25
    if reliable.sum() >= max(5, min_segment // 2):
        fit_values = raw.copy()
        fit_values[~reliable] = np.interp(valid_idx[~reliable], valid_idx[reliable], raw[reliable])
    else:
        fit_values = raw.copy()

    fitted = np.interp(np.arange(n), valid_idx, fit_values)
    boundaries = [0, n]

    def split(start, end, depth=0):
        if end - start < 2 * min_segment or depth >= 10:
            return
        section = fitted[start:end]
        base_median = float(np.median(section))
        base_sse = float(np.sum((section - base_median) ** 2))
        best = None
        for cut in range(start + min_segment, end - min_segment + 1):
            left = fitted[start:cut]
            right = fitted[cut:end]
            left_med, right_med = float(np.median(left)), float(np.median(right))
            if abs(left_med - right_med) < change_threshold:
                continue
            sse = float(np.sum((left - left_med) ** 2) + np.sum((right - right_med) ** 2))
            if best is None or sse < best[0]:
                best = (sse, cut)
        if best is not None and base_sse > 0 and best[0] < base_sse * 0.70:
            _, cut = best
            boundaries.append(cut)
            split(start, cut, depth + 1)
            split(cut, end, depth + 1)

    split(0, n)
    boundaries = sorted(set(boundaries))
    output = [None] * n
    first, last = int(valid_idx[0]), int(valid_idx[-1])
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        segment = fitted[start:end]
        median = float(np.median(segment))
        for i in range(start, end):
            output[i] = median
    for i in range(first):
        output[i] = None
    for i in range(last + 1, n):
        output[i] = None
    return output


def analyse(polyline, stations, xy, rgb, args, tree):
    long_half = args.longitudinal_half or args.spacing * 0.6
    left, right, lc, rc, statuses = [], [], [], [], []
    previous_left = previous_right = None
    for index, station in enumerate(stations):
        _, x, y, tx, ty, nx, ny = station
        if tree is not None:
            indices = np.asarray(tree.query_ball_point([x, y],
                                   np.hypot(args.max_width, long_half)), dtype=int)
        else:
            delta = xy - [x, y]
            indices = np.flatnonzero((np.abs(delta[:, 0]) <= args.max_width + long_half) &
                                     (np.abs(delta[:, 1]) <= args.max_width + long_half))
        if not len(indices):
            left.append(None); right.append(None); lc.append(0.0); rc.append(0.0)
            statuses.append("no_neighbour_points")
            continue
        delta = xy[indices] - [x, y]
        along = delta[:, 0] * tx + delta[:, 1] * ty
        lateral = delta[:, 0] * nx + delta[:, 1] * ny
        mask = ((np.abs(along) <= long_half) & (np.abs(lateral) <= args.max_width))
        lateral, colours = lateral[mask], rgb[indices[mask]]
        absolute = np.abs(lateral)
        core_mask = (absolute >= args.core_inner) & (absolute < args.core_outer)
        if core_mask.sum() < args.min_points_bin:
            core_mask = absolute < min(args.max_width, 0.6)
        if not core_mask.any():
            left.append(None); right.append(None); lc.append(0.0); rc.append(0.0)
            statuses.append("insufficient_center_points")
            continue
        core = np.median(colours[core_mask], axis=0)
        centers_l, values_l = rgb_profile(lateral, colours, +1, args.max_width,
                                          args.bin_size, args.min_points_bin)
        centers_r, values_r = rgb_profile(lateral, colours, -1, args.max_width,
                                          args.bin_size, args.min_points_bin)
        lw, lconf = find_edge(centers_l, values_l, core, args.min_transition,
                              args.min_consecutive_bins, previous_left, args.continuity_weight)
        rw, rconf = find_edge(centers_r, values_r, core, args.min_transition,
                              args.min_consecutive_bins, previous_right, args.continuity_weight)
        if lw is not None:
            previous_left = lw
        if rw is not None:
            previous_right = rw
        left.append(lw); right.append(rw); lc.append(lconf); rc.append(rconf)
        if lw is None and rw is None:
            status = "uncertain_both_sides"
        elif lw is None or rw is None:
            status = "uncertain_one_side"
        elif min(lconf, rconf) < 0.25:
            status = "weak_rgb_contrast"
        else:
            status = "ok"
        statuses.append(status)
        if (index + 1) % 500 == 0 or index + 1 == len(stations):
            print("  polyline {}: {}/{} sections".format(polyline["id"], index + 1, len(stations)))
    return left, right, lc, rc, statuses


def write_dxf(filename, rows_by_id):
    import ezdxf
    doc = ezdxf.new("R2013")
    modelspace = doc.modelspace()
    for pid, rows in rows_by_id.items():
        left, right = [], []
        for row in rows:
            if row["left"] is not None:
                left.append((row["x"] + row["nx"] * row["left"],
                             row["y"] + row["ny"] * row["left"], 0))
            if row["right"] is not None:
                right.append((row["x"] - row["nx"] * row["right"],
                              row["y"] - row["ny"] * row["right"], 0))
        if len(left) >= 2:
            modelspace.add_polyline3d(left, dxfattribs={"layer": "PATH_{:02d}_LEFT".format(pid)})
        if len(right) >= 2:
            modelspace.add_polyline3d(right, dxfattribs={"layer": "PATH_{:02d}_RIGHT".format(pid)})
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(filename)


def main():
    args = arguments()
    if min(args.spacing, args.max_width, args.bin_size) <= 0:
        raise ValueError("spacing, max-width and bin-size must be positive")
    if args.core_outer <= args.core_inner:
        raise ValueError("core-outer must exceed core-inner")
    polylines = read_polylines(args.dxf)
    stations_by_id = {p["id"]: sample_line(p, args.spacing, args.max_stations_per_polyline)
                      for p in polylines}
    if not any(stations_by_id.values()):
        raise ValueError("No stations generated from input polylines.")
    long_half = args.longitudinal_half or args.spacing * 0.6
    xy, rgb = read_corridor(args.cloud, polylines, args.max_width, long_half, args.chunk_size)
    tree = cKDTree(xy) if cKDTree is not None else None
    if tree is None:
        print("WARNING: scipy is not installed; neighbour search may be slow.")
    rows_by_id, csv_rows = {}, []
    for polyline in polylines:
        stations = stations_by_id[polyline["id"]]
        print("Analysing polyline {} (handle={}, stations={})".format(
            polyline["id"], polyline["handle"], len(stations)))
        left, right, lc, rc, statuses = analyse(polyline, stations, xy, rgb, args, tree)
        rows = []
        for st, lw, rw, lconf, rconf, status in zip(stations, left, right, lc, rc, statuses):
            s, x, y, tx, ty, nx, ny = st
            row = {"polyline_id": polyline["id"], "handle": polyline["handle"],
                   "station": s, "x": x, "y": y, "nx": nx, "ny": ny,
                   "left": lw, "right": rw,
                   "total": lw + rw if lw is not None and rw is not None else None,
                   "left_confidence": lconf, "right_confidence": rconf, "status": status}
            rows.append(row)
        rows_by_id[polyline["id"]] = rows
        for key, confidence_key in (("left", "left_confidence"), ("right", "right_confidence")):
            raw_values = [r[key] for r in rows]
            confidences = [r[confidence_key] for r in rows]
            if args.width_mode == "constant":
                width = robust_constant_width(raw_values, confidences)
                values = [width] * len(rows) if width is not None else raw_values
            elif args.width_mode == "piecewise":
                values = piecewise_constant_widths(
                    raw_values,
                    confidences,
                    change_threshold=args.width_change_threshold,
                    min_segment=max(2, args.min_width_segment),
                )
            else:
                values = smooth(raw_values, args.smooth_window)
            for row, value in zip(rows, values):
                row[key] = value
        for row in rows:
            row["total"] = (row["left"] + row["right"]
                            if row["left"] is not None and row["right"] is not None else None)
            csv_rows.append(row)
    output = Path(args.csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["polyline_id", "dxf_handle", "station_m", "center_x", "center_y",
                         "left_width_m", "right_width_m", "total_width_m",
                         "left_confidence", "right_confidence", "status"])
        for r in csv_rows:
            writer.writerow([r["polyline_id"], r["handle"], "{:.3f}".format(r["station"]),
                "{:.4f}".format(r["x"]), "{:.4f}".format(r["y"]),
                "" if r["left"] is None else "{:.3f}".format(r["left"]),
                "" if r["right"] is None else "{:.3f}".format(r["right"]),
                "" if r["total"] is None else "{:.3f}".format(r["total"]),
                "{:.3f}".format(r["left_confidence"]), "{:.3f}".format(r["right_confidence"]),
                r["status"]])
    if not args.no_out_dxf:
        write_dxf(args.out_dxf, rows_by_id)
        print("Boundary DXF saved: {}".format(args.out_dxf))
    complete = [r["total"] for r in csv_rows if r["total"] is not None]
    uncertain = sum(r["status"] != "ok" for r in csv_rows)
    print("CSV saved: {}".format(output))
    print("Sections: {}; uncertain: {} ({:.1f}%)".format(
        len(csv_rows), uncertain, 100 * uncertain / max(1, len(csv_rows))))
    if complete:
        print("Total width min/median/max: {:.2f}/{:.2f}/{:.2f} m".format(
            min(complete), float(np.median(complete)), max(complete)))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        sys.exit(1)
