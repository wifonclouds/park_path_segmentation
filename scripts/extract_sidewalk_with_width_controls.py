#!/usr/bin/env python3
"""Split a LAS point cloud into sidewalk corridor and remaining points.

The corridor follows DXF axis polylines. Width defaults to 1.1 m on each
side and is locally controlled by perpendicular 2-vertex sections in a
second DXF file.
"""
import argparse
from pathlib import Path

import ezdxf
import laspy
import numpy as np
from scipy.spatial import cKDTree


def polyline_points(entity):
    kind = entity.dxftype()
    if kind == "POLYLINE":
        return np.asarray([[v.dxf.location.x, v.dxf.location.y, v.dxf.location.z]
                           for v in entity.vertices], dtype=float)
    if kind == "LWPOLYLINE":
        xy = np.asarray([p[:2] for p in entity.get_points()], dtype=float)
        return np.column_stack((xy, np.zeros(len(xy))))
    return None


def read_axes(path):
    doc = ezdxf.readfile(str(path))
    axes = []
    for ent in doc.modelspace():
        pts = polyline_points(ent)
        if pts is not None and len(pts) >= 2:
            axes.append({"layer": ent.dxf.layer, "points": pts})
    if not axes:
        raise ValueError(f"No axis polylines found in {path}")
    return axes


def read_controls(path):
    doc = ezdxf.readfile(str(path))
    controls = []
    for ent in doc.modelspace():
        kind = ent.dxftype()
        if kind == "LINE":
            a, b = np.asarray(ent.dxf.start, float), np.asarray(ent.dxf.end, float)
        elif kind == "POLYLINE":
            pts = np.asarray([[v.dxf.location.x, v.dxf.location.y, v.dxf.location.z]
                              for v in ent.vertices], dtype=float)
            if len(pts) != 2:
                continue
            a, b = pts
        elif kind == "LWPOLYLINE":
            pts = list(ent.get_points())
            if len(pts) != 2:
                continue
            a = np.array([pts[0][0], pts[0][1], 0.0])
            b = np.array([pts[1][0], pts[1][1], 0.0])
        else:
            continue
        if np.linalg.norm((b - a)[:2]) > 1e-6:
            controls.append((a, b))
    return controls


def project_point(point, points):
    best = None
    station_base = 0.0
    for a, b in zip(points[:-1], points[1:]):
        vec = b[:2] - a[:2]
        length = float(np.linalg.norm(vec))
        if length < 1e-9:
            continue
        t = float(np.clip(np.dot(point - a[:2], vec) / (length * length), 0.0, 1.0))
        projected = a[:2] + t * vec
        dist = float(np.linalg.norm(point - projected))
        if best is None or dist < best[0]:
            tangent = vec / length
            best = (dist, station_base + t * length, projected, tangent)
        station_base += length
    return best


def prepare_axis_data(axes, controls, default_width, max_control_distance,
                      min_perpendicularity, transition_length, segment_length):
    """Create width profiles and a spatial index over short axis subsegments."""
    profiles = [[] for _ in axes]
    for a, b in controls:
        midpoint = (a[:2] + b[:2]) / 2.0
        candidates = []
        for axis_id, axis in enumerate(axes):
            proj = project_point(midpoint, axis["points"])
            if proj is not None:
                candidates.append((proj[0], axis_id, proj))
        if not candidates:
            continue
        distance, axis_id, proj = min(candidates, key=lambda x: x[0])
        _, station, projected, tangent = proj
        normal = np.array([-tangent[1], tangent[0]])
        vec = b[:2] - a[:2]
        perpendicularity = abs(float(np.dot(vec / np.linalg.norm(vec), normal)))
        if distance > max_control_distance or perpendicularity < min_perpendicularity:
            continue
        signed_a = float(np.dot(a[:2] - projected, normal))
        signed_b = float(np.dot(b[:2] - projected, normal))
        left = max(signed_a, signed_b, 0.0)
        right = max(-signed_a, -signed_b, 0.0)
        if left > 0 and right > 0:
            profiles[axis_id].append((station, left, right))

    segments = []
    for axis_id, axis in enumerate(axes):
        pts = axis["points"]
        cumulative = 0.0
        for a, b in zip(pts[:-1], pts[1:]):
            vec = b[:2] - a[:2]
            length = float(np.linalg.norm(vec))
            if length < 1e-8:
                continue
            count = max(1, int(np.ceil(length / segment_length)))
            for j in range(count):
                t0, t1 = j / count, (j + 1) / count
                p0 = a[:2] + t0 * vec
                p1 = a[:2] + t1 * vec
                mid = (p0 + p1) / 2.0
                segments.append((mid, p0, p1, axis_id,
                                 cumulative + t0 * length,
                                 cumulative + t1 * length))
            cumulative += length

    if not segments:
        raise ValueError("Axis DXF contains no usable XY segments.")

    mids = np.asarray([s[0] for s in segments])
    p0s = np.asarray([s[1] for s in segments])
    p1s = np.asarray([s[2] for s in segments])
    axis_ids = np.asarray([s[3] for s in segments], dtype=np.int32)
    st0s = np.asarray([s[4] for s in segments])
    st1s = np.asarray([s[5] for s in segments])

    # Build one station/left/right profile per axis. Widths are linearly
    # interpolated between controls, with a short ramp to/from the default.
    width_profiles = []
    for controls_for_axis in profiles:
        if not controls_for_axis:
            width_profiles.append(None)
            continue
        arr = np.asarray(controls_for_axis, dtype=float)
        arr = arr[np.argsort(arr[:, 0])]
        # Merge controls that project to effectively the same station.
        merged = []
        for row in arr:
            if merged and abs(row[0] - merged[-1][0]) < 0.05:
                old = merged[-1]
                merged[-1] = tuple((np.asarray(old) + row) / 2.0)
            else:
                merged.append(tuple(row))
        arr = np.asarray(merged)
        stations = np.concatenate(([arr[0, 0] - transition_length],
                                   arr[:, 0],
                                   [arr[-1, 0] + transition_length]))
        lefts = np.concatenate(([default_width], arr[:, 1], [default_width]))
        rights = np.concatenate(([default_width], arr[:, 2], [default_width]))
        width_profiles.append((stations, lefts, rights))

    return (cKDTree(mids), p0s, p1s, axis_ids, st0s, st1s, width_profiles)


def classify_chunk(points_xy, spatial, default_width, max_axis_distance, candidates):
    tree, p0s, p1s, axis_ids, st0s, st1s, profiles = spatial
    k = min(candidates, len(p0s))
    distances, indices = tree.query(points_xy, k=k, workers=-1)
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    best_dist = np.full(len(points_xy), np.inf)
    best_axis = np.full(len(points_xy), -1, dtype=np.int32)
    best_station = np.zeros(len(points_xy), dtype=float)
    best_lateral = np.zeros(len(points_xy), dtype=float)

    for rank in range(k):
        ids = indices[:, rank]
        a, b = p0s[ids], p1s[ids]
        vec = b - a
        lengths2 = np.einsum("ij,ij->i", vec, vec)
        t = np.clip(np.einsum("ij,ij->i", points_xy - a, vec) /
                    np.maximum(lengths2, 1e-12), 0.0, 1.0)
        proj = a + t[:, None] * vec
        delta = points_xy - proj
        dist = np.linalg.norm(delta, axis=1)
        update = dist < best_dist
        if not np.any(update):
            continue
        tangent = vec / np.maximum(np.sqrt(lengths2)[:, None], 1e-12)
        normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        lateral = np.einsum("ij,ij->i", delta, normal)
        best_dist[update] = dist[update]
        best_axis[update] = axis_ids[ids[update]]
        best_station[update] = (st0s[ids] + t * (st1s[ids] - st0s[ids]))[update]
        best_lateral[update] = lateral[update]

    keep = best_dist <= max_axis_distance
    left_width = np.full(len(points_xy), default_width, dtype=float)
    right_width = np.full(len(points_xy), default_width, dtype=float)
    for axis_id, profile in enumerate(profiles):
        if profile is None:
            continue
        mask = keep & (best_axis == axis_id)
        if not np.any(mask):
            continue
        stations, lefts, rights = profile
        left_width[mask] = np.interp(best_station[mask], stations, lefts)
        right_width[mask] = np.interp(best_station[mask], stations, rights)

    sidewalk = keep & (best_lateral <= left_width) & (best_lateral >= -right_width)
    return sidewalk


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("las", type=Path, help="Input LAS/LAZ point cloud")
    ap.add_argument("--axes", type=Path, default=Path("input/path_repaired.dxf"))
    ap.add_argument("--controls", type=Path, default=Path("input/width_control_repaired.dxf"))
    ap.add_argument("--sidewalk-out", type=Path, default=Path("output/sidewalk.las"))
    ap.add_argument("--remaining-out", type=Path, default=Path("output/remaining.las"))
    ap.add_argument("--default-width", type=float, default=1.1,
                    help="Default width on each side of axis, in metres (default: 1.1).")
    ap.add_argument("--max-axis-distance", type=float, default=8.0,
                    help="Maximum XY distance from an axis for a point to be considered.")
    ap.add_argument("--max-control-distance", type=float, default=5.0)
    ap.add_argument("--min-perpendicularity", type=float, default=0.70)
    ap.add_argument("--transition-length", type=float, default=5.0,
                    help="Ramp length from/to default width outside the first/last control.")
    ap.add_argument("--segment-length", type=float, default=1.0,
                    help="Maximum length of indexed axis subsegments.")
    ap.add_argument("--chunk-size", type=int, default=500_000)
    ap.add_argument("--candidates", type=int, default=12,
                    help="Number of nearby indexed axis subsegments checked per point.")
    args = ap.parse_args()

    if args.default_width <= 0 or args.max_axis_distance <= 0 or args.segment_length <= 0:
        ap.error("Widths and distances must be positive.")
    if args.chunk_size < 1 or args.candidates < 1 or args.transition_length < 0:
        ap.error("Invalid chunk size, candidates or transition length.")

    axes = read_axes(args.axes)
    controls = read_controls(args.controls)
    spatial = prepare_axis_data(
        axes, controls, args.default_width, args.max_control_distance,
        args.min_perpendicularity, args.transition_length, args.segment_length
    )
    accepted_count = sum(p is not None for p in spatial[-1])
    print(f"Axes: {len(axes)} | width controls: {len(controls)} | "
          f"axes with accepted controls: {accepted_count}", flush=True)
    print(f"Default corridor: {args.default_width:.2f} m per side | "
          f"transition: {args.transition_length:.2f} m", flush=True)

    args.sidewalk_out.parent.mkdir(parents=True, exist_ok=True)
    args.remaining_out.parent.mkdir(parents=True, exist_ok=True)
    total = sidewalk_total = 0
    with laspy.open(args.las) as reader:
        with laspy.open(args.sidewalk_out, mode="w", header=reader.header) as sidewalk_writer, \
             laspy.open(args.remaining_out, mode="w", header=reader.header) as remaining_writer:
            for chunk_no, chunk in enumerate(reader.chunk_iterator(args.chunk_size), 1):
                xy = np.column_stack((np.asarray(chunk.x), np.asarray(chunk.y)))
                mask = classify_chunk(xy, spatial, args.default_width,
                                      args.max_axis_distance, args.candidates)
                sidewalk_writer.write_points(chunk[mask])
                remaining_writer.write_points(chunk[~mask])
                total += len(chunk)
                sidewalk_total += int(mask.sum())
                print(f"Chunk {chunk_no}: {len(chunk):,} points | "
                      f"sidewalk {int(mask.sum()):,} | total {total:,}", flush=True)

    print("\nFinished.")
    print(f"Total points: {total:,}")
    print(f"Sidewalk: {sidewalk_total:,} ({100.0 * sidewalk_total / max(total, 1):.2f}%)")
    print(f"Remaining: {total - sidewalk_total:,}")
    print(f"Saved: {args.sidewalk_out}")
    print(f"Saved: {args.remaining_out}")


if __name__ == "__main__":
    main()
