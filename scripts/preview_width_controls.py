#!/usr/bin/env python3
"""Create a DXF preview of axes, manual width controls and their axis assignments."""
import argparse
from pathlib import Path

import ezdxf
import numpy as np

from inspect_width_controls import load_axes, load_controls, project_to_axis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axes", type=Path, default=Path("input/path_repaired.dxf"))
    parser.add_argument("--controls", type=Path, default=Path("input/width_control_repaired.dxf"))
    parser.add_argument("--out", type=Path, default=Path("output/width_control_preview.dxf"))
    parser.add_argument("--max-axis-distance", type=float, default=5.0)
    parser.add_argument("--min-perpendicularity", type=float, default=0.70)
    args = parser.parse_args()

    axes = load_axes(args.axes)
    controls = load_controls(args.controls)
    out = ezdxf.new("R2010")
    msp = out.modelspace()

    layers = {
        "AXES": 3,                 # green
        "CONTROL_ACCEPTED": 5,     # blue
        "CONTROL_REJECTED": 1,     # red
        "CONTROL_CONNECTOR": 8,    # grey
        "CONTROL_LABELS": 2,       # yellow
    }
    for name, color in layers.items():
        if name not in out.layers:
            out.layers.new(name, dxfattribs={"color": color})

    for axis_id, axis in enumerate(axes, 1):
        pts = [tuple(p) for p in axis["points"]]
        msp.add_polyline3d(pts, dxfattribs={"layer": "AXES"})
        midpoint = axis["points"][len(axis["points"]) // 2]
        msp.add_text(
            f"AXIS {axis_id}",
            dxfattribs={"layer": "CONTROL_LABELS", "height": 0.7,
                        "insert": (float(midpoint[0]), float(midpoint[1]), float(midpoint[2]))},
        )

    for control_id, control in enumerate(controls, 1):
        midpoint = (control["a"][:2] + control["b"][:2]) / 2.0
        candidates = []
        for axis_id, axis in enumerate(axes, 1):
            projection = project_to_axis(midpoint, axis["points"])
            if projection is not None:
                candidates.append((projection["distance"], axis_id, axis, projection))
        if not candidates:
            continue
        _, axis_id, axis, projection = min(candidates, key=lambda item: item[0])

        normal = np.array([-projection["tangent"][1], projection["tangent"][0]])
        vector = control["b"][:2] - control["a"][:2]
        unit = vector / np.linalg.norm(vector)
        perpendicularity = abs(float(np.dot(unit, normal)))
        signed_a = float(np.dot(control["a"][:2] - projection["projected"], normal))
        signed_b = float(np.dot(control["b"][:2] - projection["projected"], normal))
        accepted = (projection["distance"] <= args.max_axis_distance
                    and perpendicularity >= args.min_perpendicularity)
        layer = "CONTROL_ACCEPTED" if accepted else "CONTROL_REJECTED"

        msp.add_line(tuple(control["a"]), tuple(control["b"]),
                     dxfattribs={"layer": layer})
        projected = projection["projected"]
        connector_start = (float(midpoint[0]), float(midpoint[1]), float(control["a"][2]))
        connector_end = (float(projected[0]), float(projected[1]), float(control["a"][2]))
        msp.add_line(connector_start, connector_end,
                     dxfattribs={"layer": "CONTROL_CONNECTOR"})
        label = f"C{control_id} / A{axis_id} / L{max(signed_a, signed_b, 0):.2f} R{max(-signed_a, -signed_b, 0):.2f}"
        msp.add_text(
            label,
            dxfattribs={"layer": "CONTROL_LABELS", "height": 0.45,
                        "insert": (float(midpoint[0]), float(midpoint[1]), float(control["a"][2]))},
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.saveas(args.out)
    print(f"Saved preview: {args.out}")
    print(f"Axes: {len(axes)} | controls: {len(controls)}")
    print("DXF layers: AXES (green), CONTROL_ACCEPTED (blue), CONTROL_REJECTED (red),")
    print("            CONTROL_CONNECTOR (grey), CONTROL_LABELS (yellow)")


if __name__ == "__main__":
    main()
