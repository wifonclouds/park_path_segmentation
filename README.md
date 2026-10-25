# Park Path Segmentation

Python tools for segmenting paved paths from a terrain-classified point cloud.

## Repository layout

```text
park_path_segmentation/
├── input/                 # Local point-cloud inputs (not committed)
├── output/                # Generated results (not committed)
├── scripts/
│   ├── segment_paving_rgb.py
│   └── segment_sidewalk_polyline.py
├── requirements.txt
└── README.md
```

## Setup (Ubuntu)

From the project directory:

```bash
sudo apt install gdal-bin
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
mkdir -p input output
```

Keep large LAS/LAZ and project-specific DXF files in `input/`; they are not committed to Git.

## Segment sidewalks using all polylines in a DXF

The script reads DXF line geometry with GDAL/OGR's `ogr2ogr`, so it can work with DXF files that `ezdxf` cannot parse. It collects all LineString and MultiLineString geometries, including their Z coordinates, and uses them together as the sidewalk centerlines.

Expected defaults:
- Input point cloud: `input/ground.las`
- DXF centerlines: `input/path.dxf`
- Half-width: 1.1 m (total nominal width 2.2 m)
- Vertical tolerance from interpolated line Z: ±0.10 m
- Sampling interval: maximum 0.25 m
- Processing block size: 250,000 points
- Outputs: `output/sidewalk_segmented.las` and `output/remaining_points.las`

Run from the repository root:

```bash
python3 scripts/segment_sidewalk_polyline.py
```

Or specify files and parameters explicitly:

```bash
python3 scripts/segment_sidewalk_polyline.py \
  --input input/ground.las \
  --dxf input/path.dxf \
  --half-width 1.1 \
  --vertical-tolerance 0.10 \
  --spacing 0.25 \
  --chunk-size 250000 \
  --output-dir output
```

For variable width, provide a CSV with columns `polyline_id,distance_m,half_width_m`. Polyline IDs are 1-based in the order GDAL returns line features; distances are cumulative 3D metres from each line's start. Lines without entries in the CSV use the default half-width.

The segmentation selects points whose XY distance to a candidate polyline segment is within the half-width and whose Z differs from the interpolated polyline elevation by no more than the vertical tolerance. Check the output in CloudCompare before using it downstream, especially at crossing or overlapping paths.
