# Park Path Segmentation

Python project for segmenting paved paths from a terrain-classified point cloud using RGB/color features.

## Repository layout

```text
park_path_segmentation/
├── input/                 # Local point-cloud inputs (not committed)
├── output/                # Generated results (not committed)
├── scripts/
│   ├── segment_paving_rgb.py
│   └── estimate_sidewalk_width.py
├── requirements.txt
└── README.md
```

## Setup (Ubuntu)

From the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
mkdir -p input output
```

Put local point clouds and DXF files in `input/`. LAS/LAZ files and generated outputs should not be committed because they can be large and may contain project data.

## RGB inspection

```bash
python scripts/segment_paving_rgb.py input/your_terrain_cloud.las --output-dir output
```

This script reports RGB/color statistics and saves a compact color sample for exploration. It is not a final pavement segmentation classifier.

## Estimate sidewalk width from DXF polylines

`estimate_sidewalk_width.py` samples every DXF `POLYLINE` and `LWPOLYLINE` at regular intervals, projects 2D/3D polylines to XY, and estimates left/right sidewalk edges from RGB transitions in the LAS/LAZ point cloud. Point Z is intentionally ignored when selecting nearby points. The script reads the point cloud in chunks and uses SciPy's `cKDTree` for faster neighbourhood queries.

Example using the project input folder:

```bash
python scripts/estimate_sidewalk_width.py \
  input/ground.las input/path.dxf \
  --spacing 0.25 \
  --max-width 2.0 \
  --csv output/sidewalk_width.csv \
  --out-dxf output/sidewalk_boundaries.dxf
```

For a quick test, limit the number of cross-sections per polyline:

```bash
python scripts/estimate_sidewalk_width.py \
  input/ground.las input/path.dxf \
  --max-stations-per-polyline 100
```

Use `--help` to list all parameters. The CSV contains station coordinates, left/right/total width estimates, confidence heuristics and a status field. The output DXF contains estimated boundary polylines on separate layers. RGB transitions are only candidate edges, not guaranteed sidewalk boundaries; inspect the results in CloudCompare before using measurements. The analysis is entirely in XY, so points at different elevations can contribute if they are close in plan view.
