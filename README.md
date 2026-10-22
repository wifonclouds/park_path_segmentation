# Park Path Segmentation

Python project for segmenting paved paths from a terrain-classified point cloud using RGB/color features.

## Current objective

The input is a cleaned terrain point cloud exported from CloudCompare after CSF ground extraction. Pavement and grass have visibly different RGB distributions. The first step is to inspect the color distribution and establish a reproducible baseline before implementing automated classification.

## Repository layout

```text
park_path_segmentation/
├── input/                 # Local point-cloud inputs (not committed)
├── output/                # Generated results (not committed)
├── scripts/
│   └── segment_paving_rgb.py
├── .gitignore
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

Put the cleaned terrain point cloud in `input/`. LAS and LAZ files are ignored by Git because they can be very large and may contain project data.

## First run

```bash
python scripts/segment_paving_rgb.py input/your_terrain_cloud.las --output-dir output
```

The script currently reports RGB/color statistics and saves a compact color sample for exploration; it does **not** yet produce a final pavement segmentation. The next development step is to inspect the observed colors, then add a configurable classifier and export separate pavement/non-pavement point clouds.

Use `--help` to see available options.
