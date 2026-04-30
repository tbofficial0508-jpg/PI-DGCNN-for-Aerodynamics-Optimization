NOTE: THIS PIPELINE ONLY WORKS FOR NVIDIA GPU MODEL TRAINING 

# PIDGCNN — Physics-Informed Aerodynamic Surrogate

Physics-Informed Dynamic Graph CNN surrogate for EDF drone aerodynamics.
Predicts body drag, EDF thrust, and mid-plane flow fields (p, u, v, w) from
surface point clouds. Supports optional PhysicsNeMo Tier-2 physics losses
(continuity + Euler momentum + wall BC).

---

## Installation

**Requirements:** Python ≥ 3.9, NVIDIA GPU with CUDA ≥ 11.8

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install --upgrade pip
```

Install PyTorch with CUDA (pick the command for your CUDA version at
https://pytorch.org/get-started/locally/):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Install remaining dependencies:

```bash
pip install -r requirements.txt
```

**PhysicsNeMo** (optional — only needed for `--use_physicsnemo` Phase 3 losses,
requires Linux + CUDA):

```bash
pip install -r requirements-physicsnemo.txt
```

Verify:

```bash
python model.py            # prints architecture + parameter count
python test_hybrid_loss.py # unit tests for the loss functions
```

---

## Files

| File | What it does |
|---|---|
| `config.py` | All hyperparameters in one dataclass |
| `model.py` | DGCNN + FiLM-SIREN architecture |
| `dataset.py` | Loads point-cloud `.npz` files and midplane CSVs |
| `losses.py` | Supervised MSE + optional physics residuals |
| `physics.py` | Tier-1: autograd continuity + wall BC |
| `physics_nemo.py` | Tier-2: PhysicsNeMo full NS losses |
| `train.py` | 3-phase training loop with EMA and early stopping |
| `run_pipeline.py` | Master CLI — runs any combination of stages |
| `build_pointcloud_dataset.py` | Converts Ansys Fluent exports to `.npz` dataset |
| `inference.py` | Single-sample inference |
| `optimise.py` | Grid search (20³) + Differential Evolution |
| `visualise.py` | Publication figures (convergence, scatter, fields, landscape) |
| `diagnose.py` | Debug suite — data, normalisation, and training isolation checks |
| `utils.py` | k-NN graph, Farthest Point Sampling, geometry morphing |
| `plotting_utils.py` | Shared matplotlib style |
| `test_hybrid_loss.py` | Unit tests |
| `visualize_geometries_sidebyside.py` | Side-by-side geometry comparison figure |
| `plot_geometry_pointclouds.py` | T1/J1/J2 point-cloud panel |

---

## Usage

### 1 — Build the dataset

Point `--input_root` at your Ansys Fluent export folders:

```bash
python build_pointcloud_dataset.py \
  --input_root  fluent_exports \
  --output_root dataset_pointcloud \
  --num_points  4096 \
  --edf_fraction 0.25 \
  --seed 1234
```

Each case folder needs `body_drag_*.out`, `edf_thrust_*.out`,
`body_surface_*.csv`, `edf_surface_*.csv`, and a midplane CSV whose filename
contains `GeometryT1`, `GeometryJ1`, or `GeometryJ2`.

To exclude an unconverged case without deleting it, move its `.npz` into
`dataset_pointcloud/excluded/` — the loader ignores subdirectories.

### 2 — Train

**Supervised DGCNN:**

```bash
python run_pipeline.py --train \
  --output_dir runs/my_run \
  --epochs 400 \
  --fourier_levels 10 \
  --field_finetune_epochs 2000 \
  --field_finetune_lr 5e-5
```

**PIDGCNN with PhysicsNeMo losses:**

```bash
python run_pipeline.py --train \
  --use_physicsnemo \
  --lambda_continuity 0.01 --lambda_momentum 1e-4 --lambda_wall_bc 0.10 \
  --physics_finetune_epochs 100 \
  --output_dir runs/my_run_pidgcnn \
  --epochs 400 --field_finetune_epochs 2000 --field_finetune_lr 5e-5
```

Training runs in three phases:
1. Joint scalar + field training — scalar head converges; field stalls (normal).
2. Encoder frozen, FiLM-SIREN re-initialised — field RMSE drops 2–3 orders.
3. Physics losses added (PIDGCNN only).

For small datasets (< 10 samples), add `--phase2_ema_decay 0.99` to prevent
EMA corruption from instability spikes.

### 3 — Evaluate, visualise, optimise

```bash
python run_pipeline.py --evaluate  --output_dir runs/my_run
python run_pipeline.py --visualise --output_dir runs/my_run
python run_pipeline.py --optimise  --output_dir runs/my_run --grid_batch_size 4
```

Or run everything in one go:

```bash
python run_pipeline.py --all --output_dir runs/my_run --grid_batch_size 4
```

> **Windows PowerShell:** replace each `\` line continuation with a backtick `` ` ``.

---

## Key flags

| Flag | Default | Notes |
|---|---|---|
| `--epochs` | 400 | Phase 1 budget |
| `--fourier_levels` | 10 | Fourier encoding depth (don't go below 10) |
| `--field_finetune_epochs` | 2000 | Phase 2 budget |
| `--field_finetune_lr` | 5e-5 | Phase 2 learning rate |
| `--physics_finetune_epochs` | 0 | Phase 3 budget (0 = off) |
| `--use_physicsnemo` | off | Enable Tier-2 NS physics losses |
| `--lambda_continuity` | 0.0 | Continuity residual weight |
| `--lambda_momentum` | 0.0 | Euler momentum weight (start at 1e-4) |
| `--lambda_wall_bc` | 0.0 | No-slip wall BC weight |
| `--batch_size` | 4 | Keep at 4; GroupNorm unstable above this |
| `--norm_type` | group | Always `group` — never `batch` |
| `--grid_batch_size` | 4 | Reduce if OOM during grid search |
| `--phase2_ema_decay` | 0.999 | Use 0.99 for small / duplicate datasets |
| `--skip_phase1` | off | Skip Phase 1, load existing checkpoint |

---

## Adding new geometry families

See [`ReadmeIfMoreGeometries.md`](ReadmeIfMoreGeometries.md).
