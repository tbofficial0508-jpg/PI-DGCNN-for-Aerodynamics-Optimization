# What to Do When You Have Real Multi-Geometry Data

This guide picks up from the point where you have replaced the 20 duplicate
CFD cases with 20 (or more) distinct geometry-simulation pairs in
`fluent_exports/`. All code is already written — this is purely a checklist
of what to run and what to change.

---

## Step 0 — Verify Your Fluent Exports

Each case folder must contain the same file patterns as before:

```text
fluent_exports/
  case_0001/          <- baseline geometry (your existing case)
    body_drag_*.out
    edf_thrust_*.out
    body_surface_*.csv
    edf_surface_*.csv
    midplane_z0_*.csv
  case_0002/          <- new geometry, different DVs
    ...
  case_0003/
    ...
```

Before doing anything else, confirm the range of your design variables covers
the space you care about. A quick sanity check:

```powershell
python .\build_pointcloud_dataset.py --input_root .\fluent_exports --output_root .\dataset_pointcloud_multi --num_points 4096 --edf_fraction 0.25 --seed 1234
```

Open `dataset_pointcloud_multi/manifest.json` and check that drag and thrust
vary meaningfully across cases. If all thrust values are within 0.01 N of each
other your geometry perturbations are too small to be useful.

---

## Step 1 — Two Config Changes (Required)

Open `config.py` and make exactly two edits:

**1. Disable overfit mode** (was `True` for the single-geometry case):
```python
overfit_mode: bool = False      # was True
train_fraction: float = 0.8     # 80% train, 20% validation
```

**2. Reduce Phase 2 fine-tune epochs** (field generalisation is harder than
memorisation; 2000 epochs will overfit to training geometries):
```python
field_finetune_epochs: int = 500    # was 2000
field_finetune_lr: float = 5e-5     # unchanged
```

Everything else — GroupNorm, Fourier levels, FiLM-SIREN architecture,
MultiStepLR schedule — stays exactly as it is.

---

## Step 2 — Rebuild the Dataset

```powershell
python .\build_pointcloud_dataset.py `
  --input_root  .\fluent_exports `
  --output_root .\dataset_pointcloud_multi `
  --num_points  4096 `
  --edf_fraction 0.25 `
  --seed 1234
```

Point all subsequent commands at `--data_root .\dataset_pointcloud_multi`.

---

## Step 3 — Train

```powershell
python .\run_pipeline.py `
  --train `
  --data_root   .\dataset_pointcloud_multi `
  --output_dir  .\runs\multi_geom_v1 `
  --epochs 400 `
  --fourier_levels 10 `
  --field_finetune_epochs 500 `
  --field_finetune_lr 5e-5
```

**What to watch during Phase 1:**

| Signal | Healthy | Investigate if... |
|---|---|---|
| `train_loss` decreasing | Smooth drop over ~100 epochs | Still at 0.063 after 50 epochs |
| `val_loss` tracking `train_loss` | Within ~20% | Val loss 5× train loss → overfitting |
| `thrust_MAE_N` | Below 0.05 N by epoch 100 | Above 0.1 N → model not learning |
| `drag_MAE_N` | Below 0.1 N by epoch 100 | Flat → check normalisation |

**What to watch during Phase 2:**

| Signal | Healthy | Investigate if... |
|---|---|---|
| `RMSE_norm` at sub-epoch 1 | ~0.25–1.0 (random init) | Already low → Phase 2 skipped |
| `RMSE_norm` at sub-epoch 100 | Below 0.05 | Still at 0.25 → FiLM not conditioning |
| Final `RMSE_norm` | Below 0.01 | Above 0.05 → need more Phase 2 epochs |

If Phase 2 stalls above RMSE_norm = 0.05, increase `--field_finetune_epochs`
to 1000 and re-run from the saved checkpoint (the best checkpoint is preserved).

---

## Step 4 — Evaluate

```powershell
python .\run_pipeline.py `
  --evaluate `
  --data_root  .\dataset_pointcloud_multi `
  --output_dir .\runs\multi_geom_v1
```

The key number to check is whether **held-out validation cases** (the 20% not
used in training) have similar MAE to the training cases. If validation MAE is
more than 3× the training MAE the model is overfitting — collect more data or
reduce model capacity.

**Target scalar accuracy for a useful optimisation surrogate:**

| Metric | Target |
|---|---|
| Thrust MAE (val) | Below 0.02 N (~2%) |
| Drag MAE (val) | Below 0.05 N (~1%) |

If you cannot reach these targets with 20 geometries, you need more CFD cases
before running optimisation — surrogate-based optimisation with a poorly fitted
model will find spurious optima.

---

## Step 5 — Optimise

Once scalar MAE is acceptable on held-out cases:

```powershell
python .\run_pipeline.py `
  --optimise `
  --data_root  .\dataset_pointcloud_multi `
  --output_dir .\runs\multi_geom_v1 `
  --grid_batch_size 4
```

**How to read the results:**

Open `runs/multi_geom_v1/optimisation_outcome.json`. Look at:

```json
"improvement_pct": X.XX
```

- If `improvement_pct` is near **0%**: the surrogate landscape is still flat —
  your geometry variation is too small or your surrogate hasn't converged.
- If `improvement_pct` is **positive and > 0.5%**: you have a genuine predicted
  optimum worth investigating with CFD.
- If `improvement_pct` is **implausibly large (> 10%)**: the surrogate is
  extrapolating outside its training data — treat with scepticism.

**The thrust landscape figure is your most important diagnostic.**
`pub_optimisation_landscape.pdf` should show a smooth gradient with a clear
peak. If it looks speckled or random, the surrogate is not yet reliable enough
for optimisation.

---

## Step 6 — Validate the Optimum with CFD

This is the step that makes the result scientifically credible.

1. Read the optimal design variables from `optimisation_outcome.json`:
   ```json
   "opt_dvs": [d0, d1, d2]
   ```
2. Generate the corresponding geometry in your CAD/meshing tool using those DV
   values.
3. Run a full Fluent simulation on the optimal geometry.
4. Compare the CFD thrust against:
   - The surrogate's prediction (`opt_thrust_N` in the JSON)
   - The baseline CFD thrust (`baseline.thrust_N`)

If the CFD-validated improvement agrees with the surrogate prediction to within
~20%, the surrogate is working as a design tool.

---

## Step 7 — Visualise

```powershell
python .\run_pipeline.py `
  --visualise `
  --data_root  .\dataset_pointcloud_multi `
  --output_dir .\runs\multi_geom_v1
```

Publication figures will update automatically to reflect the multi-geometry
results. The prediction scatter plot (`pub_prediction_scatter.pdf`) is
especially informative — with real geometry variation it should show a spread
of drag/thrust values rather than a single cluster.

---

## Troubleshooting Multi-Geometry Issues

**Surrogate predicts the same thrust for all geometries**
- Geometry DVs are too small — increase the design space bounds in `config.py`
- Normalisation is collapsing variation — check `manifest.json` scalar range

**Validation loss much higher than training loss**
- Need more data (aim for at least 5 cases per design variable dimension)
- Reduce `dropout` in `config.py` from 0.3 to 0.1
- Reduce `field_finetune_epochs` to prevent field head overfitting

**Field RMSE on validation cases above 5 m/s**
- Expected with only 20 geometries — field generalisation is harder than scalar
- Prioritise scalar accuracy for optimisation; field accuracy improves with data

**CUDA OOM during grid search**
- Use `--grid_batch_size 4` (already in the command above)
- Reduce `grid_resolution` in `config.py` from 20 to 10 (1000 evaluations)

**DE finds an optimum outside the training data range**
- The surrogate is extrapolating — add bounds constraints or collect CFD at
  the extremes of the design space before trusting the optimum

---

## How Many CFD Cases Do You Need?

| Cases | What you can expect |
|---|---|
| 5–10 | Landscape shape visible; scalar MAE probably 0.05–0.1 N; optimum unreliable |
| 20–30 | Reasonable scalar generalisation; optimum worth a CFD verification |
| 50–100 | Field generalisation starts to work; landscape smooth enough for gradient methods |
| 100+ | Full surrogate-in-the-loop optimisation with flow field interpretation |

For a PhD project, **50 cases** is a realistic minimum for publishable
generalisation results. **20 cases** is enough to demonstrate the method and
motivate the larger study.
