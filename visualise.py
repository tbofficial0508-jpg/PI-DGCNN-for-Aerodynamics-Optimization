"""
visualise.py - Publication-quality figure generation for the DGCNN CFD surrogate.

Figures produced
----------------
  (A) training_convergence.png/pdf  - 2x2 convergence grid
  (B) prediction_scatter.png/pdf    - predicted vs GT scatter for thrust + drag
  (C) field_comparison_<var>.png/pdf - midplane field CFD/pred/error for p, u, v, w
  (D) error_histograms.png/pdf       - point-wise error distributions + Gaussian fits
  (E) optimisation_landscape.png/pdf - thrust contour over DV0/DV1 at opt DV2
  (F) optimisation_convergence.png/pdf - thrust vs DE iteration

Rendering approach (Tecplot-quality)
-------------------------------------
  Scattered midplane points are interpolated onto a regular Cartesian grid
  using scipy cubic griddata, then rendered with imshow + bicubic interpolation.
  No contour lines are drawn. This matches how Tecplot and ParaView render
  unstructured CFD data: smooth continuous colour fill with no triangulation
  artefacts.

Usage
-----
  python visualise.py --checkpoint runs/pipeline/checkpoints/best.pt
                       --data_root dataset_pointcloud
                       --output_dir runs/pipeline
                       --all
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch
from scipy.interpolate import griddata
from scipy.ndimage import (
    gaussian_filter,
    distance_transform_edt,
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
)
from scipy.stats import norm as scipy_norm

from config import Config, cfg
from dataset import CFDDataset
from inference import predict_sample
from plotting_utils import (
    BODY_COLOR,
    EDF_COLOR,
    apply_publication_style,
    panel_label,
    save_figure_png_pdf,
)
from train import load_checkpoint

apply_publication_style()

_BODY_GREY = "#f6f6f6"   # near-white body fill to match CFD mask style


def _surface_points_from_npz(npz_path: str, y_eps: float = 0.025) -> np.ndarray:
    """
    Return midplane surface points as (N, 2) [x, z] from dense geometry arrays.
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception:
        return np.zeros((0, 2), dtype=np.float32)

    pts2d = []
    for key in ("dense_body_points", "dense_edf_points"):
        if key not in data:
            continue
        pts3d = np.asarray(data[key], dtype=np.float32)
        if pts3d.ndim != 2 or pts3d.shape[1] != 3:
            continue
        mid = pts3d[np.abs(pts3d[:, 1]) < y_eps][:, [0, 2]]
        if len(mid) > 0:
            pts2d.append(mid.astype(np.float32))
    if not pts2d:
        return np.zeros((0, 2), dtype=np.float32)
    return np.concatenate(pts2d, axis=0)


def _auto_zoom_from_surface(surface_xz: np.ndarray) -> dict[str, float]:
    """
    Derive a tight zoom box around geometry to reveal EDF and inner channel detail.
    """
    if surface_xz.size == 0:
        return {"xmin": -2.0, "xmax": 2.0, "ymin": -2.0, "ymax": 2.0}

    mins = np.min(surface_xz, axis=0)
    maxs = np.max(surface_xz, axis=0)
    span = np.maximum(maxs - mins, 0.75)
    pad = 0.18 * span
    return {
        "xmin": float(mins[0] - pad[0]),
        "xmax": float(maxs[0] + pad[0]),
        "ymin": float(mins[1] - pad[1]),
        "ymax": float(maxs[1] + pad[1]),
    }


def _body_mask_from_npz(
    npz_path: str,
    Xi: np.ndarray,
    Yi: np.ndarray,
    fluid_x: np.ndarray | None = None,
    fluid_y: np.ndarray | None = None,
    y_eps: float = 0.025,
) -> np.ndarray:
    """
    Build a filled solid mask from projected geometry while preserving the
    internal EDF/channel fluid region.
    """
    mask = np.zeros(Xi.shape, dtype=bool)
    pts = _surface_points_from_npz(npz_path, y_eps=y_eps)
    if len(pts) == 0:
        return mask

    def _rasterize_points(points: np.ndarray) -> np.ndarray:
        m = np.zeros(Xi.shape, dtype=bool)
        if points is None or len(points) == 0:
            return m
        ny, nx = Xi.shape
        x0, x1 = float(Xi[0, 0]), float(Xi[0, -1])
        y0, y1 = float(Yi[0, 0]), float(Yi[-1, 0])
        if abs(x1 - x0) < 1e-12 or abs(y1 - y0) < 1e-12:
            return m
        ix = np.rint((points[:, 0] - x0) / (x1 - x0) * (nx - 1)).astype(np.int64)
        iy = np.rint((points[:, 1] - y0) / (y1 - y0) * (ny - 1)).astype(np.int64)
        valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        if np.any(valid):
            m[iy[valid], ix[valid]] = True
        return m

    # 1) Surface raster (dense shell)
    shell = _rasterize_points(pts)
    if not shell.any():
        return mask
    # Bridge sparse surface-point gaps, fill enclosed regions, then shrink back.
    bridge_px = int(np.clip(round(0.016 * float(min(Xi.shape))), 5, 18))
    shell_closed = binary_dilation(shell, iterations=bridge_px)
    filled = binary_fill_holes(shell_closed)
    mask = binary_erosion(filled, iterations=max(1, bridge_px - 3))
    mask = binary_dilation(mask, iterations=1)
    mask = binary_erosion(mask, iterations=1)

    # Guard rail: if mask is implausibly small/large, fall back to thin shell mask
    frac = float(np.mean(mask))
    if frac < 5e-4 or frac > 0.45:
        mask = binary_dilation(shell, iterations=1)

    return mask


# -- Tecplot-style colormap ----------------------------------------------------
# Smooth rainbow matching Tecplot's default "Modified Rainbow":
# dark blue -> blue -> cyan -> green -> yellow -> orange -> red
_TECPLOT_CDICT = [
    (0.000, (0.00, 0.00, 0.56)),
    (0.100, (0.00, 0.00, 1.00)),
    (0.225, (0.00, 0.60, 1.00)),
    (0.350, (0.00, 1.00, 1.00)),
    (0.500, (0.00, 0.90, 0.00)),
    (0.650, (1.00, 1.00, 0.00)),
    (0.775, (1.00, 0.55, 0.00)),
    (0.900, (1.00, 0.00, 0.00)),
    (1.000, (0.56, 0.00, 0.00)),
]
CMAP_FIELD = LinearSegmentedColormap.from_list(
    "tecplot_rainbow",
    [(p, c) for p, c in _TECPLOT_CDICT],
    N=512,
)

# Diverging error colormap - clean blue/white/red (no green)
CMAP_ERROR = "RdBu_r"

# Grid resolution for scatter-to-grid interpolation
_N_GRID = 500


# -- Shared helpers ------------------------------------------------------------

def _r2_rmse_mae(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float("nan") if ss_tot <= 1e-12 else float(1.0 - ss_res / ss_tot)
    return r2, rmse, mae


def _robust_bounds(values: np.ndarray, q_low: float = 1.0, q_high: float = 99.0) -> tuple[float, float]:
    lo = float(np.percentile(values, q_low))
    hi = float(np.percentile(values, q_high))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def _collect_all_results(
    model: torch.nn.Module,
    ds: CFDDataset,
    device: torch.device,
    norm_stats: dict,
) -> list[dict]:
    results = []
    for idx in range(len(ds)):
        item = ds[idx]
        results.append(predict_sample(model, item, device, norm_stats))
    return results


def _scatter_to_grid(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    n_grid: int = _N_GRID,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate scattered (x, y, values) onto a regular Cartesian grid.

    Uses cubic griddata for smooth Tecplot-quality rendering. Any NaN cells
    left by cubic (outside convex hull) are filled by linear, leaving true
    exterior regions as NaN so they render as the axes background colour.

    Returns
    -------
    Xi, Yi : (n_grid, n_grid) meshgrid arrays
    Zi     : (n_grid, n_grid) interpolated values, NaN outside data hull
    """
    xi = np.linspace(xmin, xmax, n_grid)
    yi = np.linspace(ymin, ymax, n_grid)
    Xi, Yi = np.meshgrid(xi, yi)

    # Cubic first for smoothness
    try:
        Zi = griddata((x, y), values, (Xi, Yi), method="cubic")
    except Exception:
        Zi = np.full(Xi.shape, np.nan)

    # Fill NaN holes (hull edges) with linear
    nan_mask = ~np.isfinite(Zi)
    if nan_mask.any():
        try:
            Zi_lin = griddata((x, y), values, (Xi, Yi), method="linear")
            Zi[nan_mask] = Zi_lin[nan_mask]
        except Exception:
            pass

    return Xi, Yi, Zi


def _render_smooth(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    cmap,
    norm: mcolors.Normalize,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> matplotlib.image.AxesImage:
    """
    Core Tecplot-quality rendering: scatter -> grid -> imshow.

    No contour lines. Smooth bicubic-interpolated fill.
    """
    # Use only points within the zoom window
    mask = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
    px, py, pv = x[mask], y[mask], values[mask]
    if len(px) < 8:  # fallback: use all points
        px, py, pv = x, y, values

    _, _, Zi = _scatter_to_grid(px, py, pv, xmin, xmax, ymin, ymax)

    im = ax.imshow(
        Zi,
        extent=[xmin, xmax, ymin, ymax],
        origin="lower",
        cmap=cmap,
        norm=norm,
        interpolation="bilinear",   # smooth pixel-level blending
        aspect="equal",
        zorder=1,
    )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    return im


# -- (A) Training convergence --------------------------------------------------

def fig_training_convergence(history_json: Path, save_path: Path) -> None:
    """2x2 subplot: (a) total loss  (b) scalar loss+MAE  (c) field loss  (d) LR."""
    with open(history_json, encoding="utf-8") as f:
        history = json.load(f)

    epochs = [r["epoch"] for r in history]

    def _safe(key: str, fallback: float = 1e-12) -> list[float]:
        return [max(r.get(key, fallback), 1e-12) for r in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax_total, ax_scalar, ax_field, ax_lr = axes.flat

    ax_total.plot(epochs, _safe("train_loss"), color=BODY_COLOR, label="Train")
    ax_total.plot(epochs, _safe("val_loss"), color=EDF_COLOR, linestyle="--", label="Val")
    ax_total.set_yscale("log")
    ax_total.set_xlabel("Epoch [-]")
    ax_total.set_ylabel("Total loss [-]")
    ax_total.legend(loc="best")
    panel_label(ax_total, "a")

    ax_scalar.plot(epochs, _safe("train_scalar"), color=BODY_COLOR, label="Train scalar")
    ax_scalar.plot(epochs, _safe("val_scalar"), color=EDF_COLOR, linestyle="--", label="Val scalar")
    phys_key_t = "val_thrust_mae_phys" if "val_thrust_mae_phys" in history[0] else "val_thrust_mae"
    phys_key_d = "val_drag_mae_phys"   if "val_drag_mae_phys"   in history[0] else "val_drag_mae"
    ax_scalar.plot(epochs, _safe(phys_key_t), color="0.30", linestyle=":", label="Val thrust MAE [N]")
    ax_scalar.plot(epochs, _safe(phys_key_d), color="0.60", linestyle=":", label="Val drag MAE [N]")
    ax_scalar.set_yscale("log")
    ax_scalar.set_xlabel("Epoch [-]")
    ax_scalar.set_ylabel("Scalar loss / MAE [-]")
    ax_scalar.legend(loc="best")
    panel_label(ax_scalar, "b")

    ax_field.plot(epochs, _safe("train_field"), color=BODY_COLOR, label="Train field")
    ax_field.plot(epochs, _safe("val_field"),   color=EDF_COLOR, linestyle="--", label="Val field")
    if "train_field_grad" in history[0]:
        ax_field.plot(epochs, _safe("train_field_grad"), color=BODY_COLOR, linestyle="-.", label="Train field-grad")
        ax_field.plot(epochs, _safe("val_field_grad"),   color=EDF_COLOR,  linestyle=":",  label="Val field-grad")
    ax_field.set_yscale("log")
    ax_field.set_xlabel("Epoch [-]")
    ax_field.set_ylabel("Field loss [-]")
    ax_field.legend(loc="best")
    panel_label(ax_field, "c")

    ax_lr.plot(epochs, [r["lr"] for r in history], color=BODY_COLOR, label="LR")
    ax_lr.set_yscale("log")
    ax_lr.set_xlabel("Epoch [-]")
    ax_lr.set_ylabel("Learning rate [-]")
    ax_lr.legend(loc="best")
    panel_label(ax_lr, "d")

    save_figure_png_pdf(fig, save_path)
    plt.close(fig)
    print(f"  (A) Training convergence -> {save_path.with_suffix('.png')}")


# -- (B) Prediction scatter ----------------------------------------------------

def fig_prediction_scatter(results: list[dict], save_path: Path, subset_label: str | None = None) -> dict:
    thrust_true = np.array([r["scalar_true"][1] for r in results])
    thrust_pred = np.array([r["scalar_pred"][1] for r in results])
    drag_true   = np.array([r["scalar_true"][0] for r in results])
    drag_pred   = np.array([r["scalar_pred"][0] for r in results])

    fig, (ax_t, ax_d) = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True)
    scatter_stats: dict = {}

    def _scatter_panel(ax, true_v, pred_v, label: str, unit: str, panel: str) -> None:
        r2, rmse, mae = _r2_rmse_mae(true_v, pred_v)
        lo = min(float(true_v.min()), float(pred_v.min()))
        hi = max(float(true_v.max()), float(pred_v.max()))
        pad = 0.05 * (hi - lo + 1e-8)
        line = np.array([lo - pad, hi + pad])
        ax.plot(line, line, "k--", linewidth=1.0, label="1:1 line")
        ax.scatter(true_v, pred_v, s=60, c=BODY_COLOR, alpha=0.85, zorder=3, label="Samples")
        ax.set_xlabel(f"CFD {label} [{unit}]")
        ax.set_ylabel(f"Predicted {label} [{unit}]")
        ax.set_xlim(line[0], line[1])
        ax.set_ylim(line[0], line[1])
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="upper left", fontsize=9)
        panel_label(ax, panel)
        _mean_abs_true = float(np.mean(np.abs(true_v)))
        _mape_pct = 100.0 * mae / _mean_abs_true if _mean_abs_true > 1e-12 else float("nan")
        scatter_stats[label] = {"R2": r2, "RMSE": rmse, "MAE": mae, "MAPE_pct": _mape_pct, "unit": unit}

    _scatter_panel(ax_t, thrust_true, thrust_pred, "Thrust", "N", "a")
    _scatter_panel(ax_d, drag_true,   drag_pred,   "Drag",   "N", "b")
    if subset_label:
        fig.suptitle(f"Prediction Scatter ({subset_label})", fontsize=11)

    save_figure_png_pdf(fig, save_path)
    plt.close(fig)
    print(f"  (B) Prediction scatter -> {save_path.with_suffix('.png')}")
    return scatter_stats


# -- (C) Midplane field comparison ---------------------------------------------

def fig_field_comparison(
    result: dict,
    save_path: Path,
    sample_idx: int = 0,
    data_root: str | None = None,
) -> dict:
    """
    One figure per field variable [p, u, v, w]:
      3 columns: CFD | Predicted | Error
      Smooth continuous colour fill (no contour lines) - Tecplot style.
      Error panel uses diverging blue-white-red colormap centred on zero.
    """
    coords = result["mid_xyz"]          # (N, 3)
    x_c = coords[:, 0]
    span_y = float(np.ptp(coords[:, 1]))
    span_z = float(np.ptp(coords[:, 2]))
    y_c = coords[:, 1] if span_y >= span_z else coords[:, 2]
    y_label = "y [norm.]" if span_y >= span_z else "z [norm.]"

    npz_path: str | None = None
    surface_xz = np.zeros((0, 2), dtype=np.float32)
    if data_root is not None:
        npz_files = sorted(Path(data_root).glob("sample_*.npz"))
        if npz_files:
            _npz_idx = min(sample_idx, len(npz_files) - 1)
            npz_path = str(npz_files[_npz_idx])
            surface_xz = _surface_points_from_npz(npz_path)

    # Fixed square domain: both axes -0.6 to 0.6
    ZOOM = {"xmin": -0.6, "xmax": 0.6, "ymin": -0.6, "ymax": 0.6}

    _box = dict(boxstyle="round,pad=0.17", facecolor="white", alpha=0.85,
                edgecolor="0.65", linewidth=0.4)
    field_stats: dict = {}

    # Pre-compute velocity magnitude as a derived field
    _vm_true = np.sqrt(sum(result["field_true"][i] ** 2 for i in (1, 2, 3)))
    _vm_pred = np.sqrt(sum(result["field_pred"][i] ** 2 for i in (1, 2, 3)))
    _derived = {"vel_magnitude": (_vm_true, _vm_pred)}

    field_specs = [
        (0,             "pressure",      "Pressure",          "Pa"),
        (1,             "u_velocity",    "u-velocity",        "m/s"),
        (2,             "v_velocity",    "v-velocity",        "m/s"),
        (3,             "w_velocity",    "w-velocity",        "m/s"),
        ("vel_magnitude","vel_magnitude","Velocity Magnitude","m/s"),
    ]

    for ch_idx, suffix, name, unit in field_specs:
        if isinstance(ch_idx, str):
            true_v, pred_v = _derived[ch_idx]
        else:
            true_v = result["field_true"][ch_idx]
            pred_v = result["field_pred"][ch_idx]
        resid  = pred_v - true_v
        zoom_mask = (
            (x_c >= ZOOM["xmin"]) & (x_c <= ZOOM["xmax"]) &
            (y_c >= ZOOM["ymin"]) & (y_c <= ZOOM["ymax"])
        )
        if int(np.count_nonzero(zoom_mask)) >= 8:
            true_eval = true_v[zoom_mask]
            pred_eval = pred_v[zoom_mask]
        else:
            true_eval = true_v
            pred_eval = pred_v
        r2, rmse, mae = _r2_rmse_mae(true_eval, pred_eval)

        # Colour limits: shared across CFD and Prediction panels
        vmin, vmax = _robust_bounds(np.concatenate([true_v, pred_v]))
        if vmax <= vmin:
            vmax = vmin + 1e-6
        main_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        # Error limits: symmetric, diverging
        res_abs = float(np.percentile(np.abs(resid), 99.0))
        if not np.isfinite(res_abs) or res_abs < 1e-12:
            res_abs = 1e-12
        res_norm = mcolors.TwoSlopeNorm(vmin=-res_abs, vcenter=0.0, vmax=res_abs)

        # Layout identical to eval_sample: 2 rows, 3 columns
        fig = plt.figure(figsize=(12.0, 5.5), constrained_layout=True)
        gs  = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 0.10])
        ax_gt = fig.add_subplot(gs[0, 0])
        ax_pr = fig.add_subplot(gs[0, 1])
        ax_rs = fig.add_subplot(gs[0, 2])
        cax_m = fig.add_subplot(gs[1, 0:2])
        cax_e = fig.add_subplot(gs[1, 2])

        # Build regular grid over the focused domain (higher resolution for detail).
        _N = 500
        xi = np.linspace(ZOOM["xmin"], ZOOM["xmax"], _N)
        yi = np.linspace(ZOOM["ymin"], ZOOM["ymax"], _N)
        Xi, Yi = np.meshgrid(xi, yi)
        pts = (x_c, y_c)

        def _to_grid(values):
            return griddata(pts, values, (Xi, Yi), method="linear")

        Zi_true  = _to_grid(true_v)
        Zi_pred  = _to_grid(pred_v)
        Zi_resid = _to_grid(resid)

        # Filled solid mask with internal-channel preservation.
        body_mask = np.zeros(Xi.shape, dtype=bool)
        if npz_path is not None:
            _bm = _body_mask_from_npz(npz_path, Xi, Yi, fluid_x=x_c, fluid_y=y_c)
            if _bm.any():
                body_mask = _bm

        def _smooth(Zi):
            """Nearest-neighbour fill into body, blur, re-mask — no wall-BC halo."""
            Zi_f = Zi.astype(float).copy()
            if body_mask.any():
                Zi_f[body_mask] = np.nan
                _, idx = distance_transform_edt(body_mask, return_indices=True)
                Zi_f[body_mask] = Zi_f[tuple(idx[:, body_mask])]
            Zi_f = gaussian_filter(Zi_f, sigma=3)
            if body_mask.any():
                Zi_f[body_mask] = np.nan
            return Zi_f

        Zi_true  = _smooth(Zi_true)
        Zi_pred  = _smooth(Zi_pred)
        Zi_resid = _smooth(Zi_resid)

        # Colormaps with grey NaN (solid body colour)
        cmap_main = plt.cm.viridis.copy()
        cmap_main.set_bad(color=_BODY_GREY)
        cmap_err = plt.cm.coolwarm.copy()
        cmap_err.set_bad(color=_BODY_GREY)
        for ax in (ax_gt, ax_pr, ax_rs):
            ax.set_facecolor(_BODY_GREY)

        _extent = [ZOOM["xmin"], ZOOM["xmax"], ZOOM["ymin"], ZOOM["ymax"]]
        _ikw = dict(origin="lower", extent=_extent, interpolation="bilinear", aspect="equal")

        im_gt = ax_gt.imshow(Zi_true,  cmap=cmap_main, norm=main_norm, **_ikw)
        ax_pr.imshow(Zi_pred,  cmap=cmap_main, norm=main_norm, **_ikw)
        im_rs = ax_rs.imshow(Zi_resid, cmap=cmap_err,  norm=res_norm,  **_ikw)

        # Draw crisp white boundaries from the mask to match CFD-style masking.
        if body_mask.any():
            for ax in (ax_gt, ax_pr, ax_rs):
                ax.contour(
                    Xi, Yi, body_mask.astype(np.float32),
                    levels=[0.5], colors="white", linewidths=0.9, alpha=0.95, zorder=7
                )

        # Axes labels and limits (grid lines on by default, matching eval_sample)
        for ax in (ax_gt, ax_pr, ax_rs):
            ax.set_xlim(ZOOM["xmin"], ZOOM["xmax"])
            ax.set_ylim(ZOOM["ymin"], ZOOM["ymax"])
            ax.set_xlabel("x [norm.]")
            ax.set_ylabel(y_label)
            ax.set_aspect("equal", adjustable="box")

        # Panel annotations matching eval_sample labels
        ax_gt.text(0.02, 0.98, "CFD data",   transform=ax_gt.transAxes, ha="left", va="top", bbox=_box)
        ax_pr.text(0.02, 0.98, "Prediction", transform=ax_pr.transAxes, ha="left", va="top", bbox=_box)
        ax_rs.text(0.02, 0.98, "Error",      transform=ax_rs.transAxes, ha="left", va="top", bbox=_box)

        _true_range = float(np.max(true_eval) - np.min(true_eval))
        _rob_lo, _rob_hi = _robust_bounds(true_eval, 1.0, 99.0)
        _rob_range = float(_rob_hi - _rob_lo)
        _std_true = float(np.std(true_eval))
        _nrmse_pct = 100.0 * rmse / _true_range if _true_range > 1e-12 else float("nan")
        _nrmse_rob_pct = 100.0 * rmse / _rob_range if _rob_range > 1e-12 else float("nan")
        _cvrmse_pct = 100.0 * rmse / _std_true if _std_true > 1e-12 else float("nan")
        field_stats[name] = {
            "R2": r2, "RMSE": rmse,
            "nRMSE_range_pct": _nrmse_pct,
            "nRMSE_p99p1_pct": _nrmse_rob_pct,
            "CVRMSE_pct": _cvrmse_pct,
            "unit": unit,
        }

        panel_label(ax_gt, "a")
        panel_label(ax_pr, "b")
        panel_label(ax_rs, "c")

        # Colorbars matching eval_sample labels
        cb_m = fig.colorbar(im_gt, cax=cax_m, orientation="horizontal")
        cb_m.set_label(f"{name} [{unit}]")
        cb_m.ax.tick_params(labelsize=10)

        cb_e = fig.colorbar(im_rs, cax=cax_e, orientation="horizontal")
        cb_e.set_label(f"Error {name} [{unit}]")
        ticks_e = np.linspace(-res_abs, res_abs, 5)
        cb_e.set_ticks(ticks_e)
        cb_e.set_ticklabels([f"{t:.2g}" for t in ticks_e])
        cb_e.ax.tick_params(labelsize=9)

        fpath = save_path.parent / f"{save_path.stem}_{suffix}.png"
        save_figure_png_pdf(fig, fpath)
        plt.close(fig)
        print(f"  (C) Field comparison [{name}] -> {fpath}")

    return field_stats


# -- (D) Error histograms ------------------------------------------------------

def fig_error_histograms(results: list[dict], save_path: Path, subset_label: str | None = None) -> dict:
    p_errs = np.concatenate([r["field_pred"][0] - r["field_true"][0] for r in results])
    u_errs = np.concatenate([r["field_pred"][1] - r["field_true"][1] for r in results])
    v_errs = np.concatenate([r["field_pred"][2] - r["field_true"][2] for r in results])
    w_errs = np.concatenate([r["field_pred"][3] - r["field_true"][3] for r in results])

    vel_mag_true = np.concatenate([
        np.sqrt(r["field_true"][1]**2 + r["field_true"][2]**2 + r["field_true"][3]**2)
        for r in results
    ])
    vel_mag_pred = np.concatenate([
        np.sqrt(r["field_pred"][1]**2 + r["field_pred"][2]**2 + r["field_pred"][3]**2)
        for r in results
    ])
    vel_errs = vel_mag_pred - vel_mag_true

    specs = [
        (p_errs,   "Pressure error [Pa]",            "a"),
        (vel_errs, "Velocity magnitude error [m/s]", "b"),
        (u_errs,   "u-velocity error [m/s]",         "c"),
        (v_errs,   "v-velocity error [m/s]",         "d"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    hist_stats: dict = {}

    for ax, (errs, xlabel, label) in zip(axes.flat, specs):
        mu, sigma = float(np.mean(errs)), float(np.std(errs))
        lo, hi = _robust_bounds(errs, 0.5, 99.5)
        bins = np.linspace(lo, hi, 60)
        ax.hist(errs, bins=bins, density=True, color=BODY_COLOR, alpha=0.65,
                edgecolor="white", linewidth=0.3, label="Histogram")
        x_fit = np.linspace(lo, hi, 300)
        ax.plot(x_fit, scipy_norm.pdf(x_fit, mu, sigma), color=EDF_COLOR,
                linewidth=1.8, label="Gaussian fit")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Probability density [-]")
        ax.legend(loc="best", fontsize=9)
        panel_label(ax, label)
        hist_stats[xlabel] = {"mu": mu, "sigma": sigma}

    if subset_label:
        fig.suptitle(f"Field Error Distributions ({subset_label})", fontsize=11)

    save_figure_png_pdf(fig, save_path)
    plt.close(fig)
    print(f"  (D) Error histograms -> {save_path.with_suffix('.png')}")
    return hist_stats


# -- (E) Optimisation landscape ------------------------------------------------

def fig_optimisation_landscape(outcome_json: Path, landscape_npz: Optional[Path],
                                save_path: Path) -> None:
    """
    Smooth filled contour of predicted thrust over DV0/DV1 at optimum DV2.
    No contour lines. Tecplot-style smooth rendering via pcolormesh.
    """
    with open(outcome_json, encoding="utf-8") as f:
        outcome = json.load(f)

    opt_dvs = np.array(outcome["de_result"]["opt_dvs"])

    if landscape_npz is None or not Path(landscape_npz).exists():
        print(f"  (E) Landscape npz not found - skipping optimisation landscape figure.")
        return

    data = np.load(landscape_npz)
    thrust_cube = data["thrust_cube"]          # (r, r, r)
    dv0_ax = data["dv0_axis"]
    dv1_ax = data["dv1_axis"]
    dv2_ax = data["dv2_axis"]

    k2 = int(np.argmin(np.abs(dv2_ax - opt_dvs[2])))
    thrust_slice = thrust_cube[:, :, k2]       # (r, r)

    opt_thrust    = float(outcome["de_result"]["opt_thrust_N"])
    base_thrust   = outcome.get("baseline", {}).get("thrust_N", None)

    # Upsample via griddata for smoother rendering
    dv0_fine = np.linspace(dv0_ax[0], dv0_ax[-1], 300)
    dv1_fine = np.linspace(dv1_ax[0], dv1_ax[-1], 300)
    DV0_c, DV1_c = np.meshgrid(dv0_ax, dv1_ax, indexing="ij")
    pts_coarse = np.column_stack([DV0_c.ravel(), DV1_c.ravel()])
    vals_coarse = thrust_slice.ravel()

    DV0_f, DV1_f = np.meshgrid(dv0_fine, dv1_fine, indexing="ij")
    thrust_fine = griddata(pts_coarse, vals_coarse, (DV0_f, DV1_f), method="cubic")
    # Fill NaN edges with linear
    nan_m = ~np.isfinite(thrust_fine)
    if nan_m.any():
        thrust_fine_lin = griddata(pts_coarse, vals_coarse, (DV0_f, DV1_f), method="linear")
        thrust_fine[nan_m] = thrust_fine_lin[nan_m]

    vmin_t, vmax_t = _robust_bounds(thrust_fine[np.isfinite(thrust_fine)], 1.0, 99.0)
    norm_t = mcolors.Normalize(vmin=vmin_t, vmax=vmax_t)

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.2), constrained_layout=True)
    ax.grid(False)
    ax.set_facecolor("0.92")

    # Smooth pcolormesh (no contour lines)
    pm = ax.pcolormesh(DV0_f, DV1_f, thrust_fine,
                       cmap=CMAP_FIELD, norm=norm_t,
                       shading="gouraud",       # smooth interpolation between cells
                       zorder=1)

    # Optimum marker
    ax.scatter([opt_dvs[0]], [opt_dvs[1]],
               c="white", marker="*", s=280, zorder=5, linewidths=0.8,
               edgecolors="black", label=f"Optimum  {opt_thrust:.4f} N")

    # Baseline marker (if available)
    if base_thrust is not None:
        ax.text(0.97, 0.04,
                f"Baseline: {base_thrust:.4f} N\nOptimum:  {opt_thrust:.4f} N\n"
                f"Gain: {outcome.get('improvement_pct', 0.0):+.2f}%",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white", alpha=0.88,
                          edgecolor="0.55", linewidth=0.5))

    cb = fig.colorbar(pm, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Predicted thrust [N]", fontsize=10)
    cb.ax.tick_params(labelsize=9)
    cb.locator = matplotlib.ticker.MaxNLocator(nbins=6)
    cb.update_ticks()

    ax.set_xlabel("DV0  body scale [-]", fontsize=11)
    ax.set_ylabel("DV1  duct scale [-]", fontsize=11)
    ax.set_title(f"DV2 EDF x-offset = {dv2_ax[k2]:.4f}", fontsize=10)
    ax.legend(loc="upper left", fontsize=9,
              framealpha=0.88, edgecolor="0.55")
    panel_label(ax, "a")

    save_figure_png_pdf(fig, save_path)
    plt.close(fig)
    print(f"  (E) Optimisation landscape -> {save_path.with_suffix('.png')}")


# -- (F) Optimisation convergence ----------------------------------------------

def fig_optimisation_convergence(outcome_json: Path, save_path: Path) -> None:
    with open(outcome_json, encoding="utf-8") as f:
        outcome = json.load(f)

    baseline_thrust = outcome["baseline_prediction"]["thrust_N"]
    opt_thrust      = outcome["de_result"]["opt_thrust_N"]
    n_evals         = outcome["de_result"]["n_evaluations"]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)

    de_history = outcome.get("de_history", None)
    if de_history is not None and isinstance(de_history, list) and len(de_history) > 1:
        iters   = [d["iteration"] for d in de_history]
        thrusts = [d["best_thrust_N"] for d in de_history]
        ax.plot(iters, thrusts, color=BODY_COLOR, linewidth=1.5, label="Best thrust")
        ax.axhline(baseline_thrust, color=EDF_COLOR, linestyle="--", linewidth=1.2,
                   label=f"Baseline {baseline_thrust:.4f} N")
        ax.set_xlabel("DE iteration [-]")
        ax.set_ylabel("Best predicted thrust [N]")
    else:
        ax.plot([0, n_evals], [baseline_thrust, opt_thrust], color=BODY_COLOR,
                linewidth=1.5, marker="o", markersize=6, label="Start -> Optimum")
        ax.axhline(baseline_thrust, color=EDF_COLOR, linestyle="--", linewidth=1.2,
                   label=f"Baseline {baseline_thrust:.4f} N")
        ax.set_xlabel("Function evaluations [-]")
        ax.set_ylabel("Best predicted thrust [N]")
        ax.annotate(
            f"Optimum\n{opt_thrust:.4f} N",
            xy=(n_evals, opt_thrust),
            xytext=(n_evals * 0.6, opt_thrust + 0.05 * (opt_thrust - baseline_thrust + 1e-6)),
            arrowprops=dict(arrowstyle="->", color="0.3"),
            fontsize=9,
        )

    gain_pct = outcome.get("improvement_pct", 0.0)
    ax.set_title(f"Thrust gain: {gain_pct:+.2f}%  (vs predicted baseline)", fontsize=10)
    ax.legend(loc="lower right", fontsize=9)
    panel_label(ax, "a")

    save_figure_png_pdf(fig, save_path)
    plt.close(fig)
    print(f"  (F) Optimisation convergence -> {save_path.with_suffix('.png')}")


# -- Error summary text file ---------------------------------------------------

def _write_error_summary(all_stats: dict, path: Path) -> None:
    """Write all collected error metrics to a plain-text summary file."""
    lines: list[str] = []
    sep = "=" * 56
    lines += [sep, "PREDICTION ERROR SUMMARY",
              f"Generated : {datetime.datetime.now():%Y-%m-%d %H:%M:%S}", sep, ""]
    for section, stats in all_stats.items():
        lines += [f"[{section}]", "-" * 44]
        for var, metrics in stats.items():
            unit = metrics.get("unit", "")
            lines.append(f"  {var}  [{unit}]" if unit else f"  {var}")
            for k, v in metrics.items():
                if k == "unit":
                    continue
                if isinstance(v, float):
                    lines.append(f"    {k:<24} = {v:.4g}")
                else:
                    lines.append(f"    {k:<24} = {v}")
            lines.append("")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [stats] Error summary -> {path}")


# -- Main ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="DGCNN CFD surrogate publication figures")
    parser.add_argument("--checkpoint",    type=str, default=str(cfg.best_checkpoint))
    parser.add_argument("--data_root",     type=str, default=str(cfg.data_root))
    parser.add_argument("--output_dir",    type=str, default=str(cfg.output_dir))
    parser.add_argument("--history_json",  type=str, default=None)
    parser.add_argument("--outcome_json",  type=str, default=None)
    parser.add_argument("--landscape_npz", type=str, default=None)
    parser.add_argument("--device",        type=str, default=cfg.device)
    parser.add_argument("--figure", type=str, default="all",
                        choices=["all", "convergence", "scatter", "fields",
                                 "histograms", "landscape", "optconv"])
    parser.add_argument("--sample_idx", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_all = args.figure == "all"
    device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")) \
        if args.device == "auto" else torch.device(args.device)

    model = norm_stats = ds = results = None

    def _ensure_model() -> None:
        nonlocal model, norm_stats, ds, results
        if model is not None:
            return
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        model, ckpt = load_checkpoint(ckpt_path, device)
        norm_stats = ckpt["norm_stats"]
        print(f"[visualise] Loaded checkpoint epoch {ckpt.get('epoch', '?')}: {ckpt_path}")
        ds = CFDDataset(data_root=args.data_root)
        print(f"[visualise] Collecting predictions for {len(ds)} samples ...")
        results = _collect_all_results(model, ds, device, norm_stats)

    print("[visualise] Starting figure generation ...")
    all_stats: dict = {}

    if run_all or args.figure == "convergence":
        hjson = Path(args.history_json) if args.history_json else out_dir / "history.json"
        if hjson.exists():
            fig_training_convergence(hjson, out_dir / "pub_training_convergence.png")
        else:
            print(f"  (A) history.json not found at {hjson} - skipping.")

    if run_all or args.figure == "scatter":
        _ensure_model()
        scatter_stats = fig_prediction_scatter(results, out_dir / "pub_prediction_scatter.png")
        all_stats["B: Scalar Predictions (all samples)"] = scatter_stats

    if run_all or args.figure == "fields":
        _ensure_model()
        idx = args.sample_idx if args.sample_idx < len(results) else 0
        field_stats = fig_field_comparison(results[idx], out_dir / f"pub_field_comparison_s{idx:03d}.png", idx,
                             data_root=args.data_root)
        all_stats[f"C: Field Comparison (sample {idx:03d})"] = field_stats

    if run_all or args.figure == "histograms":
        _ensure_model()
        hist_stats = fig_error_histograms(results, out_dir / "pub_error_histograms.png")
        all_stats["D: Error Histograms (all samples)"] = hist_stats

    if run_all or args.figure == "landscape":
        ojson = Path(args.outcome_json) if args.outcome_json else out_dir / "optimisation_outcome.json"
        lnpz  = Path(args.landscape_npz) if args.landscape_npz else out_dir / "thrust_landscape.npz"
        if ojson.exists():
            fig_optimisation_landscape(ojson, lnpz if lnpz.exists() else None,
                                       out_dir / "pub_optimisation_landscape.png")
        else:
            print(f"  (E) optimisation_outcome.json not found - skipping landscape figure.")

    if run_all or args.figure == "optconv":
        ojson = Path(args.outcome_json) if args.outcome_json else out_dir / "optimisation_outcome.json"
        if ojson.exists():
            fig_optimisation_convergence(ojson, out_dir / "pub_optimisation_convergence.png")
        else:
            print(f"  (F) optimisation_outcome.json not found - skipping convergence figure.")

    if all_stats:
        _write_error_summary(all_stats, out_dir / "error_summary.txt")

    print("[visualise] Done.")


if __name__ == "__main__":
    main()
