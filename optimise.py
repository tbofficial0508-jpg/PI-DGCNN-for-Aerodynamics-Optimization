from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import differential_evolution

from config import Config, cfg
from dataset import CFDDataset
from plotting_utils import apply_publication_style, panel_label, save_figure_png_pdf
from train import load_checkpoint
from utils import morph_geometry, plot_pointcloud


apply_publication_style()


class SurrogateObjective:
    """
    Scipy-compatible objective wrapping the surrogate model.

    Supports three objectives (controlled by the `objective` parameter):
      "thrust"     - maximise thrust (N)
      "efficiency" - maximise thrust / (drag + thrust)
      "combined"   - maximise thrust - efficiency_alpha * drag
    """

    def __init__(
        self,
        model: torch.nn.Module,
        base_pts: np.ndarray,
        mid_xyz: torch.Tensor,
        conditions: torch.Tensor,
        scalar_scale: np.ndarray,
        device: torch.device,
        call_counter: list[int],
        mc_samples: int = 1,
        uncertainty_beta: float = 0.0,
        objective: str = "thrust",
        efficiency_alpha: float = 0.5,
    ):
        self.model = model
        self.base_pts = base_pts
        self.mid_xyz = mid_xyz
        self.conditions = conditions
        self.scalar_scale = scalar_scale
        self.device = device
        self.counter = call_counter
        self.mc_samples = max(1, int(mc_samples))
        self.uncertainty_beta = max(0.0, float(uncertainty_beta))
        self.objective = str(objective)
        self.efficiency_alpha = float(efficiency_alpha)

    def _set_mc_dropout(self, enabled: bool) -> None:
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train(enabled)

    @torch.no_grad()
    def predict_scalar_stats(
        self, geo_t: torch.Tensor
    ) -> tuple[float, float, float, float]:
        """
        Returns (drag_mean, drag_std, thrust_mean, thrust_std) in physical units.
        One or more MC-dropout passes when mc_samples > 1.
        """
        self.model.eval()
        if self.mc_samples <= 1:
            scalar_n, _ = self.model(geo_t, self.mid_xyz, self.conditions)
            drag   = float(scalar_n[0, 0].cpu()) * float(self.scalar_scale[0])
            thrust = float(scalar_n[0, 1].cpu()) * float(self.scalar_scale[1])
            return drag, 0.0, thrust, 0.0

        self._set_mc_dropout(True)
        drag_s, thrust_s = [], []
        for _ in range(self.mc_samples):
            scalar_n, _ = self.model(geo_t, self.mid_xyz, self.conditions)
            drag_s.append(float(scalar_n[0, 0].cpu()) * float(self.scalar_scale[0]))
            thrust_s.append(float(scalar_n[0, 1].cpu()) * float(self.scalar_scale[1]))
        self.model.eval()

        da = np.asarray(drag_s,   dtype=np.float32)
        ta = np.asarray(thrust_s, dtype=np.float32)
        return float(da.mean()), float(da.std(ddof=0)), float(ta.mean()), float(ta.std(ddof=0))

    # Keep backward-compat alias used elsewhere in the file
    @torch.no_grad()
    def predict_thrust_stats(self, geo_t: torch.Tensor) -> tuple[float, float]:
        _, _, tm, ts = self.predict_scalar_stats(geo_t)
        return tm, ts

    @torch.no_grad()
    def __call__(self, dvs: np.ndarray) -> float:
        self.counter[0] += 1
        morphed = morph_geometry(self.base_pts, dvs[0], dvs[1], dvs[2])
        geo_t   = torch.from_numpy(morphed.T).float().unsqueeze(0).to(self.device)

        drag_mean, drag_std, thrust_mean, thrust_std = self.predict_scalar_stats(geo_t)
        score = _objective_score(drag_mean, thrust_mean, self.objective, self.efficiency_alpha)
        # Uncertainty penalty: score = mean - beta * std (only for thrust component)
        robust_score = float(score) - self.uncertainty_beta * thrust_std
        return -robust_score  # minimise negative score


def plot_initial_final_pointcloud(
    initial_pts: np.ndarray,
    optimised_pts: np.ndarray,
    save_path: Path,
) -> None:
    """
    Save a side-by-side publication figure: initial vs optimised geometry.
    """
    def _set_equal_3d(ax, xyz: np.ndarray, pad_frac: float = 0.05) -> None:
        mins = xyz.min(axis=0)
        maxs = xyz.max(axis=0)
        center = 0.5 * (mins + maxs)
        span = max(float((maxs - mins).max()), 1e-6)
        half = 0.5 * span * (1.0 + pad_frac)
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)
        try:
            ax.set_box_aspect((1.0, 1.0, 1.0))
        except Exception:
            pass

    def _style_3d(ax) -> None:
        ax.set_xlabel("x [norm.]")
        ax.set_ylabel("y [norm.]")
        ax.set_zlabel("z [norm.]")
        ax.view_init(elev=22, azim=-58)
        ax.grid(True)

    def _plot_one(ax, pts: np.ndarray, label_text: str) -> None:
        is_body = pts[:, 6] > 0.5
        is_edf = pts[:, 7] > 0.5
        body_xyz = pts[is_body, :3]
        edf_xyz = pts[is_edf, :3]
        all_xyz = np.concatenate([body_xyz, edf_xyz], axis=0)

        ax.scatter(
            body_xyz[:, 0], body_xyz[:, 1], body_xyz[:, 2],
            s=0.55, c="#1f77b4", alpha=0.95, linewidths=0.0, rasterized=True, label="Body"
        )
        ax.scatter(
            edf_xyz[:, 0], edf_xyz[:, 1], edf_xyz[:, 2],
            s=0.80, c="#ff7f0e", alpha=0.98, linewidths=0.0, rasterized=True, label="EDF"
        )
        _set_equal_3d(ax, all_xyz)
        _style_3d(ax)
        ax.text2D(0.03, 0.97, label_text, transform=ax.transAxes, ha="left", va="top")

    fig = plt.figure(figsize=(10.8, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 2)
    ax0 = fig.add_subplot(gs[0, 0], projection="3d")
    ax1 = fig.add_subplot(gs[0, 1], projection="3d")

    _plot_one(ax0, initial_pts, "Initial geometry")
    _plot_one(ax1, optimised_pts, "Optimised geometry")
    panel_label(ax0, "a")
    panel_label(ax1, "b")
    ax1.legend(loc="upper right")

    png_path, pdf_path = save_figure_png_pdf(fig, save_path)
    plt.close(fig)
    print(f"Initial/final geometry figure saved: {png_path}")
    print(f"Initial/final geometry figure saved: {pdf_path}")


@torch.no_grad()
def predict_from_geometry(
    model: torch.nn.Module,
    points: np.ndarray,
    mid_xyz: torch.Tensor,
    conditions: torch.Tensor,
    scalar_scale: np.ndarray,
    field_scale: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run one forward pass for a morphed geometry and return physical-unit outputs.

    Returns
    -------
    scalar_pred : (2,) [drag_N, thrust_N]
    field_pred  : (4, N_mid) [pressure, u, v, w]
    """
    geo_t = torch.from_numpy(points.T).float().unsqueeze(0).to(device)

    model.eval()
    scalar_n, field_n = model(geo_t, mid_xyz, conditions)

    scalar_pred = scalar_n[0].detach().cpu().numpy() * scalar_scale
    field_pred = field_n[0].detach().cpu().numpy() * field_scale[:, None]
    return scalar_pred.astype(np.float32), field_pred.astype(np.float32)


@torch.no_grad()
def eval_grid_batch(
    model: torch.nn.Module,
    base_pts: np.ndarray,
    dv_grid: np.ndarray,
    mid_xyz: torch.Tensor,
    conditions: torch.Tensor,
    scalar_scale: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate drag and thrust over a DV grid in batches.

    Returns
    -------
    drag_vals   : (M,) drag values in N
    thrust_vals : (M,) thrust values in N
    """
    m = dv_grid.shape[0]
    drag_vals   = np.empty(m, dtype=np.float32)
    thrust_vals = np.empty(m, dtype=np.float32)
    model.eval()

    for start in range(0, m, batch_size):
        end = min(start + batch_size, m)
        batch = dv_grid[start:end]
        bsz = end - start

        morphed = [morph_geometry(base_pts, *dvs) for dvs in batch]
        geo_batch = torch.from_numpy(
            np.stack([arr.T for arr in morphed], axis=0)
        ).float().to(device)

        mid_batch  = mid_xyz.expand(bsz, -1, -1)
        cond_batch = conditions.expand(bsz, -1)

        scalar_n, _ = model(geo_batch, mid_batch, cond_batch)
        scalars = scalar_n.detach().cpu().numpy()
        drag_vals[start:end]   = scalars[:, 0] * scalar_scale[0]
        thrust_vals[start:end] = scalars[:, 1] * scalar_scale[1]

    return drag_vals, thrust_vals


def _objective_score(
    drag: float | np.ndarray,
    thrust: float | np.ndarray,
    objective: str = "thrust",
    efficiency_alpha: float = 0.5,
) -> float | np.ndarray:
    """
    Compute scalar objective from drag and thrust.

    Parameters
    ----------
    objective : "thrust"     - maximise thrust (N)
                "efficiency" - maximise thrust / (drag + thrust)  [propulsive fraction]
                "combined"   - maximise thrust - efficiency_alpha * drag
    """
    if objective == "efficiency":
        denom = np.abs(drag) + np.abs(thrust) + 1e-12
        return thrust / denom
    if objective == "combined":
        return thrust - efficiency_alpha * drag
    # default: thrust
    return thrust


def plot_pareto_scatter(
    drag_vals: np.ndarray,
    thrust_vals: np.ndarray,
    opt_drag: float,
    opt_thrust: float,
    base_drag: float,
    base_thrust: float,
    objective: str,
    save_path: Path,
) -> None:
    """
    Scatter plot of all grid (drag, thrust) pairs with Pareto front highlighted.
    Engineers can see the full thrust-drag trade-off surface in one glance.
    """
    # Identify Pareto-optimal points: no other point has both higher thrust AND lower drag
    n = len(drag_vals)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        # Dominated if there exists j s.t. drag[j] <= drag[i] AND thrust[j] >= thrust[i]
        # and at least one strict
        dominated = ((drag_vals <= drag_vals[i]) & (thrust_vals >= thrust_vals[i])
                     & ((drag_vals < drag_vals[i]) | (thrust_vals > thrust_vals[i])))
        if dominated.any():
            is_pareto[i] = False

    pareto_drag   = drag_vals[is_pareto]
    pareto_thrust = thrust_vals[is_pareto]
    sort_idx = np.argsort(pareto_drag)
    pareto_drag   = pareto_drag[sort_idx]
    pareto_thrust = pareto_thrust[sort_idx]

    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)

    # All grid points
    ax.scatter(drag_vals, thrust_vals, s=2.5, c="#aec6cf", alpha=0.4,
               linewidths=0.0, rasterized=True, label="Grid points")

    # Pareto front
    ax.plot(pareto_drag, pareto_thrust, "o-", color="#d62728", lw=1.5, ms=4,
            zorder=4, label=f"Pareto front ({is_pareto.sum()} pts)")

    # Baseline and optimum
    ax.scatter([base_drag], [base_thrust], s=90, c="#1f77b4", marker="D",
               zorder=5, label=f"CFD baseline  ({base_drag:.3f}, {base_thrust:.3f}) N")
    ax.scatter([opt_drag], [opt_thrust], s=110, c="#ff7f0e", marker="*",
               zorder=6, label=f"Surrogate opt ({opt_drag:.3f}, {opt_thrust:.3f}) N")

    ax.set_xlabel("Drag [N]")
    ax.set_ylabel("Thrust [N]")
    ax.set_title(f"Thrust-Drag trade-off  (objective: {objective})")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, lw=0.4, alpha=0.5)

    png_path, pdf_path = save_figure_png_pdf(fig, save_path)
    plt.close(fig)
    print(f"Pareto scatter saved: {png_path}")


def plot_landscape(
    dv_axes: list[np.ndarray],
    thrust_cube: np.ndarray,
    opt_dvs: np.ndarray,
    save_path: Path,
) -> None:
    """
    Plot thrust landscape slice:
      - 3D surface of thrust vs DV0/DV1 at DV2 ~= optimum
      - contour map of the same slice
    """
    dv0_axis, dv1_axis, dv2_axis = dv_axes
    k2 = int(np.argmin(np.abs(dv2_axis - opt_dvs[2])))
    thrust_slice = thrust_cube[:, :, k2]

    dv0_grid, dv1_grid = np.meshgrid(dv0_axis, dv1_axis, indexing="ij")
    opt_thrust = float(thrust_slice[np.argmin(np.abs(dv0_axis - opt_dvs[0])), np.argmin(np.abs(dv1_axis - opt_dvs[1]))])

    fig = plt.figure(figsize=(12, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0])
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax2d = fig.add_subplot(gs[0, 1])

    surf = ax3d.plot_surface(
        dv0_grid,
        dv1_grid,
        thrust_slice,
        cmap="viridis",
        linewidth=0.0,
        antialiased=True,
        alpha=0.95,
    )
    ax3d.scatter(opt_dvs[0], opt_dvs[1], opt_thrust, c="k", marker="*", s=110, zorder=10)
    ax3d.set_xlabel("DV0 body scale [-]")
    ax3d.set_ylabel("DV1 duct scale [-]")
    ax3d.set_zlabel("Thrust [N]")
    ax3d.view_init(elev=28, azim=-132)
    panel_label(ax3d, "a")

    contour = ax2d.contourf(dv0_grid, dv1_grid, thrust_slice, levels=24, cmap="viridis")
    ax2d.contour(dv0_grid, dv1_grid, thrust_slice, levels=12, colors="k", linewidths=0.45, alpha=0.5)
    ax2d.scatter(opt_dvs[0], opt_dvs[1], c="k", marker="*", s=110, zorder=10)
    ax2d.set_xlabel("DV0 body scale [-]")
    ax2d.set_ylabel("DV1 duct scale [-]")
    panel_label(ax2d, "b")

    cbar = fig.colorbar(contour, ax=[ax3d, ax2d], fraction=0.05, pad=0.02)
    cbar.set_label("Thrust [N]")

    fig.text(
        0.5,
        0.98,
        f"Slice at DV2 EDF x-offset = {dv2_axis[k2]:.4f} [norm.]",
        ha="center",
        va="top",
    )

    png_path, pdf_path = save_figure_png_pdf(fig, save_path)
    plt.close(fig)
    print(f"Landscape saved: {png_path}")
    print(f"Landscape saved: {pdf_path}")


def optimise(
    c: Config = cfg,
    checkpoint_path: Path | None = None,
    save_only_outcome: bool | None = None,
    use_grid_search: bool | None = None,
    objective: str = "thrust",
    efficiency_alpha: float = 0.5,
    base_sample_idx: int = 0,
) -> dict:
    ckpt_path = checkpoint_path or c.best_checkpoint
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    save_only_outcome = c.optimise_save_only_outcome if save_only_outcome is None else save_only_outcome
    if use_grid_search is None:
        use_grid_search = c.optimise_use_grid_search
    use_grid_search = bool(use_grid_search) and (not save_only_outcome)

    device = c.resolve_device()
    model, ckpt = load_checkpoint(Path(ckpt_path), device)
    norm_stats = ckpt["norm_stats"]
    scalar_scale = np.array(norm_stats["scalar_scale"], dtype=np.float32)
    field_scale = np.array(norm_stats["field_scale"], dtype=np.float32)

    print(f"[optimise] Checkpoint: {ckpt_path}")
    print(f"[optimise] Device:     {device}")
    print(f"[optimise] save_only_outcome={save_only_outcome} use_grid_search={use_grid_search}")

    ds = CFDDataset(data_root=c.data_root, config=c)
    base_item = ds[base_sample_idx]
    print(f"[optimise] Base sample index: {base_sample_idx} (of {len(ds)})")
    base_meta: dict[str, object] = {"sample_index": int(base_sample_idx)}
    manifest = c.data_root / "manifest.json"
    if manifest.exists():
        try:
            rows = json.loads(manifest.read_text(encoding="utf-8"))
            if isinstance(rows, list) and 0 <= base_sample_idx < len(rows) and isinstance(rows[base_sample_idx], dict):
                row = rows[base_sample_idx]
                case_dir = str(row.get("case_dir", ""))
                geometry = str(row.get("geometry", "UNKNOWN")).upper()
                base_meta["case_dir"] = case_dir
                base_meta["geometry"] = geometry
                print(f"[optimise] Base sample meta: case={Path(case_dir).name}  geometry={geometry}")
        except Exception as e:
            print(f"[optimise][warn] Could not parse manifest metadata: {e}")

    base_pts = base_item["geometry_points"].numpy().T.copy()
    mid_xyz = base_item["midplane_xyz"].unsqueeze(0).to(device)
    cond = base_item["conditions"].unsqueeze(0).to(device)

    out_dir = c.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[optimise] Running differential_evolution (objective={objective}) ...")
    print(f"[optimise] bounds={c.dv_bounds} popsize={c.de_popsize} maxiter={c.de_maxiter}")
    print(
        f"[optimise] uncertainty: samples={c.optimise_uncertainty_samples} "
        f"beta={c.optimise_uncertainty_beta:.3f}  efficiency_alpha={efficiency_alpha:.3f}"
    )

    counter = [0]
    surrogate_obj = SurrogateObjective(
        model,
        base_pts,
        mid_xyz,
        cond,
        scalar_scale,
        device,
        counter,
        mc_samples=c.optimise_uncertainty_samples,
        uncertainty_beta=c.optimise_uncertainty_beta,
        objective=objective,
        efficiency_alpha=efficiency_alpha,
    )

    t0 = time.time()
    de_result = differential_evolution(
        surrogate_obj,
        bounds=c.dv_bounds,
        popsize=c.de_popsize,
        maxiter=c.de_maxiter,
        seed=c.de_seed,
        tol=c.de_tol,
        mutation=c.de_mutation,
        recombination=c.de_recombination,
        disp=True,
        workers=1,
    )
    de_time = time.time() - t0

    opt_dvs = np.asarray(de_result.x, dtype=np.float32)
    opt_pts = morph_geometry(base_pts, *opt_dvs)

    pred_scalar_opt, pred_field_opt = predict_from_geometry(
        model=model,
        points=opt_pts,
        mid_xyz=mid_xyz,
        conditions=cond,
        scalar_scale=scalar_scale,
        field_scale=field_scale,
        device=device,
    )
    pred_scalar_base, _ = predict_from_geometry(
        model=model,
        points=base_pts,
        mid_xyz=mid_xyz,
        conditions=cond,
        scalar_scale=scalar_scale,
        field_scale=field_scale,
        device=device,
    )

    baseline_scalar = base_item["scalar_raw"].numpy().astype(np.float32)  # CFD reference
    baseline_drag = float(baseline_scalar[0])
    baseline_thrust = float(baseline_scalar[1])
    baseline_pred_drag = float(pred_scalar_base[0])
    baseline_pred_thrust = float(pred_scalar_base[1])
    opt_drag = float(pred_scalar_opt[0])
    opt_thrust = float(pred_scalar_opt[1])

    # Predicted gain must be computed against predicted baseline (same model space).
    improvement_pct = 0.0
    if abs(baseline_pred_thrust) > 1e-12:
        improvement_pct = 100.0 * (opt_thrust / baseline_pred_thrust - 1.0)
    improvement_pct_vs_cfd = 0.0
    if abs(baseline_thrust) > 1e-12:
        improvement_pct_vs_cfd = 100.0 * (opt_thrust / baseline_thrust - 1.0)

    base_geo_t = torch.from_numpy(base_pts.T).float().unsqueeze(0).to(device)
    opt_geo_t  = torch.from_numpy(opt_pts.T).float().unsqueeze(0).to(device)
    _, _, base_thrust_mean, base_thrust_std = surrogate_obj.predict_scalar_stats(base_geo_t)
    _, _, opt_thrust_mean,  opt_thrust_std  = surrogate_obj.predict_scalar_stats(opt_geo_t)
    base_robust = base_thrust_mean - float(c.optimise_uncertainty_beta) * base_thrust_std
    opt_robust = opt_thrust_mean - float(c.optimise_uncertainty_beta) * opt_thrust_std
    robust_gain_pct = 0.0
    if abs(base_robust) > 1e-12:
        robust_gain_pct = 100.0 * (opt_robust / base_robust - 1.0)

    print(
        f"[optimise] DE done in {de_time:.1f}s, evals={counter[0]}, "
        f"opt_thrust={opt_thrust:.5f} N, pred_baseline={baseline_pred_thrust:.5f} N, "
        f"pred_gain={improvement_pct:.2f}% (vs CFD baseline: {improvement_pct_vs_cfd:.2f}%) "
        f"robust_gain={robust_gain_pct:.2f}%"
    )

    results: dict[str, object] = {
        "de_result": {
            "success": bool(de_result.success),
            "message": str(de_result.message),
            "n_evaluations": int(counter[0]),
            "elapsed_s": float(de_time),
            "opt_dvs": opt_dvs.tolist(),
            "opt_thrust_N": float(opt_thrust),
        },
        "baseline": {
            "drag_N": baseline_drag,
            "thrust_N": baseline_thrust,
            "sample": base_meta,
        },
        "baseline_prediction": {
            "drag_N": baseline_pred_drag,
            "thrust_N": baseline_pred_thrust,
        },
        "optimised_prediction": {
            "drag_N": opt_drag,
            "thrust_N": opt_thrust,
        },
        "improvement_pct": float(improvement_pct),
        "improvement_pct_vs_cfd_baseline": float(improvement_pct_vs_cfd),
        "uncertainty_objective": {
            "samples": int(c.optimise_uncertainty_samples),
            "beta": float(c.optimise_uncertainty_beta),
            "baseline_thrust_mean_N": float(base_thrust_mean),
            "baseline_thrust_std_N": float(base_thrust_std),
            "baseline_robust_score_N": float(base_robust),
            "opt_thrust_mean_N": float(opt_thrust_mean),
            "opt_thrust_std_N": float(opt_thrust_std),
            "opt_robust_score_N": float(opt_robust),
            "robust_gain_pct": float(robust_gain_pct),
        },
    }

    if use_grid_search:
        r = c.grid_resolution
        print(f"[optimise] Running {r}^3 grid evaluations ...")

        dv_axes = [np.linspace(b[0], b[1], r) for b in c.dv_bounds]
        dv0_g, dv1_g, dv2_g = np.meshgrid(*dv_axes, indexing="ij")
        dv_grid = np.stack([dv0_g.ravel(), dv1_g.ravel(), dv2_g.ravel()], axis=1)

        t1 = time.time()
        grid_drag_vals, grid_thrust_vals = eval_grid_batch(
            model=model,
            base_pts=base_pts,
            dv_grid=dv_grid,
            mid_xyz=mid_xyz,
            conditions=cond,
            scalar_scale=scalar_scale,
            device=device,
            batch_size=c.grid_batch_size,
        )
        grid_time = time.time() - t1

        # Objective scores for grid (for landscape best-point selection)
        grid_scores = _objective_score(
            grid_drag_vals, grid_thrust_vals, objective, efficiency_alpha
        )
        thrust_cube = grid_thrust_vals.reshape(r, r, r)
        score_cube  = grid_scores.reshape(r, r, r)
        max_grid_score = float(score_cube.max())
        max_idx = np.unravel_index(int(score_cube.argmax()), score_cube.shape)
        best_grid_dvs = np.array([dv_axes[i][max_idx[i]] for i in range(3)], dtype=np.float32)
        best_grid_thrust = float(thrust_cube[max_idx])
        best_grid_drag   = float(grid_drag_vals.reshape(r,r,r)[max_idx])

        results["grid_result"] = {
            "resolution": int(r),
            "n_evaluations": int(r ** 3),
            "elapsed_s": float(grid_time),
            "best_dvs": best_grid_dvs.tolist(),
            "best_thrust_N": best_grid_thrust,
            "best_drag_N":   best_grid_drag,
            "best_score":    max_grid_score,
            "objective":     objective,
        }

        # Save landscape npz (thrust cube used by visualise.py)
        np.savez_compressed(
            out_dir / "thrust_landscape.npz",
            thrust_cube=thrust_cube.astype(np.float32),
            drag_cube=grid_drag_vals.reshape(r, r, r).astype(np.float32),
            dv0_axis=np.array(dv_axes[0], dtype=np.float32),
            dv1_axis=np.array(dv_axes[1], dtype=np.float32),
            dv2_axis=np.array(dv_axes[2], dtype=np.float32),
        )
        print(f"[optimise] Saved thrust_landscape.npz  ({r}^3 = {r**3} points)")

        if not save_only_outcome:
            plot_landscape(
                dv_axes=dv_axes,
                thrust_cube=thrust_cube,
                opt_dvs=opt_dvs,
                save_path=out_dir / "thrust_landscape.png",
            )
            # Pareto scatter: thrust vs drag trade-off across all grid points
            plot_pareto_scatter(
                drag_vals=grid_drag_vals,
                thrust_vals=grid_thrust_vals,
                opt_drag=opt_drag,
                opt_thrust=opt_thrust,
                base_drag=baseline_pred_drag,
                base_thrust=baseline_pred_thrust,
                objective=objective,
                save_path=out_dir / "pareto_scatter.png",
            )

    np.save(out_dir / "optimal_geometry.npy", opt_pts.astype(np.float32))

    # Always save the best-thrust geometry visualisation as a publication figure.
    is_body = opt_pts[:, 6] > 0.5
    is_edf = opt_pts[:, 7] > 0.5
    plot_pointcloud(
        opt_pts[is_body, :3],
        opt_pts[is_edf, :3],
        title=(
            f"Best-thrust geometry  DV0={opt_dvs[0]:.3f} "
            f"DV1={opt_dvs[1]:.3f} DV2={opt_dvs[2]:.3f}  "
            f"Pred thrust={opt_thrust:.4f} N"
        ),
        save_path=out_dir / "optimal_geometry_best_thrust.png",
    )
    plot_initial_final_pointcloud(
        initial_pts=base_pts,
        optimised_pts=opt_pts,
        save_path=out_dir / "initial_vs_optimal_geometry.png",
    )

    outcome_npz = out_dir / "optimisation_outcome.npz"
    np.savez_compressed(
        outcome_npz,
        geometry_points=opt_pts.astype(np.float32),
        design_variables=opt_dvs.astype(np.float32),
        predicted_scalar=pred_scalar_opt.astype(np.float32),
        baseline_predicted_scalar=pred_scalar_base.astype(np.float32),
        predicted_midplane_fields=pred_field_opt.astype(np.float32),
        baseline_scalar=baseline_scalar.astype(np.float32),
        midplane_xyz=base_item["midplane_xyz"].numpy().T.astype(np.float32),
        conditions=base_item["conditions"].numpy().astype(np.float32),
    )

    results_path = out_dir / "optimisation_outcome.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[optimise] Saved: {outcome_npz}")
    print(f"[optimise] Saved: {results_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DGCNN geometry optimisation")
    parser.add_argument("--checkpoint", type=str, default=str(cfg.best_checkpoint))
    parser.add_argument("--data_root", type=str, default=str(cfg.data_root))
    parser.add_argument("--output_dir", type=str, default=str(cfg.output_dir))
    parser.add_argument("--grid_res", type=int, default=cfg.grid_resolution)
    parser.add_argument("--uncertainty_samples", type=int, default=cfg.optimise_uncertainty_samples)
    parser.add_argument("--uncertainty_beta", type=float, default=cfg.optimise_uncertainty_beta)
    parser.add_argument("--save_only_outcome", action="store_true")
    parser.add_argument("--no_grid_search", action="store_true")
    args = parser.parse_args()

    c = Config(
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
        grid_resolution=args.grid_res,
        optimise_uncertainty_samples=args.uncertainty_samples,
        optimise_uncertainty_beta=args.uncertainty_beta,
        optimise_save_only_outcome=args.save_only_outcome,
        optimise_use_grid_search=not args.no_grid_search,
    )

    optimise(
        c,
        checkpoint_path=Path(args.checkpoint),
        save_only_outcome=args.save_only_outcome,
        use_grid_search=not args.no_grid_search,
    )
