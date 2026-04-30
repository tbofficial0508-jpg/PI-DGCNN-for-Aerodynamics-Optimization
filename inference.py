from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path as MplPath
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.spatial import ConvexHull
import torch

from config import Config, cfg
from dataset import CFDDataset
from plotting_utils import apply_publication_style, panel_label, save_figure_png_pdf
from train import load_checkpoint


apply_publication_style()

# Solid-body colour: medium grey, clearly distinct from both dark-blue far-field
# and bright near-body flow. Used as the masked (NaN) colour in imshow.
_BODY_GREY = "0.78"


def _body_mask_from_npz(
    npz_path: str,
    Xi: np.ndarray,
    Yi: np.ndarray,
    y_eps: float = 0.025,
) -> np.ndarray:
    """
    Return a boolean grid mask where True = solid body / EDF nacelle.

    Loads dense surface geometry from a dataset .npz file, extracts the
    midplane cross-section (|y| < y_eps metres), computes the convex hull
    of the xz projection, and marks all grid cells inside as solid.

    This is geometrically exact and does not depend on flow-data sparsity.
    """
    mask = np.zeros(Xi.shape, dtype=bool)
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception:
        return mask

    grid_pts = np.column_stack([Xi.ravel(), Yi.ravel()])

    for key in ("dense_body_points", "dense_edf_points"):
        if key not in data:
            continue
        pts3d = data[key]                                          # (N, 3) metres
        pts_mid = pts3d[np.abs(pts3d[:, 1]) < y_eps][:, [0, 2]]  # xz only
        if len(pts_mid) < 4:
            continue
        try:
            hull = ConvexHull(pts_mid)
            hpath = MplPath(pts_mid[hull.vertices])
            inside = hpath.contains_points(grid_pts).reshape(Xi.shape)
            mask |= inside
        except Exception:
            pass

    return mask


def denorm(arr: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return arr * scale


def _sample_index_from_name(stem: str, prefix: str) -> int | None:
    if not stem.startswith(prefix):
        return None
    tail = stem[len(prefix):]
    digits = []
    for ch in tail:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    if not digits:
        return None
    return int("".join(digits))


def cleanup_sample_plots(
    output_dir: Path,
    sample_prefix: str,
    keep_indices: set[int],
) -> None:
    """
    Remove previously generated sample plots not in keep_indices.
    This keeps first/last-only mode from accumulating stale files.
    """
    if not output_dir.exists():
        return
    for suffix in (".png", ".pdf"):
        for path in output_dir.glob(f"{sample_prefix}*{suffix}"):
            idx = _sample_index_from_name(path.stem, sample_prefix)
            if idx is None or idx in keep_indices:
                continue
            try:
                path.unlink(missing_ok=True)
            except PermissionError:
                # OneDrive-synced files can carry read-only flags; keep cleanup non-fatal.
                try:
                    path.chmod(0o666)
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            except OSError:
                pass


@torch.no_grad()
def predict_sample(
    model,
    item: dict,
    device: torch.device,
    norm_stats: dict,
) -> dict:
    """
    Run the model on one dataset item and return predictions in physical units.
    """
    scalar_scale = np.array(norm_stats["scalar_scale"], dtype=np.float32)
    field_scale = np.array(norm_stats["field_scale"], dtype=np.float32)

    # Midplane coordinates are normalised in the dataset. We denormalise one level
    # (dataset-wide normalisation) for plotting but they remain in normalised geometry
    # coordinates, not absolute CAD units.
    mid_center = np.array(norm_stats["mid_xyz_center"], dtype=np.float32)
    mid_scale = np.array(norm_stats["mid_xyz_scale"], dtype=np.float32)

    geo = item["geometry_points"].unsqueeze(0).to(device)
    mid_xyz = item["midplane_xyz"].unsqueeze(0).to(device)
    cond = item["conditions"].unsqueeze(0).to(device)

    model.eval()
    scalar_n, field_n = model(geo, mid_xyz, cond)

    scalar_pred = denorm(scalar_n.squeeze(0).cpu().numpy(), scalar_scale)
    scalar_true = item["scalar_raw"].numpy()

    field_pred = denorm(
        field_n.squeeze(0).cpu().numpy(),
        field_scale[:, None],
    )
    field_true = denorm(
        item["midplane_fields"].numpy(),
        field_scale[:, None],
    )

    mid_xyz_np = item["midplane_xyz"].numpy().T
    mid_xyz_plot = mid_xyz_np * mid_scale[None, :] + mid_center[None, :]

    return {
        "scalar_pred": scalar_pred,
        "scalar_true": scalar_true,
        "field_pred": field_pred,
        "field_true": field_true,
        "mid_xyz": mid_xyz_plot,
    }


def _r2_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float("nan") if ss_tot <= 1e-12 else float(1.0 - ss_res / ss_tot)
    return r2, rmse


def _merge_duplicate_points(
    x_coord: np.ndarray,
    y_coord: np.ndarray,
    true_values: np.ndarray,
    pred_values: np.ndarray,
    decimals: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Merge duplicate (x, y) locations by averaging field values.
    This reduces local triangulation artifacts from repeated nodes.
    """
    x_r = np.round(x_coord.astype(np.float64), decimals=decimals)
    y_r = np.round(y_coord.astype(np.float64), decimals=decimals)
    coords = np.column_stack([x_r, y_r])

    uniq, inv = np.unique(coords, axis=0, return_inverse=True)
    n_unique = uniq.shape[0]

    counts = np.bincount(inv, minlength=n_unique).astype(np.float64)
    sum_true = np.bincount(inv, weights=true_values.astype(np.float64), minlength=n_unique)
    sum_pred = np.bincount(inv, weights=pred_values.astype(np.float64), minlength=n_unique)

    x_m = uniq[:, 0].astype(np.float32)
    y_m = uniq[:, 1].astype(np.float32)
    true_m = (sum_true / np.maximum(counts, 1.0)).astype(np.float32)
    pred_m = (sum_pred / np.maximum(counts, 1.0)).astype(np.float32)
    return x_m, y_m, true_m, pred_m


def _build_masked_triangulation(
    x_coord: np.ndarray,
    y_coord: np.ndarray,
    use_long_edge_mask: bool = True,
) -> mtri.Triangulation:
    """
    Build triangulation and mask poor/long triangles to avoid visual artifacts.
    """
    tri = mtri.Triangulation(x_coord, y_coord)
    pts = np.column_stack([x_coord, y_coord])
    tri_idx = tri.triangles

    p0 = pts[tri_idx[:, 0]]
    p1 = pts[tri_idx[:, 1]]
    p2 = pts[tri_idx[:, 2]]
    e01 = np.linalg.norm(p0 - p1, axis=1)
    e12 = np.linalg.norm(p1 - p2, axis=1)
    e20 = np.linalg.norm(p2 - p0, axis=1)
    max_edge = np.maximum(np.maximum(e01, e12), e20)

    edge_threshold = float(np.percentile(max_edge, 99.7))
    if not np.isfinite(edge_threshold) or edge_threshold <= 0:
        edge_threshold = float(np.percentile(max_edge, 99.0))

    analyzer = mtri.TriAnalyzer(tri)
    flat_mask = analyzer.get_flat_tri_mask(min_circle_ratio=0.01)
    if use_long_edge_mask:
        long_mask = max_edge > edge_threshold
    else:
        long_mask = np.zeros_like(flat_mask, dtype=bool)
    mask = flat_mask | long_mask

    # Guard against over-masking on highly non-uniform point clouds.
    keep_fraction = float(np.mean(~mask))
    if keep_fraction < 0.35:
        mask = flat_mask

    tri.set_mask(mask)
    return tri


def _append_domain_anchors(
    x_coord: np.ndarray,
    y_coord: np.ndarray,
    true_values: np.ndarray,
    pred_values: np.ndarray,
    domain: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Add sparse anchor points on the plotting boundary so tricontourf fills
    the full requested zoom window without corner omissions.
    Anchor values are taken from nearest existing points.
    """
    anchors = np.array(
        [
            [domain["xmin"], domain["ymin"]],
            [domain["xmin"], domain["ymax"]],
            [domain["xmax"], domain["ymin"]],
            [domain["xmax"], domain["ymax"]],
            [domain["xmin"], 0.5 * (domain["ymin"] + domain["ymax"])],
            [domain["xmax"], 0.5 * (domain["ymin"] + domain["ymax"])],
            [0.5 * (domain["xmin"] + domain["xmax"]), domain["ymin"]],
            [0.5 * (domain["xmin"] + domain["xmax"]), domain["ymax"]],
        ],
        dtype=np.float32,
    )

    pts = np.column_stack([x_coord, y_coord]).astype(np.float32)
    keep_anchor = np.ones(anchors.shape[0], dtype=bool)
    for i, a in enumerate(anchors):
        d2_min = float(np.min(np.sum((pts - a[None, :]) ** 2, axis=1)))
        if d2_min < 1e-10:
            keep_anchor[i] = False
    anchors = anchors[keep_anchor]
    if anchors.size == 0:
        return x_coord, y_coord, true_values, pred_values

    diff = pts[:, None, :] - anchors[None, :, :]
    nearest_idx = np.argmin(np.sum(diff * diff, axis=2), axis=0)
    true_anchor = true_values[nearest_idx]
    pred_anchor = pred_values[nearest_idx]

    x_aug = np.concatenate([x_coord, anchors[:, 0]], axis=0)
    y_aug = np.concatenate([y_coord, anchors[:, 1]], axis=0)
    true_aug = np.concatenate([true_values, true_anchor], axis=0)
    pred_aug = np.concatenate([pred_values, pred_anchor], axis=0)
    return x_aug, y_aug, true_aug, pred_aug


def _robust_bounds(values: np.ndarray, q_low: float = 1.0, q_high: float = 99.0) -> tuple[float, float]:
    lo = float(np.percentile(values, q_low))
    hi = float(np.percentile(values, q_high))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(values))
        hi = float(np.max(values))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def _plot_field_comparison(
    x_coord: np.ndarray,
    y_coord: np.ndarray,
    true_values: np.ndarray,
    pred_values: np.ndarray,
    variable_name: str,
    unit: str,
    y_axis_label: str,
    save_path: Path,
    npz_path: str | None = None,
) -> None:
    zoom_limits = {
        "xmin": -2.0,
        "xmax": 2.0,
        "ymin": -2.0,
        "ymax": 2.0,
    }

    resid = pred_values - true_values
    zoom_mask = (
        (x_coord >= zoom_limits["xmin"]) & (x_coord <= zoom_limits["xmax"]) &
        (y_coord >= zoom_limits["ymin"]) & (y_coord <= zoom_limits["ymax"])
    )
    if int(np.count_nonzero(zoom_mask)) >= 8:
        true_eval = true_values[zoom_mask]
        pred_eval = pred_values[zoom_mask]
    else:
        true_eval = true_values
        pred_eval = pred_values
    r2, rmse = _r2_rmse(true_eval, pred_eval)

    fig = plt.figure(figsize=(12.8, 5.0), constrained_layout=True)
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.0, 1.0, 1.0],
        height_ratios=[1.0, 0.12],
    )

    ax_gt = fig.add_subplot(gs[0, 0])
    ax_pr = fig.add_subplot(gs[0, 1])
    ax_rs = fig.add_subplot(gs[0, 2])
    cax_main = fig.add_subplot(gs[1, 0:2])
    cax_res = fig.add_subplot(gs[1, 2])

    combined = np.concatenate([true_values, pred_values], axis=0)
    vmin, vmax = _robust_bounds(combined, q_low=1.0, q_high=99.0)
    main_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    res_absmax = float(np.percentile(np.abs(resid), 99.0))
    if not np.isfinite(res_absmax) or res_absmax < 1e-12:
        res_absmax = 1e-12
    res_norm = mcolors.TwoSlopeNorm(vmin=-res_absmax, vcenter=0.0, vmax=res_absmax)

    # Build regular grid spanning the fixed zoom window
    _N = 600
    xi = np.linspace(zoom_limits["xmin"], zoom_limits["xmax"], _N)
    yi = np.linspace(zoom_limits["ymin"], zoom_limits["ymax"], _N)
    Xi, Yi = np.meshgrid(xi, yi)
    pts = (x_coord, y_coord)

    def _to_grid(values):
        # Linear: monotone, no Gibbs ringing at steep wall gradients
        return griddata(pts, values, (Xi, Yi), method="linear")

    Zi_true  = _to_grid(true_values)
    Zi_pred  = _to_grid(pred_values)
    Zi_resid = _to_grid(resid)

    # Compute body mask first so we can use it during smoothing
    body_mask = np.zeros(Xi.shape, dtype=bool)
    if npz_path is not None:
        _bm = _body_mask_from_npz(npz_path, Xi, Yi)
        if _bm.any():
            body_mask = _bm

    def _smooth(Zi):
        """
        Blur without contaminating fluid from the zero-velocity wall BC.

        Strategy:
          1. Temporarily fill the body interior with the nearest *fluid*
             neighbour value (nearest-neighbour extrapolation into the mask).
             This means the Gaussian kernel at the body boundary sees
             fluid-like values on both sides, not the wall BC zero.
          2. Apply Gaussian blur.
          3. Re-apply the body mask (NaN) to restore the grey solid region.
        """
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

    # Colormaps with grey NaN colour (solid body region)
    cmap_main = plt.cm.viridis.copy()
    cmap_main.set_bad(color=_BODY_GREY)
    cmap_err = plt.cm.coolwarm.copy()
    cmap_err.set_bad(color=_BODY_GREY)
    for ax in (ax_gt, ax_pr, ax_rs):
        ax.set_facecolor(_BODY_GREY)

    _extent = [zoom_limits["xmin"], zoom_limits["xmax"],
               zoom_limits["ymin"], zoom_limits["ymax"]]
    _ikw = dict(origin="lower", extent=_extent, interpolation="bilinear", aspect="equal")

    cfd_map = ax_gt.imshow(Zi_true,  cmap=cmap_main, norm=main_norm, **_ikw)
    ax_pr.imshow(Zi_pred,  cmap=cmap_main, norm=main_norm, **_ikw)
    err_map = ax_rs.imshow(Zi_resid, cmap=cmap_err,  norm=res_norm,  **_ikw)

    for ax in (ax_gt, ax_pr, ax_rs):
        ax.set_xlim(zoom_limits["xmin"], zoom_limits["xmax"])
        ax.set_ylim(zoom_limits["ymin"], zoom_limits["ymax"])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [norm.]")
        ax.set_ylabel(y_axis_label)

    role_box = dict(
        boxstyle="round,pad=0.17",
        facecolor="white",
        alpha=0.85,
        edgecolor="0.65",
        linewidth=0.4,
    )
    ax_gt.text(0.02, 0.98, "CFD data", transform=ax_gt.transAxes, ha="left", va="top", bbox=role_box)
    ax_pr.text(0.02, 0.98, "Prediction", transform=ax_pr.transAxes, ha="left", va="top", bbox=role_box)
    ax_rs.text(0.02, 0.98, "Error", transform=ax_rs.transAxes, ha="left", va="top", bbox=role_box)

    _true_range = float(np.max(true_eval) - np.min(true_eval))
    _rob_lo, _rob_hi = _robust_bounds(true_eval, 1.0, 99.0)
    _rob_range = float(_rob_hi - _rob_lo)
    _std_true = float(np.std(true_eval))
    _nrmse_pct = 100.0 * rmse / _true_range if _true_range > 1e-12 else float("nan")
    _nrmse_rob_pct = 100.0 * rmse / _rob_range if _rob_range > 1e-12 else float("nan")
    _cvrmse_pct = 100.0 * rmse / _std_true if _std_true > 1e-12 else float("nan")
    _fit_flag = "POOR FIT (R^2<0)" if np.isfinite(r2) and r2 < 0.0 else "Fit OK"
    metric_text = (
        f"$R^2$ = {r2:.4f} ({_fit_flag})\n"
        f"RMSE = {rmse:.3e} {unit}\n"
        f"nRMSE(range) = {_nrmse_pct:.2f}%\n"
        f"nRMSE(p99-p1) = {_nrmse_rob_pct:.2f}%\n"
        f"CVRMSE = {_cvrmse_pct:.2f}%"
    )
    ax_rs.text(
        0.98,
        0.94,
        metric_text,
        transform=ax_rs.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white", alpha=0.85, edgecolor="0.6", linewidth=0.4),
    )

    panel_label(ax_gt, "a")
    panel_label(ax_pr, "b")
    panel_label(ax_rs, "c")

    cb_main = fig.colorbar(cfd_map, cax=cax_main, orientation="horizontal")
    cb_main.set_label(f"{variable_name} [{unit}]")
    cb_main.ax.tick_params(labelsize=10)

    cb_res = fig.colorbar(err_map, cax=cax_res, orientation="horizontal")
    cb_res.set_label(f"Error {variable_name} [{unit}]")
    res_tick_values = np.linspace(-res_absmax, res_absmax, 5)
    cb_res.set_ticks(res_tick_values)
    cb_res.set_ticklabels([f"{tick:.2g}" for tick in res_tick_values])
    cb_res.ax.tick_params(labelsize=9)

    png_path, pdf_path = save_figure_png_pdf(fig, save_path)
    plt.close(fig)
    print(f"Field plot saved: {png_path}")
    print(f"Field plot saved: {pdf_path}")


def plot_fields(
    result: dict,
    save_path: Path,
    sample_idx: int = 0,
    data_root: str | None = None,
) -> None:
    """
    Save per-field publication figures:
      top row: ground truth vs prediction (shared colourbar)
      bottom: residual with R^2 and RMSE annotation
    """
    coords = result["mid_xyz"]
    x_coord = coords[:, 0]
    span_y = float(np.ptp(coords[:, 1]))
    span_z = float(np.ptp(coords[:, 2]))
    if span_y >= span_z:
        y_coord = coords[:, 1]
        y_axis_label = "y [norm.]"
    else:
        y_coord = coords[:, 2]
        y_axis_label = "z [norm.]"

    # Use the matching sample geometry for masking (not always sample_000).
    npz_path: str | None = None
    if data_root is not None:
        npz_files = sorted(Path(data_root).glob("sample_*.npz"))
        if npz_files:
            _npz_idx = min(max(int(sample_idx), 0), len(npz_files) - 1)
            npz_path = str(npz_files[_npz_idx])

    fields = [
        ("pressure", "Pressure", "Pa"),
        ("u_velocity", "u-velocity", "m/s"),
        ("v_velocity", "v-velocity", "m/s"),
        ("w_velocity", "w-velocity", "m/s"),
    ]

    for i, (suffix, name, unit) in enumerate(fields):
        out_path = save_path.with_name(f"{save_path.stem}_{suffix}.png")
        _plot_field_comparison(
            x_coord=x_coord,
            y_coord=y_coord,
            true_values=result["field_true"][i],
            pred_values=result["field_pred"][i],
            variable_name=name,
            unit=unit,
            y_axis_label=y_axis_label,
            save_path=out_path,
            npz_path=npz_path,
        )


def print_scalar_errors(result: dict, sample_idx: int = 0) -> None:
    sp = result["scalar_pred"]
    st = result["scalar_true"]
    print(f"\nSample {sample_idx}")
    print(f"  {'':12s} {'GT':>12s} {'Pred':>12s} {'AbsErr':>12s} {'RelErr%':>10s}")
    near_zero_gt = False
    for name, gt, pred in zip(["drag (N)", "thrust (N)"], st, sp):
        aerr = abs(pred - gt)
        if abs(gt) < 1e-3:
            rel_txt = f"{'n/a*':>10s}"
            near_zero_gt = True
        else:
            rerr = 100.0 * aerr / abs(gt)
            rel_txt = f"{rerr:10.2f}%"
        print(f"  {name:12s} {gt:12.5f} {pred:12.5f} {aerr:12.5f} {rel_txt}")
    if near_zero_gt:
        print("  [note] RelErr% shown as n/a* when |GT| < 1e-3 N; use AbsErr for those rows.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference with trained DGCNN")
    parser.add_argument("--checkpoint", type=str, default=str(cfg.best_checkpoint))
    parser.add_argument("--data_root", type=str, default=str(cfg.data_root))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--all", action="store_true", help="Run on every sample in the dataset")
    parser.add_argument(
        "--plot_mode",
        type=str,
        default=cfg.eval_plot_mode,
        choices=["all", "first_last"],
        help="When running --all, choose whether to save all slice figures or only first/last sample figures.",
    )
    parser.add_argument("--output_dir", type=str, default=str(cfg.output_dir))
    parser.add_argument("--device", type=str, default=cfg.device)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model, ckpt = load_checkpoint(ckpt_path, device)
    norm_stats = ckpt["norm_stats"]

    print(f"[inference] Loaded checkpoint: {ckpt_path}")
    print(f"[inference] Checkpoint epoch:  {ckpt.get('epoch', '?')}")
    print(f"[inference] Best val metrics:  {ckpt.get('metrics', {})}")

    ds = CFDDataset(data_root=args.data_root)
    out_dir = Path(args.output_dir)

    indices = list(range(len(ds))) if args.all else [args.index]
    if args.all and args.plot_mode == "first_last" and len(indices) > 1:
        plot_indices = {indices[0], indices[-1]}
    else:
        plot_indices = set(indices)

    if args.all and args.plot_mode == "first_last":
        cleanup_sample_plots(out_dir, "inference_sample", plot_indices)

    for idx in indices:
        item = ds[idx]
        result = predict_sample(model, item, device, norm_stats)
        print_scalar_errors(result, idx)
        if idx in plot_indices:
            plot_fields(result, out_dir / f"inference_sample{idx:03d}.png", idx,
                        data_root=args.data_root)


if __name__ == "__main__":
    main()
