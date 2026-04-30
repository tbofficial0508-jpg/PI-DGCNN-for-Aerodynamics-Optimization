from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from plotting_utils import (
    BODY_COLOR,
    EDF_COLOR,
    apply_publication_style,
    panel_label,
    save_figure_png_pdf,
)


apply_publication_style()


def fps_numpy(points: np.ndarray, n_samples: int, seed: int = 0) -> np.ndarray:
    """
    Farthest Point Sampling on an (N, D) numpy array.
    Distances are computed on the first 3 columns (xyz).
    """
    n = points.shape[0]
    if n <= n_samples:
        return np.arange(n)

    rng = np.random.default_rng(seed)
    xyz = points[:, :3].astype(np.float32)

    selected = np.empty(n_samples, dtype=np.int64)
    selected[0] = int(rng.integers(0, n))

    dist2 = np.full(n, np.inf, dtype=np.float32)

    for i in range(1, n_samples):
        last = xyz[selected[i - 1]]
        d2 = np.sum((xyz - last) ** 2, axis=1)
        dist2 = np.minimum(dist2, d2)
        selected[i] = int(np.argmax(dist2))

    return selected


def fps_torch(points: torch.Tensor, n_samples: int) -> torch.Tensor:
    """
    Batched Farthest Point Sampling.

    points: (B, N, 3+) - uses only first 3 dims for distance.
    Returns sampled indices: (B, n_samples).
    """
    bsz, n_points, _ = points.shape
    device = points.device
    xyz = points[:, :, :3]

    selected = torch.zeros(bsz, n_samples, dtype=torch.long, device=device)
    dist = torch.full((bsz, n_points), float("inf"), device=device)
    selected[:, 0] = torch.randint(0, n_points, (bsz,), device=device)

    for i in range(1, n_samples):
        prev_idx = selected[:, i - 1]
        prev_pts = xyz[torch.arange(bsz), prev_idx, :]
        d2 = ((xyz - prev_pts.unsqueeze(1)) ** 2).sum(dim=2)
        dist = torch.minimum(dist, d2)
        selected[:, i] = dist.argmax(dim=1)

    return selected


def knn_graph(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Build a k-NN graph for x.

    x: (B, C, N)
    Returns: (B, N, k)
    """
    inner = torch.bmm(x.transpose(2, 1), x)
    xx = (x ** 2).sum(dim=1, keepdim=True)
    sq_dist = xx + xx.transpose(2, 1) - 2.0 * inner
    idx = (-sq_dist).topk(k=k + 1, dim=-1)[1][:, :, 1:]
    return idx


def plot_pointcloud(
    body_pts: np.ndarray,
    edf_pts: np.ndarray,
    title: str = "Geometry point cloud",
    save_path: Optional[str | Path] = None,
    max_body: int = 30_000,
    max_edf: int = 12_000,
    body_size: float = 0.55,
    edf_size: float = 0.80,
    body_alpha: float = 0.95,
    edf_alpha: float = 0.98,
    seed: int = 1234,
) -> None:
    """
    Publication-style geometry rendering.

    Layout and view settings are matched to the existing publication contrast style:
      - 5 panels (isometric + top/side/front + planform detail)
      - elev=22, azim=-58 for isometric panel
      - blue body and orange EDF with fixed marker sizes/alphas
    """
    rng = np.random.default_rng(seed)

    def _subsample(points: np.ndarray, max_points: int) -> np.ndarray:
        if max_points <= 0 or points.shape[0] <= max_points:
            return points
        idx = rng.choice(points.shape[0], size=max_points, replace=False)
        return points[idx]

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

    def _set_equal_2d(ax, all_points: np.ndarray, i: int, j: int, pad_frac: float = 0.05) -> None:
        mins = all_points[:, [i, j]].min(axis=0)
        maxs = all_points[:, [i, j]].max(axis=0)
        center = 0.5 * (mins + maxs)
        span = max(float((maxs - mins).max()), 1e-6)
        half = 0.5 * span * (1.0 + pad_frac)

        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_aspect("equal", adjustable="box")

    def _style_3d_axes(ax) -> None:
        ax.grid(True)
        ax.set_facecolor("#d8d8d8")
        ax.set_xlabel("x [norm.]")
        ax.set_ylabel("y [norm.]")
        ax.set_zlabel("z [norm.]")

        try:
            ax.xaxis._axinfo["grid"]["color"] = (0.45, 0.45, 0.45, 0.8)
            ax.yaxis._axinfo["grid"]["color"] = (0.45, 0.45, 0.45, 0.8)
            ax.zaxis._axinfo["grid"]["color"] = (0.45, 0.45, 0.45, 0.8)
            ax.xaxis._axinfo["grid"]["linewidth"] = 0.7
            ax.yaxis._axinfo["grid"]["linewidth"] = 0.7
            ax.zaxis._axinfo["grid"]["linewidth"] = 0.7
        except Exception:
            pass

        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            try:
                axis.pane.fill = True
                axis.pane.set_facecolor((0.86, 0.86, 0.86, 1.0))
                axis.pane.set_edgecolor((0.35, 0.35, 0.35, 1.0))
            except Exception:
                pass

    def _style_2d_axes(ax, xlabel: str, ylabel: str) -> None:
        ax.set_facecolor("#dddddd")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, color="0.45", linewidth=0.6, alpha=0.8)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color("0.15")

    body_view = _subsample(body_pts, max_body)
    edf_view = _subsample(edf_pts, max_edf)
    all_view = np.concatenate([body_view, edf_view], axis=0)

    fig = plt.figure(figsize=(16, 11), constrained_layout=True)
    fig.patch.set_facecolor("#cfcfcf")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.0, 1.0], height_ratios=[1.0, 1.0])

    # (a) Isometric view
    ax0 = fig.add_subplot(gs[:, 0], projection="3d")
    ax0.scatter(body_view[:, 0], body_view[:, 1], body_view[:, 2], s=body_size, c=BODY_COLOR, alpha=body_alpha, linewidths=0.0, rasterized=True, label="Body")
    ax0.scatter(edf_view[:, 0], edf_view[:, 1], edf_view[:, 2], s=edf_size, c=EDF_COLOR, alpha=edf_alpha, linewidths=0.0, rasterized=True, label="EDF")
    ax0.view_init(elev=22, azim=-58)
    _set_equal_3d(ax0, all_view)
    _style_3d_axes(ax0)
    legend = ax0.legend(loc="upper right", frameon=True)
    legend.get_frame().set_edgecolor("0.35")
    legend.get_frame().set_facecolor("#d9d9d9")
    legend.get_frame().set_alpha(1.0)
    panel_label(ax0, "a")

    # (b) Top view (x-y)
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.scatter(body_view[:, 0], body_view[:, 1], s=body_size, c=BODY_COLOR, alpha=body_alpha, linewidths=0.0, rasterized=True)
    ax1.scatter(edf_view[:, 0], edf_view[:, 1], s=edf_size, c=EDF_COLOR, alpha=edf_alpha, linewidths=0.0, rasterized=True)
    _set_equal_2d(ax1, all_view, 0, 1)
    _style_2d_axes(ax1, "x [norm.]", "y [norm.]")
    panel_label(ax1, "b")

    # (c) Side view (x-z)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.scatter(body_view[:, 0], body_view[:, 2], s=body_size, c=BODY_COLOR, alpha=body_alpha, linewidths=0.0, rasterized=True)
    ax2.scatter(edf_view[:, 0], edf_view[:, 2], s=edf_size, c=EDF_COLOR, alpha=edf_alpha, linewidths=0.0, rasterized=True)
    _set_equal_2d(ax2, all_view, 0, 2)
    _style_2d_axes(ax2, "x [norm.]", "z [norm.]")
    panel_label(ax2, "c")

    # (d) Front/rear view (y-z)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.scatter(body_view[:, 1], body_view[:, 2], s=body_size, c=BODY_COLOR, alpha=body_alpha, linewidths=0.0, rasterized=True)
    ax3.scatter(edf_view[:, 1], edf_view[:, 2], s=edf_size, c=EDF_COLOR, alpha=edf_alpha, linewidths=0.0, rasterized=True)
    _set_equal_2d(ax3, all_view, 1, 2)
    _style_2d_axes(ax3, "y [norm.]", "z [norm.]")
    panel_label(ax3, "d")

    # (e) Planform detail
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.scatter(body_view[:, 0], body_view[:, 1], s=0.35, c=BODY_COLOR, alpha=0.98, linewidths=0.0, rasterized=True)
    ax4.scatter(edf_view[:, 0], edf_view[:, 1], s=0.55, c=EDF_COLOR, alpha=0.98, linewidths=0.0, rasterized=True)
    _set_equal_2d(ax4, all_view, 0, 1)
    _style_2d_axes(ax4, "x [norm.]", "y [norm.]")
    panel_label(ax4, "e")

    if save_path is not None:
        png_path, pdf_path = save_figure_png_pdf(fig, save_path)
        plt.close(fig)
        print(f"Saved: {png_path}")
        print(f"Saved: {pdf_path}")
    else:
        plt.show()


def morph_geometry(
    points: np.ndarray,
    dv0_body_scale: float = 1.0,
    dv1_duct_scale: float = 1.0,
    dv2_edf_offset: float = 0.0,
) -> np.ndarray:
    """
    Apply parametric geometry perturbations to the combined surface point cloud
    and analytically correct the surface normals.

    points columns: [x, y, z, nx, ny, nz, is_body, is_edf]

    Normal correction uses the inverse-transpose of the deformation Jacobian.
    For a diagonal scaling J = diag(sx, sy, sz), the normal transforms as:
        n' = J^{-T} n = (nx/sx, ny/sy, nz/sz), then renormalised.

    DV0 (body x-scale sx):   J = diag(sx, 1, 1)  ->  n' = (nx/sx, ny, nz)
    DV1 (duct yz-scale sy):  J = diag(1, sy, sy) ->  n' = (nx, ny/sy, nz/sy)
    DV2 (EDF x-offset):      pure translation -> no normal change
    """
    pts = points.copy().astype(np.float32)
    is_body = pts[:, 6] > 0.5
    is_edf  = pts[:, 7] > 0.5

    if dv0_body_scale != 1.0 and is_body.any():
        cx = float(pts[is_body, 0].mean())
        pts[is_body, 0] = cx + (pts[is_body, 0] - cx) * dv0_body_scale
        # Correct normals: n_x scales by 1/dv0, n_y and n_z unchanged
        pts[is_body, 3] /= dv0_body_scale
        nrm = np.linalg.norm(pts[is_body, 3:6], axis=1, keepdims=True)
        pts[is_body, 3:6] /= np.maximum(nrm, 1e-8)

    if dv1_duct_scale != 1.0 and is_edf.any():
        cy = float(pts[is_edf, 1].mean())
        cz = float(pts[is_edf, 2].mean())
        pts[is_edf, 1] = cy + (pts[is_edf, 1] - cy) * dv1_duct_scale
        pts[is_edf, 2] = cz + (pts[is_edf, 2] - cz) * dv1_duct_scale
        # Correct normals: n_y and n_z scale by 1/dv1, n_x unchanged
        pts[is_edf, 4] /= dv1_duct_scale
        pts[is_edf, 5] /= dv1_duct_scale
        nrm = np.linalg.norm(pts[is_edf, 3:6], axis=1, keepdims=True)
        pts[is_edf, 3:6] /= np.maximum(nrm, 1e-8)

    if dv2_edf_offset != 0.0 and is_edf.any():
        pts[is_edf, 0] += dv2_edf_offset
        # Translation: Jacobian = I, normals unchanged

    return pts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise a dataset sample.")
    parser.add_argument("--sample", type=str, default="dataset_pointcloud/sample_00000.npz")
    parser.add_argument("--save_path", type=str, default="runs/pipeline/utils_test.png")
    args = parser.parse_args()

    with np.load(args.sample, allow_pickle=True) as npz:
        body = npz["dense_body_points"].astype(np.float32)
        edf = npz["dense_edf_points"].astype(np.float32)
        meta = json.loads(npz["meta"][0])

    print(f"Body points: {body.shape}, EDF points: {edf.shape}")
    print(f"Case: {meta.get('case_dir', 'unknown')}")

    plot_pointcloud(body, edf, title=Path(meta.get("case_dir", "")).name, save_path=args.save_path)
    print("Done.")
