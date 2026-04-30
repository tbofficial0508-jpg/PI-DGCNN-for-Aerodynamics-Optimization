"""
Side-by-side point-cloud comparison for the T1, J1, J2 geometries.

Produces a single publication-style figure with three columns (one per
geometry). Each column has an isometric view (top) and a planform / top-down
view (bottom). Body points are rendered blue, EDF points orange, matching the
existing visualize_geometry_publication.py styling.

Usage:
    python visualize_geometries_sidebyside.py \
        --dataset dataset_pointcloud \
        --save_path runs/geometry_comparison.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)


# Fixed mapping from geometry tag to the representative sample index in the
# regenerated dataset. T1 is case 1 (sample 0), J1 is case 2 (sample 1),
# J2 is case 18 (sample 17). Change via --sample_* if you relabel cases.
DEFAULT_SAMPLES = {"T1": 0, "J1": 1, "J2": 17}


def load_sample(ds_root: Path, idx: int):
    data = np.load(ds_root / f"sample_{idx:05d}.npz", allow_pickle=True)
    body = data["dense_body_points"].astype(np.float32)
    edf = data["dense_edf_points"].astype(np.float32)
    meta = json.loads(data["meta"][0]) if "meta" in data else {}
    return body, edf, meta


def subsample(pts: np.ndarray, n_max: int, rng: np.random.Generator) -> np.ndarray:
    if n_max <= 0 or pts.shape[0] <= n_max:
        return pts
    idx = rng.choice(pts.shape[0], size=n_max, replace=False)
    return pts[idx]


def set_equal_3d(ax, all_xyz: np.ndarray, pad_frac: float = 0.05) -> None:
    mins = all_xyz.min(axis=0)
    maxs = all_xyz.max(axis=0)
    c = 0.5 * (mins + maxs)
    span = max(float((maxs - mins).max()), 1e-6)
    h = 0.5 * span * (1.0 + pad_frac)
    ax.set_xlim(c[0] - h, c[0] + h)
    ax.set_ylim(c[1] - h, c[1] + h)
    ax.set_zlim(c[2] - h, c[2] + h)
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except Exception:
        pass


def style_3d(ax) -> None:
    ax.grid(True)
    ax.set_facecolor("#d8d8d8")
    ax.set_xlabel("x [norm.]")
    ax.set_ylabel("y [norm.]")
    ax.set_zlabel("z [norm.]")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.fill = True
            axis.pane.set_facecolor((0.86, 0.86, 0.86, 1.0))
            axis.pane.set_edgecolor((0.35, 0.35, 0.35, 1.0))
        except Exception:
            pass


def style_2d(ax, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor("#dddddd")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color="0.45", linewidth=0.6, alpha=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("0.15")
    ax.set_aspect("equal", adjustable="box")


BODY_COLOR = "#1f6fb5"
EDF_COLOR = "#e07b00"


def render_column(fig, gs_col, body, edf, label: str, meta: dict,
                  rng: np.random.Generator,
                  n_body: int, n_edf: int,
                  body_size: float, edf_size: float,
                  body_alpha: float, edf_alpha: float) -> None:
    body_v = subsample(body, n_body, rng)
    edf_v = subsample(edf, n_edf, rng)
    all_v = np.concatenate([body_v, edf_v], axis=0)

    # Row 0: isometric 3D
    ax3 = fig.add_subplot(gs_col[0], projection="3d")
    ax3.view_init(elev=22, azim=-58)
    ax3.scatter(body_v[:, 0], body_v[:, 1], body_v[:, 2],
                s=body_size, c=BODY_COLOR, alpha=body_alpha,
                linewidths=0, depthshade=True)
    ax3.scatter(edf_v[:, 0], edf_v[:, 1], edf_v[:, 2],
                s=edf_size, c=EDF_COLOR, alpha=edf_alpha,
                linewidths=0, depthshade=True)
    set_equal_3d(ax3, all_v)
    style_3d(ax3)

    drag = meta.get("body_drag", float("nan"))
    thrust = meta.get("edf_thrust", float("nan"))
    rpm = meta.get("rpm", float("nan"))
    u_inf = meta.get("u_inf", float("nan"))
    is_unconverged = (abs(drag) < 1e-3) and (abs(thrust) < 1e-3)
    status = "  [UNCONVERGED]" if is_unconverged else ""
    title = (f"{label}{status}\n"
             f"U$_\\infty$={u_inf:.0f} m/s   RPM={rpm:.0f}\n"
             f"drag={drag:.3g} N   thrust={thrust:.3g} N\n"
             f"body pts={body.shape[0]:,}   edf pts={edf.shape[0]:,}")
    ax3.set_title(title, fontsize=11)

    # Row 1: top-down (x vs z) projection
    ax2 = fig.add_subplot(gs_col[1])
    ax2.scatter(body_v[:, 0], body_v[:, 2],
                s=body_size * 0.9, c=BODY_COLOR, alpha=body_alpha,
                linewidths=0, label="body")
    ax2.scatter(edf_v[:, 0], edf_v[:, 2],
                s=edf_size * 0.9, c=EDF_COLOR, alpha=edf_alpha,
                linewidths=0, label="EDF")
    style_2d(ax2, "x [norm.]", "z [norm.]")
    if label == "T1":
        ax2.legend(loc="upper right", fontsize=9, framealpha=0.85)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="dataset_pointcloud")
    p.add_argument("--save_path", type=str, default="runs/geometry_comparison.png")
    p.add_argument("--sample_t1", type=int, default=DEFAULT_SAMPLES["T1"])
    p.add_argument("--sample_j1", type=int, default=DEFAULT_SAMPLES["J1"])
    p.add_argument("--sample_j2", type=int, default=DEFAULT_SAMPLES["J2"])
    p.add_argument("--max_body_points", type=int, default=30000)
    p.add_argument("--max_edf_points", type=int, default=12000)
    p.add_argument("--body_size", type=float, default=0.55)
    p.add_argument("--edf_size", type=float, default=0.85)
    p.add_argument("--body_alpha", type=float, default=0.95)
    p.add_argument("--edf_alpha", type=float, default=0.98)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    ds = Path(args.dataset)
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    picks = [("T1", args.sample_t1), ("J1", args.sample_j1), ("J2", args.sample_j2)]
    rng = np.random.default_rng(args.seed)

    fig = plt.figure(figsize=(18, 11), constrained_layout=True)
    fig.patch.set_facecolor("#cfcfcf")
    outer = fig.add_gridspec(1, 3, wspace=0.04)

    for col, (label, idx) in enumerate(picks):
        body, edf, meta = load_sample(ds, idx)
        inner = outer[col].subgridspec(2, 1, height_ratios=[1.55, 1.0], hspace=0.04)
        render_column(fig, inner, body, edf, label, meta, rng,
                      args.max_body_points, args.max_edf_points,
                      args.body_size, args.edf_size,
                      args.body_alpha, args.edf_alpha)

    fig.suptitle("Training-set geometries: T1, J1, J2 point clouds",
                 fontsize=14, fontweight="bold")

    for ext in (".png", ".pdf"):
        out = save_path.with_suffix(ext)
        fig.savefig(out, dpi=220, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
