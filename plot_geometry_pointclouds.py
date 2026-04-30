"""
plot_geometry_pointclouds.py
Publication-quality three-panel point-cloud figure: T1 / J1 / J2.
White background, no pane fills, no grey outlines.

Usage:
    python plot_geometry_pointclouds.py
    python plot_geometry_pointclouds.py --dataset dataset_pointcloud --out runs/geometry_pointclouds
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

matplotlib.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        9,
    "axes.labelsize":   8,
    "axes.titlesize":   10,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
    "figure.dpi":       150,
    "savefig.dpi":      300,
})

# ── colour palette ────────────────────────────────────────────────────────────
BODY_COL = "#1f77b4"   # matplotlib tab-blue
EDF_COL  = "#ff7f0e"   # matplotlib tab-orange

# ── geometry catalogue ────────────────────────────────────────────────────────
GEOMETRIES = [
    {
        "label":   "T1",
        "idx":     0,
        "n_body":  18_000,
        "n_edf":   12_000,
        "elev":    22,
        "azim":    -55,
    },
    {
        "label":   "J1",
        "idx":     1,
        "n_body":  18_000,
        "n_edf":   12_000,
        "elev":    22,
        "azim":    -55,
    },
    {
        "label":   "J2",
        "idx":     17,
        "n_body":  18_000,
        "n_edf":   12_000,
        "elev":    22,
        "azim":    -55,
    },
]


def load_sample(ds_root: Path, idx: int):
    data = np.load(ds_root / f"sample_{idx:05d}.npz", allow_pickle=True)
    body = data["dense_body_points"].astype(np.float32)
    edf  = data["dense_edf_points"].astype(np.float32)
    meta = json.loads(data["meta"][0]) if "meta" in data else {}
    return body, edf, meta


def subsample(pts: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0 or pts.shape[0] <= n:
        return pts
    return pts[rng.choice(pts.shape[0], size=n, replace=False)]


def normalise_pts(body: np.ndarray, edf: np.ndarray):
    """Centre and scale both clouds by the body bounding-box diagonal."""
    all_pts = np.concatenate([body, edf], axis=0)
    centre  = 0.5 * (all_pts.min(0) + all_pts.max(0))
    span    = float(np.linalg.norm(all_pts.max(0) - all_pts.min(0))) + 1e-9
    return (body - centre) / span, (edf - centre) / span


def equal_axes_3d(ax, body: np.ndarray, edf: np.ndarray, pad: float = 0.06):
    all_pts = np.concatenate([body, edf], axis=0)
    lo, hi  = all_pts.min(0), all_pts.max(0)
    centre  = 0.5 * (lo + hi)
    half    = 0.5 * float((hi - lo).max()) * (1.0 + pad)
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_zlim(centre[2] - half, centre[2] + half)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def clean_3d_axes(ax):
    """Remove all pane fills and grey outlines; keep only thin axis spines."""
    ax.set_facecolor("none")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((0.75, 0.75, 0.75, 0.4))
        axis.pane.set_linewidth(0.5)
    # Thin grid lines
    ax.xaxis._axinfo["grid"]["color"]     = (0.85, 0.85, 0.85, 0.6)
    ax.yaxis._axinfo["grid"]["color"]     = (0.85, 0.85, 0.85, 0.6)
    ax.zaxis._axinfo["grid"]["color"]     = (0.85, 0.85, 0.85, 0.6)
    ax.xaxis._axinfo["grid"]["linewidth"] = 0.4
    ax.yaxis._axinfo["grid"]["linewidth"] = 0.4
    ax.zaxis._axinfo["grid"]["linewidth"] = 0.4
    ax.tick_params(axis="both", labelsize=6, pad=1)


def format_val(v, unconverged: bool) -> str:
    if unconverged:
        return "---"
    if abs(v) >= 1.0:
        return f"{v:.2f}"
    return f"{v:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset_pointcloud")
    ap.add_argument("--out",     default="runs/geometry_pointclouds")
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--pt_body", type=float, default=0.6,
                    help="Scatter marker size for body points")
    ap.add_argument("--pt_edf",  type=float, default=0.9,
                    help="Scatter marker size for EDF points")
    ap.add_argument("--alpha_body", type=float, default=0.85)
    ap.add_argument("--alpha_edf",  type=float, default=0.90)
    args = ap.parse_args()

    ds_root = Path(args.dataset)
    out_stem = Path(args.out)
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # ── figure layout ─────────────────────────────────────────────────────────
    # Three columns; white background; 3D subplot each
    fig = plt.figure(figsize=(13, 4.8))
    fig.patch.set_facecolor("white")

    axes = []
    for col in range(3):
        ax = fig.add_subplot(1, 3, col + 1, projection="3d")
        axes.append(ax)

    # ── render each geometry ──────────────────────────────────────────────────
    for ax, geo in zip(axes, GEOMETRIES):
        body_raw, edf_raw, meta = load_sample(ds_root, geo["idx"])

        # Normalise coordinates so all three geometries are comparable in size
        body_n, edf_n = normalise_pts(body_raw, edf_raw)

        body_v = subsample(body_n, geo["n_body"], rng)
        edf_v  = subsample(edf_n,  geo["n_edf"],  rng)

        ax.scatter(
            body_v[:, 0], body_v[:, 1], body_v[:, 2],
            s=args.pt_body, c=BODY_COL, alpha=args.alpha_body,
            linewidths=0, depthshade=True, rasterized=True,
        )
        ax.scatter(
            edf_v[:, 0], edf_v[:, 1], edf_v[:, 2],
            s=args.pt_edf, c=EDF_COL, alpha=args.alpha_edf,
            linewidths=0, depthshade=True, rasterized=True,
        )

        equal_axes_3d(ax, body_v, edf_v)
        clean_3d_axes(ax)
        ax.view_init(elev=geo["elev"], azim=geo["azim"])

        ax.set_xlabel("x", labelpad=1)
        ax.set_ylabel("y", labelpad=1)
        ax.set_zlabel("z", labelpad=1)

        # ── title block ───────────────────────────────────────────────────────
        drag    = float(meta.get("body_drag",   float("nan")))
        thrust  = float(meta.get("edf_thrust",  float("nan")))
        rpm     = float(meta.get("rpm",         float("nan")))
        u_inf   = float(meta.get("u_inf",       float("nan")))
        unconv  = (abs(drag) < 1e-3 and abs(thrust) < 1e-3)

        label_str = geo["label"]
        if unconv:
            label_str += "  [unconverged]"

        title_lines = [
            r"$\bf{" + label_str.replace(" ", r"\ ") + r"}$",
            f"$u_\\infty$ = {u_inf:.0f} m/s,   RPM = {rpm:.0f}",
            f"$F_D$ = {format_val(drag, unconv)} N,   "
            f"$F_T$ = {format_val(thrust, unconv)} N",
        ]
        ax.set_title("\n".join(title_lines), fontsize=9, linespacing=1.55,
                     pad=4)

    # ── shared legend ─────────────────────────────────────────────────────────
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=BODY_COL, markersize=6, label="Body"),
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=EDF_COL, markersize=6, label="EDF nacelle"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])

    for ext in ("pdf", "png"):
        path = out_stem.with_suffix(f".{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight",
                    facecolor="white", transparent=False)
        print(f"Saved: {path}")

    plt.show()


if __name__ == "__main__":
    main()
