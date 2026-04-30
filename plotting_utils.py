from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


BODY_COLOR = "#1f77b4"
EDF_COLOR = "#ff7f0e"


def apply_publication_style() -> None:
    """
    Apply journal-style defaults globally.
    """
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "axes.facecolor": "white",
            "axes.edgecolor": "0.15",
            "axes.linewidth": 0.8,
            "figure.dpi": 300,
            "figure.facecolor": "white",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "lines.linewidth": 1.5,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.6,
            "legend.fontsize": 10,
            "legend.framealpha": 0.9,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
        }
    )


def save_figure_png_pdf(fig: plt.Figure, save_path: str | Path) -> tuple[Path, Path]:
    """
    Save a figure to both PNG and PDF.
    """
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    png_path = p if p.suffix.lower() == ".png" else p.with_suffix(".png")
    pdf_path = png_path.with_suffix(".pdf")

    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    return png_path, pdf_path


def panel_label(ax, label: str) -> None:
    """
    Place panel label in lower-left corner.
    """
    text_kwargs = {
        "ha": "left",
        "va": "bottom",
        "fontsize": 11,
        "color": "black",
    }
    if hasattr(ax, "text2D"):
        ax.text2D(0.02, 0.02, f"({label})", transform=ax.transAxes, **text_kwargs)
    else:
        ax.text(0.02, 0.02, f"({label})", transform=ax.transAxes, **text_kwargs)
