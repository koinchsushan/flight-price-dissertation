"""Shared plotting configuration and figure persistence.

Every figure in the dissertation is produced through :func:`save_fig` so that
resolution, format and location are identical across notebooks, and so that the
figures directory always reflects the committed code rather than whatever was
last run by hand.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from flightprice.config import FIGURE_DPI, FIGURE_FORMAT, FIGURES_DIR

#: Colour-blind-safe qualitative palette, used for route and category series.
PALETTE: tuple[str, ...] = (
    "#4C72B0",  # blue
    "#DD8452",  # orange
    "#55A868",  # green
    "#C44E52",  # red
    "#8172B3",  # purple
)


def set_plot_style() -> None:
    """Apply the project-wide matplotlib/seaborn style.

    Call once near the top of each notebook, after the imports.
    """
    sns.set_theme(style="whitegrid", palette=list(PALETTE))
    mpl.rcParams.update(
        {
            "figure.dpi": 100,          # on-screen; save_fig overrides on write
            "savefig.dpi": FIGURE_DPI,
            "savefig.bbox": "tight",
            "figure.titlesize": 13,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_fig(fig: plt.Figure, name: str, verbose: bool = True) -> Path:
    """Write a figure to ``reports/figures`` at publication resolution.

    Args:
        fig: The figure to write.
        name: Filename stem, without extension. Use a numeric prefix matching
            the notebook, e.g. ``"01_fare_distribution"``.
        verbose: Print the destination path.

    Returns:
        The path written to.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.{FIGURE_FORMAT}"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")

    if verbose:
        print(f"saved -> {path.relative_to(FIGURES_DIR.parents[1])}")

    return path
