"""Shared settings for charts, so every figure in the dissertation matches.

Every single figure goes through save_fig() below. That guarantees they all
share the same resolution, the same file format and the same folder -- so the
dissertation never ends up with one crisp chart next to one blurry one.

It also means the figures folder always reflects the code that is committed,
rather than whatever happened to be run by hand last.
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
    """Apply the shared look for charts: colours, grid, font sizes.

    Call once near the top of each notebook, just after the imports.
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
    """Save a chart to reports/figures at print quality (300 dpi).

    300 dpi is the resolution printed work needs. A chart saved at screen
    resolution looks fine in a notebook and turns to mush in a printed
    dissertation.

    Args:
        fig: The chart to save.
        name: The filename without the extension. Prefix it with the notebook
            number, e.g. "01_fare_distribution", so the figures folder sorts
            itself into the order the work was done.
        verbose: Print where it was written.

    Returns:
        The path it was written to.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.{FIGURE_FORMAT}"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")

    if verbose:
        print(f"saved -> {path.relative_to(FIGURES_DIR.parents[1])}")

    return path
