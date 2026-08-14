"""Central configuration: paths, constants and reproducibility settings.

Every hard-coded value used across the pipeline lives here so that it can be
cited once in the methodology chapter rather than hunted through the codebase.
Values marked "verified" were confirmed against a primary source during the
research phase; each is documented in the README.
"""

from __future__ import annotations

import random
from datetime import date
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
PROCESSED_DIR: Path = DATA_DIR / "processed"
FIGURES_DIR: Path = PROJECT_ROOT / "reports" / "figures"

#: The pre-extracted four-route subset of ``dilwong/flightprices``
#: (2,156,316 rows, ~662 MB). The full 31 GB raw file is never loaded --
#: see ``data/README.md`` for how to obtain and reproduce it.
RAW_SUBSET: Path = DATA_DIR / "jfk_lax_bos_lga.csv"

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #

#: Single seed used for every stochastic component (train/test shuffling within
#: folds, SMOTE/ADASYN synthesis, XGBoost subsampling, LSTM weight init).
#: A fixed seed is what makes the three-way model comparison defensible.
RANDOM_SEED: int = 42


def set_seeds(seed: int = RANDOM_SEED) -> None:
    """Seed every RNG the pipeline touches.

    Call once at the top of each notebook, before any model is built. ``torch``
    is imported lazily so that the data-only notebooks do not pay its import
    cost.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:  # torch is only needed for the LSTM
        return

    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


# --------------------------------------------------------------------------- #
# Dataset constants
# --------------------------------------------------------------------------- #

#: Observation window of the source dataset. No November/December data exists,
#: so no Thanksgiving or Christmas signal is observable -- seasonality claims
#: must be scoped accordingly (README, "Scope and limitations").
DATA_START: date = date(2022, 4, 16)
DATA_END: date = date(2022, 10, 5)

#: ``segmentsCabinCode`` values representing a pure-coach itinerary across all
#: legs. These cover 99.83% of the subset; the remaining 0.17%
#: (business/first/premium/mixed) sits in the extreme tail of the fare
#: distribution and is filtered out before modelling (README, "Data").
COACH_ONLY_CODES: frozenset[str] = frozenset(
    {"coach", "coach||coach", "coach||coach||coach"}
)

#: Directional airport pairs retained in the subset.
ROUTES: tuple[str, ...] = ("JFK-LAX", "LAX-JFK", "BOS-LGA", "LGA-BOS")

#: Long-haul transcontinental route -- primary, fully tuned.
LONG_HAUL_ROUTES: tuple[str, ...] = ("JFK-LAX", "LAX-JFK")

#: Short-haul shuttle route -- secondary, lighter validation pass.
SHORT_HAUL_ROUTES: tuple[str, ...] = ("BOS-LGA", "LGA-BOS")

# --------------------------------------------------------------------------- #
# Calendar features
# --------------------------------------------------------------------------- #

#: US federal holidays falling inside the observation window. Verified directly
#: against the US Office of Personnel Management calendar.
US_HOLIDAYS_2022: dict[date, str] = {
    date(2022, 5, 30): "Memorial Day",
    date(2022, 6, 20): "Juneteenth (observed)",
    date(2022, 7, 4): "Independence Day",
    date(2022, 9, 5): "Labor Day",
}

#: Summer-season proxy: Memorial Day to Labor Day inclusive. Stands in for US
#: school-term dates, for which no clean single national source exists. This is
#: a stated simplification, not an oversight (README, "Method").
SUMMER_START: date = date(2022, 5, 30)
SUMMER_END: date = date(2022, 9, 5)

# --------------------------------------------------------------------------- #
# Spike definition
# --------------------------------------------------------------------------- #

#: A spike is a fare deviating from its rolling mean by more than this many
#: rolling standard deviations (Lee et al., 2024).
SPIKE_SIGMA: float = 2.0

#: Starting window size, in observations, inherited from Lee et al. (2024) who
#: used 20 daily commodity closes. NOT yet validated on flight data -- the unit
#: here is days-before-departure, not calendar days. Tuned empirically in
#: notebook 02.
SPIKE_WINDOW_DEFAULT: int = 20

# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

FIGURE_DPI: int = 300
FIGURE_FORMAT: str = "png"
