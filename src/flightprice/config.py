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

#: Window over which fares were *scraped*. Nothing was observed after
#: ``SEARCH_END``, which is what right-censors the late departures below.
SEARCH_START: date = date(2022, 4, 16)
SEARCH_END: date = date(2022, 10, 5)

#: Window of *departure* dates present in the data. This runs 45 days beyond
#: ``SEARCH_END`` because a search on any given day returns flights departing
#: up to roughly two months later. Verified against the subset in notebook 01.
FLIGHT_START: date = date(2022, 4, 17)
FLIGHT_END: date = date(2022, 11, 19)

#: Last departure date whose price trajectory is observable all the way down to
#: one day before departure. Flights departing after this are right-censored:
#: scraping stopped on ``SEARCH_END``, so a flight departing 19 Nov is never
#: observed closer than 27 days out. Since the final approach to departure is
#: exactly where revenue-management price surges occur, spike analysis is
#: restricted to departures on or before this date (see notebook 01).
FULL_TRAJECTORY_END: date = date(2022, 10, 12)

#: Thanksgiving 2022 (24 Nov) falls five days after ``FLIGHT_END`` and
#: Christmas is far outside it, so neither is observable. Seasonality claims are
#: scoped to summer travel and the four in-window federal holidays below.
THANKSGIVING_2022: date = date(2022, 11, 24)

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

#: Rolling window, in observations, over a single flight's fare trajectory.
#: Tuned empirically in notebook 02; Lee et al. (2024) used 20 daily commodity
#: closes, which does not transfer directly here -- the unit is successive
#: searches for one flight, not calendar days, and 85.8% of consecutive searches
#: are exactly one day apart so a window of 10 spans roughly 10 days.
SPIKE_WINDOW: int = 10

#: Prior observations required before a spike verdict is issued. The median
#: flight is observed 9 times, so demanding a full window would discard most
#: trajectories; 5 keeps coverage at ~72% of rows while still estimating the
#: standard deviation from a usable sample.
SPIKE_MIN_PERIODS: int = 5

# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

#: Number of rolling-origin (walk-forward) folds for the primary route.
#: García Crespi et al. (2026) found model rankings can reverse between a single
#: chronological split and rolling-origin validation, so the model comparison --
#: which is what research questions 1 and 2 turn on -- cannot rest on one split.
VALIDATION_N_SPLITS: int = 5

#: Length of each fold's test window, in days. Five folds of 14 days each leaves
#: roughly 100 days to train the earliest fold, and every test window carries
#: 9,600-12,200 spike events, so no fold is too thin for a stable F1.
VALIDATION_TEST_DAYS: int = 14

#: Column defining the split timeline. Departure date, NOT search date: one
#: itinerary is searched repeatedly over weeks, so cutting on `searchDate`
#: leaves 61.8% of test flights also present in training. See
#: `flightprice.evaluation.splitting`.
VALIDATION_DATE_COL: str = "flightDate"

# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

FIGURE_DPI: int = 300
FIGURE_FORMAT: str = "png"
