"""Loading and first-pass cleaning of the four-route flight-price subset.

The full 82 M-row / 31 GB source file is never touched here -- everything reads
from the pre-extracted subset (see ``data/README.md``), which fits comfortably in
memory on a normal machine.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from flightprice.config import COACH_ONLY_CODES, RAW_SUBSET

#: Columns retained for modelling. The dropped columns are per-segment raw
#: timestamps, airport/airline codes duplicated elsewhere, and
#: ``segmentsEquipmentDescription`` (aircraft type, 6% null, not a price driver
#: for this study). ``segmentsCabinCode`` is kept because the coach-only filter
#: depends on it.
MODELLING_COLUMNS: tuple[str, ...] = (
    "legId",
    "searchDate",
    "flightDate",
    "startingAirport",
    "destinationAirport",
    "travelDuration",
    "elapsedDays",
    "isBasicEconomy",
    "isRefundable",
    "isNonStop",
    "baseFare",
    "totalFare",
    "seatsRemaining",
    "totalTravelDistance",
    "segmentsDepartureTimeRaw",
    "segmentsAirlineName",
    "segmentsCabinCode",
)

#: Low-cardinality columns worth storing as ``category`` to keep the frame small.
_CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "startingAirport",
    "destinationAirport",
    "segmentsAirlineName",
    "segmentsCabinCode",
)


def load_raw(
    path: Path | str = RAW_SUBSET,
    columns: tuple[str, ...] | None = MODELLING_COLUMNS,
) -> pd.DataFrame:
    """Read the four-route subset from CSV.

    Args:
        path: Location of the subset CSV.
        columns: Columns to read. Pass ``None`` to read all 27.

    Returns:
        The raw frame, with ``searchDate`` and ``flightDate`` parsed as
        datetimes and no rows filtered.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Subset not found at {path}. Expected the pre-extracted "
            "jfk_lax_bos_lga.csv described in data/README.md."
        )

    df = pd.read_csv(
        path,
        usecols=list(columns) if columns else None,
        parse_dates=["searchDate", "flightDate"],
    )

    for col in _CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


def filter_coach_only(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Keep only itineraries that are coach class on every segment.

    Business, first, premium and mixed-cabin itineraries make up 0.17% of the
    subset but occupy the extreme tail of the fare distribution, which would
    otherwise dominate both the spike definition and the regression targets.

    Args:
        df: Frame containing a ``segmentsCabinCode`` column.
        verbose: Print how many rows were removed.

    Returns:
        A filtered copy.
    """
    if "segmentsCabinCode" not in df.columns:
        raise KeyError("segmentsCabinCode is required for the coach-only filter")

    before = len(df)
    mask = df["segmentsCabinCode"].astype("string").isin(COACH_ONLY_CODES)
    out = df[mask].copy()
    removed = before - len(out)

    if verbose:
        print(
            f"Coach-only filter: {before:,} -> {len(out):,} rows "
            f"({removed:,} removed, {removed / before:.2%})"
        )

    return out


def add_route(df: pd.DataFrame) -> pd.DataFrame:
    """Add a directional ``route`` column, e.g. ``"JFK-LAX"``.

    Directional rather than symmetric: JFK->LAX and LAX->JFK price differently
    and are treated as separate series throughout.
    """
    out = df.copy()
    out["route"] = (
        df["startingAirport"].astype("string")
        + "-"
        + df["destinationAirport"].astype("string")
    ).astype("category")
    return out


def load_clean(
    path: Path | str = RAW_SUBSET,
    columns: tuple[str, ...] | None = MODELLING_COLUMNS,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load the subset, apply the coach-only filter and add ``route``.

    This is the standard entry point for every downstream notebook.
    """
    df = load_raw(path, columns)
    if verbose:
        print(f"Loaded {len(df):,} rows x {len(df.columns)} columns from {Path(path).name}")
    df = filter_coach_only(df, verbose=verbose)
    return add_route(df)
