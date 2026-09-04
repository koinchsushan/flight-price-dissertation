"""Loading the data and doing the first round of cleaning.

A NOTE ON FILE SIZES, WHICH SHAPED THE WHOLE PROJECT
The original dataset is 82 million rows and 31 GB. Nothing here ever opens that
file. Trying to load it normally runs out of memory -- confirmed, not assumed,
including on hosted machines with more memory than a laptop.

What we work from instead is a pre-extracted slice covering just our four
routes: 2.1 million rows, about 662 MB, which opens fine in the ordinary way.
data/README.md explains how that slice was produced.
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
    """Read the data file, with the two date columns converted into real dates.

    Nothing is filtered here -- this is the raw data as it comes.

    Args:
        path: Where the file is.
        columns: Which columns to read. Reading only what is needed keeps
            memory down. Pass None to read all 27.

    Returns:
        The data, with searchDate and flightDate as proper dates rather than
        text, so that date arithmetic works.
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
    """Keep economy-class tickets only, and drop everything else.

    WHY, WITH THE NUMBERS
    Business and first class tickets are only 0.17% of the data -- about 3,700
    rows out of 2.15 million -- but they sit right at the top of the price range
    and would distort everything downstream.

    Removing them drops the highest fare from $4,782.60 to $2,281.61, while the
    middle fare does not move by a single cent. That pair of facts is the
    justification: we removed a thin tail, not a chunk of the actual data.

    Args:
        df: The data, which must have the cabin-class column.
        verbose: Print how many rows were dropped.

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
    """Add a single "route" column, like "JFK-LAX", built from the two airport codes.

    Direction is kept separate on purpose. JFK-to-LAX and LAX-to-JFK are treated
    as two different routes throughout the project, because they genuinely price
    differently -- demand is not symmetric.
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
    """Do all three loading steps in one call: read, filter to economy, add route.

    This is the one function the notebooks actually call. Everything above is a
    step it runs on your behalf.
    """
    df = load_raw(path, columns)
    if verbose:
        print(f"Loaded {len(df):,} rows x {len(df.columns)} columns from {Path(path).name}")
    df = filter_coach_only(df, verbose=verbose)
    return add_route(df)
