"""Feature construction for fare regression and spike classification.

**The governing constraint is what is knowable at prediction time.** Research
question 2 asks whether a spike can be predicted *before* it happens, so the
prediction is made standing at observation *t-1* and asking about *t*. That
divides the columns three ways:

*Known about t* — calendar facts fixed the moment the flight was scheduled:
booking horizon, day of week, holiday proximity, season.

*Static per itinerary* — route, stops, airline, departure time of day, duration,
distance, fare flags. Constant along a trajectory, so no timing issue arises.

*Observed, and therefore lagged* — `totalFare` and `seatsRemaining` at *t* are
not available when the prediction is made. They enter only as values from *t-1*
and earlier. Rolling statistics are already causal by construction (see
:mod:`flightprice.spikes.labelling`).

*Forbidden* — `totalFare` and `baseFare` at *t*, and anything derived from them
(`spikeZ`, `isSpike`). The label is a function of `totalFare` at *t*, so
including it leaks the answer directly. :func:`assert_no_leaky_features` guards
against this and should be called before every fit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from flightprice.config import SUMMER_END, SUMMER_START, US_HOLIDAYS_2022

#: Columns that must never be used as predictors. The classification label is
#: derived from ``totalFare`` at the observation being predicted, so these leak
#: the answer. ``baseFare`` is ``totalFare`` less taxes and leaks just as badly.
LEAKY_COLUMNS: frozenset[str] = frozenset(
    {"totalFare", "baseFare", "spikeZ", "isSpike", "isSpikeEvent", "isLabelable"}
)


def parse_travel_duration(series: pd.Series) -> pd.Series:
    """Convert ISO-8601 durations such as ``"PT8H47M"`` to whole minutes.

    Handles hours-only (``"PT5H"``) and minutes-only (``"PT47M"``) forms.
    """
    text = series.astype("string")
    hours = text.str.extract(r"(\d+)H", expand=False).astype("Float64").fillna(0)
    minutes = text.str.extract(r"(\d+)M", expand=False).astype("Float64").fillna(0)
    return (hours * 60 + minutes).astype("Float64")


def add_calendar_features(df: pd.DataFrame, date_col: str = "flightDate") -> pd.DataFrame:
    """Add day-of-week, month, holiday and season features for the departure.

    All of these are knowable in advance — they depend only on the departure
    date, not on anything observed during the booking window.
    """
    out = df.copy()
    dates = out[date_col]

    out["dayOfWeek"] = dates.dt.dayofweek           # Monday = 0
    out["isWeekend"] = dates.dt.dayofweek >= 5
    out["month"] = dates.dt.month
    out["weekOfYear"] = dates.dt.isocalendar().week.astype("int32")

    holidays = pd.to_datetime(sorted(US_HOLIDAYS_2022))
    out["isHoliday"] = dates.isin(holidays)

    # Signed distance to the nearest in-window federal holiday. Fare effects
    # straddle a holiday rather than landing only on the day itself, so the
    # distance carries more signal than the flag alone.
    departure_days = dates.to_numpy("datetime64[D]").astype("int64")
    holiday_days = holidays.to_numpy("datetime64[D]").astype("int64")
    deltas = departure_days[:, None] - holiday_days[None, :]
    nearest = np.take_along_axis(deltas, np.abs(deltas).argmin(axis=1)[:, None], axis=1).ravel()
    out["daysToNearestHoliday"] = nearest

    out["isSummerSeason"] = (dates >= pd.Timestamp(SUMMER_START)) & (
        dates <= pd.Timestamp(SUMMER_END)
    )
    return out


def add_itinerary_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features fixed for the whole trajectory: airline, timing, duration.

    ``segmentsAirlineName`` and ``segmentsDepartureTimeRaw`` hold one entry per
    leg joined by ``||``. The operating carrier of the first leg and the
    scheduled departure time of the first leg are the relevant summaries; a
    flag records whether one carrier operates the whole itinerary.
    """
    out = df.copy()

    airlines = out["segmentsAirlineName"].astype("string")
    out["nSegments"] = airlines.str.count(r"\|\|").fillna(0).astype("int8") + 1
    out["airline"] = airlines.str.split("||", regex=False).str[0].astype("category")
    out["isSingleCarrier"] = airlines.str.split("||", regex=False).apply(
        lambda parts: len(set(parts)) == 1 if isinstance(parts, list) else True
    )

    # Local scheduled departure time of the first leg. The raw value carries a
    # UTC offset, so the hour is read from the string rather than converted --
    # what matters commercially is the local clock time of the flight.
    raw = out["segmentsDepartureTimeRaw"].astype("string")
    out["departureHour"] = raw.str.slice(11, 13).astype("Int8")
    out["departureTimeOfDay"] = pd.cut(
        out["departureHour"],
        bins=[-1, 5, 8, 12, 17, 21, 24],
        labels=["red-eye", "early", "morning", "afternoon", "evening", "night"],
    )

    out["travelDurationMinutes"] = parse_travel_duration(out["travelDuration"])
    return out


def add_lagged_features(
    df: pd.DataFrame,
    lags: tuple[int, ...] = (1, 2, 3),
    group_col: str = "legId",
    time_col: str = "searchDate",
) -> pd.DataFrame:
    """Add lagged fares and seat counts, plus momentum derived from them.

    Every column produced here is shifted by at least one observation, so none
    of it depends on the observation being predicted.

    Args:
        df: Frame carrying ``totalFare``, ``seatsRemaining`` and the rolling
            statistics from the spike-labelling step.
        lags: Which lags of the fare to add.
        group_col: Trajectory identifier.
        time_col: Ordering column within a trajectory.
    """
    out = df.sort_values([group_col, time_col]).reset_index(drop=True)
    grouped = out.groupby(group_col, observed=True)

    for lag in lags:
        out[f"fareLag{lag}"] = grouped["totalFare"].shift(lag)

    # Most recent observed movement, and its rate of change.
    out["fareChangeLag1"] = out["fareLag1"] - out["fareLag2"]
    out["fareChangeLag2"] = out["fareLag2"] - out["fareLag3"]
    out["fareAccel"] = out["fareChangeLag1"] - out["fareChangeLag2"]

    # Where the last observed fare sat relative to its own trailing baseline.
    # This is the standardised quantity the label thresholds, but computed one
    # step back, so it is legitimately available.
    out["zLag1"] = (out["fareLag1"] - out["rollMean"]) / out["rollStd"].replace(0, np.nan)

    out["seatsRemainingLag1"] = grouped["seatsRemaining"].shift(1)
    out["seatsChangeLag1"] = out["seatsRemainingLag1"] - grouped["seatsRemaining"].shift(2)

    # Position within the trajectory: how much history the model actually has.
    out["observationIndex"] = grouped.cumcount()

    out["rollStdRatio"] = out["rollStd"] / out["rollMean"].replace(0, np.nan)
    return out


#: Predictors, grouped by what makes them available at prediction time.
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "temporal": (
        "daysBeforeDeparture",
        "dayOfWeek",
        "isWeekend",
        "month",
        "weekOfYear",
        "observationIndex",
    ),
    "calendar": (
        "isHoliday",
        "daysToNearestHoliday",
        "isSummerSeason",
    ),
    "itinerary": (
        "isNonStop",
        "nSegments",
        "isBasicEconomy",
        "isRefundable",
        "totalTravelDistance",
        "travelDurationMinutes",
        "departureHour",
        "isSingleCarrier",
    ),
    "lagged": (
        "fareLag1",
        "fareLag2",
        "fareLag3",
        "fareChangeLag1",
        "fareChangeLag2",
        "fareAccel",
        "zLag1",
        "seatsRemainingLag1",
        "seatsChangeLag1",
        "rollMean",
        "rollStd",
        "rollStdRatio",
    ),
    "categorical": (
        "route",
        "airline",
        "departureTimeOfDay",
    ),
}

#: Flat list of every predictor.
FEATURE_COLUMNS: tuple[str, ...] = tuple(
    column for group in FEATURE_GROUPS.values() for column in group
)


def assert_no_leaky_features(columns: list[str] | tuple[str, ...]) -> None:
    """Fail if a feature list contains a column derived from the target.

    Call before fitting. The classification label is a deterministic function of
    ``totalFare`` at the predicted observation, so its presence would produce a
    near-perfect score that means nothing.

    Raises:
        AssertionError: If any forbidden column is present.
    """
    leaks = sorted(set(columns) & LEAKY_COLUMNS)
    if leaks:
        raise AssertionError(
            f"Leaky column(s) in the feature set: {', '.join(leaks)}. "
            "The label is derived from totalFare at the predicted observation, so these "
            "reveal the answer. Use the lagged equivalents (fareLag1, zLag1) instead."
        )


def build_feature_frame(df: pd.DataFrame, lags: tuple[int, ...] = (1, 2, 3)) -> pd.DataFrame:
    """Run the full feature pipeline in order.

    Returns:
        A sorted copy carrying every column in :data:`FEATURE_COLUMNS` alongside
        the identifiers and labels needed for splitting and evaluation.
    """
    out = add_calendar_features(df)
    out = add_itinerary_features(out)
    out = add_lagged_features(out, lags=lags)
    return out
