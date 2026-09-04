"""Building the 32 pieces of information the models learn from.

THE ONE RULE THAT GOVERNS EVERYTHING HERE
Only use information we would genuinely have at the moment of predicting.

Picture standing on one day, looking at tomorrow, and asking "is the price about
to jump?" Anything we could not possibly know while standing there is off
limits. That single rule sorts every column in the dataset into four buckets:

1. KNOWN IN ADVANCE - fixed the moment the flight was scheduled.
   How far ahead we are booking, the day of the week, whether it is near a
   public holiday, whether it is summer. All fine to use.

2. FIXED FOR THE WHOLE FLIGHT - route, number of stops, airline, departure
   time, journey length. These never change, so there is no timing problem.

3. YESTERDAY'S NUMBERS - the fare and the seats left are things we observe.
   Tomorrow's values are not available today, so they only ever enter as
   yesterday's values ("lagged"). fareLag1 means "the fare last time we looked".

4. BANNED - the fare on the day being predicted, and anything built from it.
   This is the important one. The spike label is calculated directly FROM the
   fare, so handing the model that fare is handing it the answer.

HOW BADLY DOES BREAKING RULE 4 MATTER?
We measured it. Including the fare pushes the score from 0.815 to 0.997 -- from
"decent" to "basically perfect". A score that good on a problem this hard is not
success, it is a warning light. assert_no_leaky_features() below refuses to run
if any banned column appears, and it is called before every fit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from flightprice.config import SUMMER_END, SUMMER_START, US_HOLIDAYS_2022

#: The banned list. Never let a model see any of these.
#: The spike label is worked out from totalFare, so totalFare gives the game
#: away. baseFare is just totalFare minus taxes, so it gives it away too. The
#: rest are all calculated from the fare, so the same applies to them.
LEAKY_COLUMNS: frozenset[str] = frozenset(
    {"totalFare", "baseFare", "spikeZ", "isSpike", "isSpikeEvent", "isLabelable"}
)


def parse_travel_duration(series: pd.Series) -> pd.Series:
    """Turn a journey length like "PT8H47M" into a plain number of minutes (527).

    The dataset stores journey times in a standard but unfriendly format:
    PT = "period of time", then hours and minutes. We pull the digits before
    the H and before the M and do the arithmetic. Some entries have only hours
    ("PT5H") or only minutes ("PT47M"), so a missing piece counts as zero.
    """
    text = series.astype("string")
    hours = text.str.extract(r"(\d+)H", expand=False).astype("Float64").fillna(0)
    minutes = text.str.extract(r"(\d+)M", expand=False).astype("Float64").fillna(0)
    return (hours * 60 + minutes).astype("Float64")


def add_calendar_features(df: pd.DataFrame, date_col: str = "flightDate") -> pd.DataFrame:
    """Add calendar facts about the departure day: weekday, month, holidays, season.

    Everything here comes from the departure date alone. Nothing depends on what
    happened while the flight was on sale, so it is all safely known in advance.
    """
    out = df.copy()
    dates = out[date_col]

    out["dayOfWeek"] = dates.dt.dayofweek           # Monday = 0
    out["isWeekend"] = dates.dt.dayofweek >= 5
    out["month"] = dates.dt.month
    out["weekOfYear"] = dates.dt.isocalendar().week.astype("int32")

    holidays = pd.to_datetime(sorted(US_HOLIDAYS_2022))
    out["isHoliday"] = dates.isin(holidays)

    # How many days away is the nearest public holiday? Negative means before.
    #
    # Why not just a yes/no "is it a holiday" flag? Because holiday pricing
    # spreads out around the date -- fares climb in the days leading up to it
    # and fall away after. "Three days before Independence Day" is useful;
    # "not a holiday" throws that away.
    #
    # The three lines below are a compact way of doing this for 1.9 million
    # rows at once, which is why they look dense:
    #   1. turn both sets of dates into plain day-numbers so we can subtract
    #   2. build a grid of every departure against every holiday and subtract,
    #      giving the distance from each flight to each of the four holidays
    #   3. for each flight, keep whichever of the four is closest
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
    """Add facts about the flight itself: airline, departure time, journey length.

    These never change across a flight's price history, so there is no timing
    worry with any of them.

    One quirk of the data to know about: a journey with a connection stores its
    legs in a single cell, separated by "||". So a two-leg trip might read
    "Delta||Delta". We take the first leg's airline and the first leg's
    departure time, and separately record whether the whole trip is flown by
    one carrier or involves a handover between two.
    """
    out = df.copy()

    airlines = out["segmentsAirlineName"].astype("string")
    out["nSegments"] = airlines.str.count(r"\|\|").fillna(0).astype("int8") + 1
    out["airline"] = airlines.str.split("||", regex=False).str[0].astype("category")
    out["isSingleCarrier"] = airlines.str.split("||", regex=False).apply(
        lambda parts: len(set(parts)) == 1 if isinstance(parts, list) else True
    )

    # Departure hour, read straight out of the text rather than converted.
    #
    # The stored time already carries a timezone offset. If we let the computer
    # "helpfully" convert it, an 8am flight from New York and an 8am flight from
    # Los Angeles would end up as different numbers. What matters for pricing is
    # what the clock says where the passenger is standing, so we take characters
    # 11 and 12 of the text, which is where the hour sits.
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
    """Add "what the price was doing recently" features.

    "Lagged" simply means shifted back in time. fareLag1 is the fare the last
    time we looked, fareLag2 the time before that, and so on. Every single
    column built here is shifted back at least one step, which is what keeps
    the rule at the top of this file intact.

    Args:
        df: Data carrying the fare, the seats left, and the rolling average
            and spread worked out during spike labelling.
        lags: How many steps back to go. We use 1, 2 and 3.
        group_col: The column identifying one flight.
        time_col: The column used to put a flight's prices in order.
    """
    out = df.sort_values([group_col, time_col]).reset_index(drop=True)
    grouped = out.groupby(group_col, observed=True)

    for lag in lags:
        out[f"fareLag{lag}"] = grouped["totalFare"].shift(lag)

    # Which way was the price moving, and was that movement speeding up?
    #   fareChangeLag1  the most recent move        ("it went up $20")
    #   fareChangeLag2  the move before that        ("it went up $5 before that")
    #   fareAccel       the difference between them ("so it is accelerating")
    out["fareChangeLag1"] = out["fareLag1"] - out["fareLag2"]
    out["fareChangeLag2"] = out["fareLag2"] - out["fareLag3"]
    out["fareAccel"] = out["fareChangeLag1"] - out["fareChangeLag2"]

    # How unusual was YESTERDAY's fare, on the same scale the spike rule uses?
    #
    # This is deliberately the same calculation as spikeZ, just one step back in
    # time. That makes it legal: it describes a day that has already happened.
    # It turned out to be one of the most useful features in the project.
    # (Dividing by zero would give nonsense, so a zero spread becomes "unknown".)
    out["zLag1"] = (out["fareLag1"] - out["rollMean"]) / out["rollStd"].replace(0, np.nan)

    out["seatsRemainingLag1"] = grouped["seatsRemaining"].shift(1)
    out["seatsChangeLag1"] = out["seatsRemainingLag1"] - grouped["seatsRemaining"].shift(2)

    # How many times has this flight been priced before now? A model behaves
    # differently on the first observation of a flight, where it has nothing to
    # go on, than on the twentieth.
    out["observationIndex"] = grouped.cumcount()

    out["rollStdRatio"] = out["rollStd"] / out["rollMean"].replace(0, np.nan)
    return out


#: The 32 features, sorted into the buckets described at the top of this file.
#: Grouping them this way is what let us report which KIND of information the
#: models leaned on most, not just which individual columns.
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


def encode_features(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Turn words into numbers, because models cannot do arithmetic on text.

    Airlines arrive as names like "Delta" or "JetBlue". Each distinct name is
    swapped for a number (Delta -> 0, JetBlue -> 1, and so on).

    This function sits here, on its own, rather than next to either model. That
    is deliberate: XGBoost and PyTorch cannot be loaded into the same program on
    this machine without it crashing, so anything both need has to live somewhere
    neutral that imports neither.

    An honest limitation, stated rather than buried: numbering categories this
    way implies an order that does not exist -- it hints that JetBlue is
    "greater than" Delta. Decision trees do not care, because they only ever ask
    "is this value in this group or not". The neural network does care a little,
    and the textbook fix (learned embeddings) was not attempted here. It is
    written up as a limitation.
    """
    out = frame[list(cols)].copy()
    for col in out.columns:
        if str(out[col].dtype) in {"category", "object", "string", "bool"}:
            out[col] = out[col].astype("category").cat.codes
    return out.astype("float32")


def assert_no_leaky_features(columns: list[str] | tuple[str, ...]) -> None:
    """Safety check: refuse to run if any banned column has crept into the features.

    Called before every fit. If the fare itself reached the model, the score
    would come out near perfect and would mean absolutely nothing, and the
    mistake would be very easy to miss because nothing would visibly break.

    Like the other guard in this project, this one was tested by feeding it a
    deliberately bad list to confirm it actually refuses.

    Raises:
        AssertionError: If a banned column is present.
    """
    leaks = sorted(set(columns) & LEAKY_COLUMNS)
    if leaks:
        raise AssertionError(
            f"Leaky column(s) in the feature set: {', '.join(leaks)}. "
            "The label is derived from totalFare at the predicted observation, so these "
            "reveal the answer. Use the lagged equivalents (fareLag1, zLag1) instead."
        )


def build_feature_frame(df: pd.DataFrame, lags: tuple[int, ...] = (1, 2, 3)) -> pd.DataFrame:
    """Run all three feature-building steps, in order.

    Order matters: the lagged features need the flights sorted by date, and
    add_lagged_features does that sorting, so it goes last.

    Returns:
        The data with all 32 features attached, ready for a model.
    """
    out = add_calendar_features(df)
    out = add_itinerary_features(out)
    out = add_lagged_features(out, lags=lags)
    return out
