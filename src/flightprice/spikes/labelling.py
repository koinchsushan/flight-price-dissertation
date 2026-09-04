"""Finding price spikes in the fare history of individual flights.

WHAT THIS FILE DOES
Each flight in the dataset was priced many times before it departed. Line those
prices up in date order and you get that flight's "fare history". This file
looks at each history and marks the moments where the price suddenly jumped.

HOW A SPIKE IS DEFINED
For any given day we work out two things from the days before it:
  1. the average fare recently          (rollMean)
  2. how much the fare usually moves    (rollStd)
A day counts as a spike if its fare is more than 2 x rollStd above rollMean.
In plain terms: "much bigger than this flight's normal wobble".

TWO DECISIONS WORTH KNOWING ABOUT (both are defended in notebook 02)

1. We only ever look backwards.
   The average for today is built from earlier days only. Today's own fare is
   never included. If it were, a big jump would be pulled into its own average
   and would partly hide itself. It would also mean using information we would
   not have in real life, since the whole point is to warn people BEFORE the
   price rises.

2. We mark the START of a rise, not every day it stays high.
   When a fare jumps it usually stays high for several days afterwards. If we
   flagged every one of those days, one price rise would be counted as five or
   six spikes. So 'isSpikeEvent' marks only the first day of each rise, and
   that is what the models are trained to predict. 'isSpike' keeps the raw
   day-by-day version so the two can be compared.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

Direction = Literal["up", "down", "both"]


def add_days_before_departure(
    df: pd.DataFrame,
    flight_col: str = "flightDate",
    search_col: str = "searchDate",
) -> pd.DataFrame:
    """Add a column for how many days before departure each price was seen.

    A price recorded on 1 June for a flight leaving on 11 June gets 10.
    This is the single most useful feature in the whole project.
    """
    out = df.copy()
    out["daysBeforeDeparture"] = (out[flight_col] - out[search_col]).dt.days
    return out


def add_rolling_stats(
    df: pd.DataFrame,
    window: int,
    min_periods: int,
    group_col: str = "legId",
    value_col: str = "totalFare",
    time_col: str = "searchDate",
) -> pd.DataFrame:
    """Work out each flight's recent average fare, and how much it usually moves.

    These two numbers are the yardstick a spike is measured against. They are
    calculated separately for every flight, using only that flight's own past.

    Args:
        df: One row per (flight, date the price was checked).
        window: How many earlier prices to look back over. We use 10.
        min_periods: How many earlier prices we insist on before we are willing
            to judge at all. We use 5. Below that, the columns are left empty
            and the row is treated as "cannot say".
        group_col: The column that identifies one flight (legId).
        value_col: The fare column.
        time_col: The column used to put a flight's prices in date order.

    Returns:
        A copy of the data, sorted, with rollMean and rollStd added.
    """
    # Sort so each flight's prices are in date order, then handle each flight
    # separately -- one flight's prices must never leak into another's average.
    out = df.sort_values([group_col, time_col]).reset_index(drop=True)
    grouped = out.groupby(group_col, observed=True)[value_col]

    # .rolling(10) = "the last 10 prices, including today's".
    # .shift(1)    = "now slide everything down one row".
    #
    # The shift is the important bit and it is easy to miss. Without it, today's
    # fare would be inside the average it is being compared against. With it,
    # each row gets the average of the 10 days BEFORE it. Same for the spread.
    out["rollMean"] = grouped.transform(
        lambda s: s.rolling(window, min_periods=min_periods).mean().shift(1)
    )
    out["rollStd"] = grouped.transform(
        lambda s: s.rolling(window, min_periods=min_periods).std().shift(1)
    )
    return out


def label_spikes(
    df: pd.DataFrame,
    sigma: float,
    direction: Direction = "both",
    group_col: str = "legId",
    value_col: str = "totalFare",
) -> pd.DataFrame:
    """Decide which prices count as spikes, using the yardstick built above.

    Adds four columns:

    - spikeZ       how many "normal wobbles" today's fare sits above the recent
                   average. A value of 2.5 means "two and a half times the usual
                   movement above where this flight has been sitting".
    - isLabelable  whether we are able to judge this row at all. We need a
                   spread that exists and is not zero. If a fare has been
                   completely flat there is no yardstick, so we say nothing
                   rather than guess.
    - isSpike      True on every day the fare is sitting high.
    - isSpikeEvent True only on the FIRST day of each rise. This is the one the
                   models actually predict.

    Args:
        df: The output of add_rolling_stats().
        sigma: How many wobbles above average counts as a spike. We use 2.
        direction: "up" for price rises only (what we use), "down" for drops,
            "both" for either.

    Returns:
        A copy of the data with the four label columns added.
    """
    if "rollStd" not in df.columns:
        raise KeyError("call add_rolling_stats() before label_spikes()")

    out = df.copy()

    # Can we judge this row? Only if a spread exists and is not zero.
    out["isLabelable"] = out["rollStd"].notna() & (out["rollStd"] > 0)

    # How far above its recent average is this fare, measured in "normal
    # wobbles"? Dividing by rollStd is what makes a $30 jump on a $150 shuttle
    # comparable with a $30 jump on a $450 transcontinental flight.
    out["spikeZ"] = (out[value_col] - out["rollMean"]) / out["rollStd"]

    if direction == "up":
        hit = out["spikeZ"] >= sigma
    elif direction == "down":
        hit = out["spikeZ"] <= -sigma
    else:
        hit = out["spikeZ"].abs() >= sigma

    # A day is a spike only if it clears the threshold AND we were able to judge it.
    out["isSpike"] = (hit & out["isLabelable"]).fillna(False)

    # Now keep only the first day of each run of high prices.
    # 'previous' is simply "was yesterday also flagged?" for the same flight.
    # A day is the START of a rise if it is flagged and yesterday was not.
    #
    #   isSpike:      F  F  T  T  T  F  T
    #   previous:     F  F  F  T  T  T  F
    #   isSpikeEvent: F  F  T  F  F  F  T     <- two rises, not five
    previous = out.groupby(group_col, observed=True)["isSpike"].shift(1, fill_value=False)
    out["isSpikeEvent"] = out["isSpike"] & ~previous

    return out


def spike_summary(df: pd.DataFrame, by: str | list[str] | None = None) -> pd.DataFrame:
    """Count up how many spikes were found, and how often.

    Percentages are worked out against the rows we could actually judge, not
    against every row. Counting rows we had to skip would water the rate down
    and make spikes look rarer than they are.
    """

    def _summarise(g: pd.DataFrame) -> pd.Series:
        labelable = int(g["isLabelable"].sum())
        return pd.Series(
            {
                "rows": len(g),
                "labelable": labelable,
                "labelable_%": labelable / len(g) * 100 if len(g) else 0.0,
                "spike_obs": int(g["isSpike"].sum()),
                "spike_events": int(g["isSpikeEvent"].sum()),
                "obs_rate_%": g["isSpike"].sum() / labelable * 100 if labelable else 0.0,
                "event_rate_%": g["isSpikeEvent"].sum() / labelable * 100 if labelable else 0.0,
            }
        )

    if by is None:
        return _summarise(df).to_frame("all").T

    return df.groupby(by, observed=True).apply(_summarise, include_groups=False)


def sweep_windows(
    df: pd.DataFrame,
    windows: list[int],
    sigma: float,
    min_periods_ratio: float = 0.5,
    direction: Direction = "up",
    group_col: str = "legId",
) -> pd.DataFrame:
    """Try several look-back window sizes and report what each one gives.

    This is the experiment that chose the window of 10. The paper we borrowed
    the spike rule from used 20, but a typical flight here is only priced 9
    times, so a 20-day window would have been unusable for most flights.

    The minimum number of earlier prices scales with the window rather than
    being fixed, otherwise a short window and a long window would be held to
    different standards and could not fairly be compared.

    Returns:
        One row per window size, showing how much data it lets us judge and
        how many spikes it finds.
    """
    rows = []
    for window in windows:
        min_periods = max(3, round(window * min_periods_ratio))
        stats = add_rolling_stats(df, window=window, min_periods=min_periods, group_col=group_col)
        labelled = label_spikes(stats, sigma=sigma, direction=direction, group_col=group_col)

        labelable = int(labelled["isLabelable"].sum())
        rows.append(
            {
                "window": window,
                "min_periods": min_periods,
                "labelable": labelable,
                "labelable_%": labelable / len(labelled) * 100,
                "spike_obs": int(labelled["isSpike"].sum()),
                "spike_events": int(labelled["isSpikeEvent"].sum()),
                "obs_rate_%": labelled["isSpike"].sum() / labelable * 100,
                "event_rate_%": labelled["isSpikeEvent"].sum() / labelable * 100,
                "obs_per_event": labelled["isSpike"].sum() / max(labelled["isSpikeEvent"].sum(), 1),
            }
        )

    return pd.DataFrame(rows).set_index("window")
