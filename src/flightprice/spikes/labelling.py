"""Price-spike detection over per-flight fare trajectories.

A *trajectory* is the sequence of fares observed for one specific itinerary
(`legId`) across successive `searchDate`s as its departure approaches. A spike
is an observation whose fare departs from that trajectory's recent level by more
than a set number of rolling standard deviations.

Two design decisions here are load-bearing and are justified in notebook 02:

**Rolling statistics are causal.** The mean and standard deviation at
observation *t* are computed from observations *t-w … t-1* and exclude *t*
itself. A centred or inclusive window would let an observation contribute to the
baseline it is judged against, which both suppresses genuine spikes and leaks
information that would not be available at prediction time. Research question 2
asks whether a spike can be predicted *before* it happens, so the labelling must
not depend on the future.

**Spikes are labelled as events, not states.** A single fare increase keeps the
fare elevated relative to a trailing window for several subsequent observations,
so a naive label marks one price change as many spikes. ``is_spike_event`` marks
only the first observation of each run — the transition — while ``is_spike``
retains the raw per-observation state for comparison.
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
    """Add ``daysBeforeDeparture`` — the booking horizon of each observation."""
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
    """Attach causal rolling mean and standard deviation per trajectory.

    The frame is sorted by ``(group_col, time_col)`` before the rolling
    statistics are computed, so callers need not pre-sort.

    Args:
        df: Observations, one row per (flight, search date).
        window: Number of prior observations in the rolling window.
        min_periods: Minimum prior observations required before a statistic is
            produced. Below this the rolling columns are ``NaN`` and the
            observation is not labelable.
        group_col: Trajectory identifier.
        value_col: Fare column.
        time_col: Ordering column within a trajectory.

    Returns:
        A sorted copy with ``rollMean`` and ``rollStd`` added.
    """
    out = df.sort_values([group_col, time_col]).reset_index(drop=True)
    grouped = out.groupby(group_col, observed=True)[value_col]

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
    """Label spikes from previously attached rolling statistics.

    Adds four columns:

    - ``spikeZ`` — standardised deviation from the trailing mean.
    - ``isLabelable`` — whether a spike verdict is defined for this observation.
      Requires a rolling standard deviation that exists and is non-zero; a
      trajectory that has been perfectly flat gives no scale against which to
      judge a deviation.
    - ``isSpike`` — the raw per-observation state.
    - ``isSpikeEvent`` — the first observation of each spike run.

    Args:
        df: Output of :func:`add_rolling_stats`.
        sigma: Threshold in rolling standard deviations.
        direction: ``"up"`` for surges only, ``"down"`` for drops only,
            ``"both"`` for either.

    Returns:
        A copy with the label columns added.
    """
    if "rollStd" not in df.columns:
        raise KeyError("call add_rolling_stats() before label_spikes()")

    out = df.copy()
    out["isLabelable"] = out["rollStd"].notna() & (out["rollStd"] > 0)
    out["spikeZ"] = (out[value_col] - out["rollMean"]) / out["rollStd"]

    if direction == "up":
        hit = out["spikeZ"] >= sigma
    elif direction == "down":
        hit = out["spikeZ"] <= -sigma
    else:
        hit = out["spikeZ"].abs() >= sigma

    out["isSpike"] = (hit & out["isLabelable"]).fillna(False)

    previous = out.groupby(group_col, observed=True)["isSpike"].shift(1, fill_value=False)
    out["isSpikeEvent"] = out["isSpike"] & ~previous

    return out


def spike_summary(df: pd.DataFrame, by: str | list[str] | None = None) -> pd.DataFrame:
    """Summarise labelable coverage and spike rates.

    Rates are expressed against *labelable* observations rather than all rows,
    since an observation with no trailing window admits no verdict either way.
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
    """Evaluate several rolling-window sizes on the same data.

    ``min_periods`` is set to ``max(3, round(window * min_periods_ratio))`` so
    that the requirement scales with the window rather than being fixed, which
    would make short and long windows incomparable.

    Returns:
        One row per window, with coverage and both spike rates.
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
