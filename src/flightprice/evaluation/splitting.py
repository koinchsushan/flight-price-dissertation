"""Time-aware train/test splitting for the model comparison.

Random k-fold cross-validation is invalid here. It would place later
observations in training and earlier ones in test, letting a model learn from
the future and returning scores that could not be reproduced in deployment.

Two schemes are provided:

- :func:`rolling_origin_splits` — walk-forward validation. The training window
  grows (or slides) and the test window advances, giving one score per fold
  instead of a single point estimate. Preferred: García Crespi et al. (2026)
  found model rankings can *reverse* between a single split and rolling-origin
  on a structurally comparable three-way comparison, so a single split cannot
  support a claim about which model family wins.
- :func:`chronological_split` — one cut date. Cheaper, and adequate for the
  secondary route's lighter validation pass.

**Splits are taken on departure date, not search date.** A single itinerary
(`legId`) is searched repeatedly over weeks, so cutting on `searchDate` leaves
the same flight on both sides of the boundary — measured at 61.8% of test
flights on this dataset. The model would then be scored partly on flights whose
fare history it had already memorised. Cutting on `flightDate` keeps every
trajectory wholly in one side, and :func:`assert_no_group_leakage` verifies it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    """One train/test division.

    ``train_mask`` and ``test_mask`` are boolean arrays aligned to the row order
    of the frame the fold was built from, so ``df[fold.train_mask]`` works
    whatever the frame's index looks like.
    """

    index: int
    train_mask: np.ndarray = field(repr=False)
    test_mask: np.ndarray = field(repr=False)
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    @property
    def n_train(self) -> int:
        return int(self.train_mask.sum())

    @property
    def n_test(self) -> int:
        return int(self.test_mask.sum())

    def __str__(self) -> str:
        return (
            f"Fold {self.index}: train {self.train_start.date()}–{self.train_end.date()} "
            f"({self.n_train:,} rows) | test {self.test_start.date()}–{self.test_end.date()} "
            f"({self.n_test:,} rows)"
        )


def rolling_origin_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    test_days: int = 14,
    date_col: str = "flightDate",
    gap_days: int = 0,
    train_days: int | None = None,
    min_train_days: int = 30,
) -> list[Fold]:
    """Build walk-forward folds over departure dates.

    Test windows tile the end of the observation period backwards, so the final
    fold tests on the most recent data available and each earlier fold steps
    back by ``test_days``.

    Args:
        df: Observations to split.
        n_splits: Number of folds.
        test_days: Length of each test window, in days.
        date_col: Date column defining the timeline. Defaults to departure date
            — see the module docstring for why this matters.
        gap_days: Optional embargo between the end of training and the start of
            testing. Not required when splitting on departure date, since whole
            trajectories already fall on one side, but available if a feature is
            later added that looks across departures.
        train_days: If given, training uses a sliding window of this many days
            rather than expanding from the start of the data.
        min_train_days: Raise if the earliest fold would train on less than this.

    Returns:
        Folds ordered earliest first.

    Raises:
        ValueError: If the requested layout does not fit the data.
    """
    if n_splits < 1:
        raise ValueError("n_splits must be at least 1")
    if date_col not in df.columns:
        raise KeyError(f"{date_col!r} is not a column of the frame")

    dates = pd.to_datetime(df[date_col])
    start, end = dates.min(), dates.max()

    # Test windows occupy the final n_splits * test_days days.
    span_needed = pd.Timedelta(days=n_splits * test_days + gap_days)
    first_test_start = end + pd.Timedelta(days=1) - pd.Timedelta(days=n_splits * test_days)

    available_train = (first_test_start - pd.Timedelta(days=gap_days) - start).days
    if available_train < min_train_days:
        raise ValueError(
            f"{n_splits} folds of {test_days} days need {span_needed.days} days at the end, "
            f"leaving only {available_train} days to train the first fold "
            f"(min_train_days={min_train_days}). Reduce n_splits or test_days."
        )

    folds: list[Fold] = []
    for i in range(n_splits):
        test_start = first_test_start + pd.Timedelta(days=i * test_days)
        test_end = test_start + pd.Timedelta(days=test_days - 1)

        train_end = test_start - pd.Timedelta(days=1 + gap_days)
        train_start = (
            start if train_days is None else max(start, train_end - pd.Timedelta(days=train_days - 1))
        )

        train_mask = ((dates >= train_start) & (dates <= train_end)).to_numpy()
        test_mask = ((dates >= test_start) & (dates <= test_end)).to_numpy()

        if not test_mask.any():
            continue

        folds.append(
            Fold(
                index=len(folds) + 1,
                train_mask=train_mask,
                test_mask=test_mask,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=min(test_end, end),
            )
        )

    if not folds:
        raise ValueError("no non-empty test windows were produced")

    return folds


def chronological_split(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
    date_col: str = "flightDate",
    gap_days: int = 0,
) -> Fold:
    """Split once at a cut date, training before it and testing after.

    The cut is placed at the ``1 - test_fraction`` quantile of *distinct dates*,
    not of rows, so an unusually busy departure date cannot drag the boundary.

    This is the fallback scheme. Where it is used instead of
    :func:`rolling_origin_splits`, say so explicitly in the write-up and give
    the reason — it yields a single point estimate with no measure of whether a
    model's advantage is consistent or an artefact of the cut date.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must lie strictly between 0 and 1")

    dates = pd.to_datetime(df[date_col])
    unique_dates = np.sort(dates.unique())
    cut = pd.Timestamp(unique_dates[int(len(unique_dates) * (1 - test_fraction))])

    train_end = cut - pd.Timedelta(days=1 + gap_days)
    train_mask = (dates <= train_end).to_numpy()
    test_mask = (dates >= cut).to_numpy()

    return Fold(
        index=1,
        train_mask=train_mask,
        test_mask=test_mask,
        train_start=dates.min(),
        train_end=train_end,
        test_start=cut,
        test_end=dates.max(),
    )


def assert_no_group_leakage(
    df: pd.DataFrame, folds: list[Fold] | Fold, group_col: str = "legId"
) -> None:
    """Verify no trajectory appears in both sides of any fold.

    Cheap to run and worth running: this is the failure mode that would quietly
    inflate every score in the comparison.

    Raises:
        AssertionError: If any group spans a fold's train and test sets.
    """
    if isinstance(folds, Fold):
        folds = [folds]

    # Factorised to integer codes once. The identifiers are strings, and
    # intersecting hundreds of thousands of them directly is both slow and
    # memory-hungry -- this guard runs before every fit, so its cost matters.
    # Reducing to unique codes first shrinks each side by roughly 16x.
    codes = pd.factorize(df[group_col], sort=False)[0]
    for fold in folds:
        train_ids = np.unique(codes[fold.train_mask])
        test_ids = np.unique(codes[fold.test_mask])
        overlap = np.intersect1d(train_ids, test_ids, assume_unique=True)
        if overlap.size:
            raise AssertionError(
                f"Fold {fold.index}: {overlap.size:,} {group_col} value(s) appear in both "
                f"train and test. Split on departure date, not search date."
            )


def describe_folds(
    df: pd.DataFrame,
    folds: list[Fold] | Fold,
    label_col: str | None = "isSpikeEvent",
    group_col: str | None = "legId",
) -> pd.DataFrame:
    """Tabulate fold sizes, date ranges and positive-class balance.

    Worth printing in the notebook before any model is fitted: a fold whose test
    window carries very few positives will produce an unstable F1 that should
    not be averaged in uncritically.
    """
    if isinstance(folds, Fold):
        folds = [folds]

    rows = []
    for fold in folds:
        row = {
            "fold": fold.index,
            "train_start": fold.train_start.date(),
            "train_end": fold.train_end.date(),
            "test_start": fold.test_start.date(),
            "test_end": fold.test_end.date(),
            "train_rows": fold.n_train,
            "test_rows": fold.n_test,
        }
        if group_col is not None:
            row["train_flights"] = df.loc[fold.train_mask, group_col].nunique()
            row["test_flights"] = df.loc[fold.test_mask, group_col].nunique()
        if label_col is not None and label_col in df.columns:
            test_labels = df.loc[fold.test_mask, label_col]
            row["test_positives"] = int(test_labels.sum())
            row["test_pos_rate_%"] = round(float(test_labels.mean()) * 100, 2)
        rows.append(row)

    return pd.DataFrame(rows).set_index("fold")
