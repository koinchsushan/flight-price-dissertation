"""Splitting the data into training and testing sets, without cheating.

WHY NOT JUST SPLIT AT RANDOM?
The usual machine learning approach shuffles the rows and deals out 80/20. That
is wrong for anything involving time. Shuffling puts October prices into the
training set and August prices into the test set, so the model gets to peek at
the future. The score would look great and would be impossible to repeat in
real life, where the future has not happened yet.

WHAT WE DO INSTEAD (rolling_origin_splits)
Train on everything up to a date, then test on the next two weeks. Slide the
date forward and repeat. We do this five times, so we get five scores instead
of one. Five scores tell us whether a model is genuinely better or just got
lucky on one lucky fortnight.

WHY FIVE ROUNDS AND NOT ONE
Garcia Crespi et al. (2026) ran the same comparison both ways and found that
which model "won" actually flipped depending on the method. One split simply
cannot support a claim about which model is best.

THE SUBTLE TRAP: WHICH DATE DO WE SPLIT ON?
This is the part worth understanding properly, because it does not look like
cheating at first glance.

Every flight is priced over and over across several weeks. So each flight has
TWO kinds of date attached to it:
    searchDate  - the day we looked the price up
    flightDate  - the day the plane actually departs

If we split on searchDate, one single flight ends up on BOTH sides of the line:
some of its prices land in training, the rest in testing. We measured this --
61.8% of test flights would also be sitting in the training data. The model
would be marked on flights it had already memorised.

Splitting on flightDate keeps a whole flight, and its whole price history,
entirely on one side. assert_no_group_leakage() below checks this actually
happened, and it is run before every single model is fitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    """One round of training and testing: which rows go where, and over what dates.

    train_mask and test_mask are simple True/False lists, one entry per row of
    the data. True means "this row belongs to this side". Using True/False lists
    rather than row numbers means df[fold.train_mask] just works, no matter how
    the data happens to be indexed.
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
    """Build the five rounds of training and testing.

    Picture the calendar as a line. We take the last ten weeks and chop them
    into five two-week test blocks. Each round trains on everything before its
    block and is tested on the block itself:

        round 1:  train |============|  test [##]
        round 2:  train |===============|  test [##]
        round 3:  train |==================|  test [##]
        round 4:  train |=====================|  test [##]
        round 5:  train |========================|  test [##]

    Training data grows each round, which mirrors real life: the longer a
    service runs, the more history it has to learn from.

    Args:
        df: The data to split up.
        n_splits: How many rounds. We use 5.
        test_days: How long each test block is, in days. We use 14.
        date_col: Which date defines the timeline. Departure date -- see the
            note at the top of this file for why that choice matters so much.
        gap_days: Optional dead zone between the end of training and the start
            of testing. We leave it at 0, because splitting on departure date
            already keeps whole flights on one side. It is here in case a
            feature is ever added that looks across different departures.
        train_days: If set, train on a sliding window of this many days instead
            of everything from the beginning. We do not use this.
        min_train_days: Refuse to run if the first round would have less than
            this much training data. A guard against a silly configuration.

    Returns:
        The rounds, earliest first.

    Raises:
        ValueError: If the requested layout does not fit inside the data.
    """
    if n_splits < 1:
        raise ValueError("n_splits must be at least 1")
    if date_col not in df.columns:
        raise KeyError(f"{date_col!r} is not a column of the frame")

    dates = pd.to_datetime(df[date_col])
    start, end = dates.min(), dates.max()

    # Work backwards from the last date in the data. Five blocks of 14 days
    # means the test blocks together occupy the final 70 days; everything
    # before that is available for training.
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
    """The simpler alternative: cut the calendar once, train before, test after.

    The cut is placed by counting distinct DATES rather than rows. If we counted
    rows, one unusually busy departure date could drag the boundary sideways.

    This is a fallback and the final project does not use it. It gives a single
    score with no way of telling whether a model's lead is consistent or just an
    accident of where the line happened to fall. Kept here because the write-up
    discusses why it was rejected.
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
    """Safety check: make sure no single flight appears in both training and testing.

    This is the guard against the trap described at the top of this file. It is
    fast, and it runs before every model is fitted, because this is the mistake
    that would quietly inflate every score in the project without ever throwing
    an error of its own.

    It has been tested by deliberately feeding it a bad split, to confirm it
    actually refuses rather than just sitting there looking reassuring.

    Raises:
        AssertionError: If any flight has rows on both sides.
    """
    if isinstance(folds, Fold):
        folds = [folds]

    # Flight IDs are long strings. Comparing hundreds of thousands of strings is
    # slow and eats memory, and this check runs before every single fit, so the
    # cost adds up. pd.factorize swaps each unique string for a plain number
    # (a -> 0, b -> 1, ...), which makes the comparison roughly 16x smaller.
    # The numbers are only used for this comparison and thrown away after.
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
    """Build a readable table of what each round contains.

    Worth printing before fitting anything. The number to look at is how many
    spikes land in each test block. A block with very few would give a wobbly,
    unreliable score that should not be quietly averaged in with the others.

    In this project every block held between 9,600 and 12,200 spikes, so that
    concern did not apply -- but it was checked rather than assumed.
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
