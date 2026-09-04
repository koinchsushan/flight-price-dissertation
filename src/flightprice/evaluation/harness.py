"""The shared testing machine that every model is put through.

WHY THIS FILE EXISTS
The whole project is a comparison between three kinds of model. If each one had
its own separate testing code, then a difference in scores might just be a
difference in how they were marked, rather than a real difference in ability.
That would sink the entire dissertation.

So all three go through this one file: the same data splits, the same safety
checks, the same scoring. Whatever differences come out at the end are down to
the models themselves.

ONE DETAIL WORTH BEING ABLE TO EXPLAIN
Models are not passed in ready-made. Instead we pass in a small function that
BUILDS a fresh model when called ("a factory"). That guarantees each round
starts from a blank slate. If we reused one model across rounds, round 3 would
already have seen round 4's data, and every score afterwards would be inflated.
"""

from __future__ import annotations

import gc
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from flightprice.evaluation.metrics import (
    best_f1_threshold,
    classification_metrics,
    regression_metrics,
)
from flightprice.evaluation.splitting import Fold, assert_no_group_leakage
from flightprice.features.build import assert_no_leaky_features


def _check_guards(
    frame: pd.DataFrame, folds: list[Fold], feature_cols: list[str], group_col: str | None
) -> None:
    """Run both safety checks. Called before every single fit, never skipped.

    Check 1: no banned columns in the feature list (nothing derived from the fare).
    Check 2: no flight appearing in both the training and testing halves.

    Together these are what stop the project producing impressive-looking
    numbers that mean nothing.
    """
    assert_no_leaky_features(feature_cols)
    if group_col is not None and group_col in frame.columns:
        assert_no_group_leakage(frame, folds, group_col=group_col)


def evaluate_classifier(
    frame: pd.DataFrame,
    folds: list[Fold],
    feature_cols: list[str],
    label_col: str,
    make_model: Callable[[float], Any],
    encode: Callable[[pd.DataFrame, list[str]], pd.DataFrame],
    *,
    design_matrix: pd.DataFrame | None = None,
    use_class_weight: bool = False,
    tune_threshold: bool = False,
    validation_days: int = 14,
    date_col: str = "flightDate",
    group_col: str | None = "legId",
    label: str = "model",
    keep_models: bool = False,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[pd.Series], list[Any]]:
    """Train and score a spike-predicting model across all five rounds.

    For each round: train on the earlier data, predict on the later data, write
    down how well it did. Repeat five times, return all five scores.

    Args:
        frame: The data, one row per (flight, date checked).
        folds: The five rounds, from splitting.py.
        feature_cols: Which columns the model may look at. Safety-checked.
        label_col: The yes/no column we are trying to predict (isSpikeEvent).
        make_model: The factory described at the top of this file.
        encode: The function that turns words into numbers.
        use_class_weight: Spikes are rare (about 1 in 13), so a lazy model could
            score well by always saying "no spike". Turning this on tells the
            model to treat missing a spike as a much more serious error.
        tune_threshold: Models output a probability, and something has to decide
            where "probably not" becomes "probably yes". Normally that line sits
            at 0.5. Turning this on moves the line to wherever works best --
            chosen on the last two weeks of the TRAINING data, never on the test
            data. Choosing it on the test data would be using the answers to
            pick the answer.
        validation_days: How much training data to set aside for that choice.
        design_matrix: Already-encoded data, if we have it. Saves re-doing the
            same conversion three times when comparing three settings.
        keep_models: Whether to hang on to the trained models. Off by default,
            because five trained models on a million rows eats memory fast.

    Returns:
        Three things: the scores per round, which features mattered per round,
        and the trained models (empty unless keep_models was set).
    """
    _check_guards(frame, folds, list(feature_cols), group_col)

    X_all = encode(frame, list(feature_cols)) if design_matrix is None else design_matrix
    y_all = frame[label_col].astype(int).to_numpy()
    dates = pd.to_datetime(frame[date_col])

    rows: list[dict[str, Any]] = []
    importances: list[pd.Series] = []
    models: list[Any] = []

    for fold in folds:
        train_mask = fold.train_mask.copy()
        threshold = 0.5
        val_mask = None

        if tune_threshold:
            # Slice the last two weeks off the END of the training data and set
            # it aside. We will use that slice, and only that slice, to pick
            # where the "probably yes" line goes.
            #
            # It has to come from training data. Picking the line on the test
            # data would mean peeking at the answers to decide how to answer --
            # a subtle form of cheating that flatters the final score.
            val_start = fold.train_end - pd.Timedelta(days=validation_days - 1)
            val_mask = train_mask & (dates >= val_start).to_numpy()
            train_mask = train_mask & (dates < val_start).to_numpy()
            if val_mask.sum() == 0 or train_mask.sum() == 0:
                raise ValueError(
                    f"Fold {fold.index}: validation slice of {validation_days} days leaves "
                    "no training data. Reduce validation_days."
                )

        X_tr, y_tr = X_all[train_mask], y_all[train_mask]
        n_pos = int(y_tr.sum())

        # spw = "scale positive weight". If there are 12 non-spikes for every
        # spike, this comes out at 12, which tells the model to treat one missed
        # spike as being as costly as twelve false alarms. Setting it to 1.0
        # means "treat both mistakes the same" -- our unweighted baseline.
        spw = ((len(y_tr) - n_pos) / n_pos) if (use_class_weight and n_pos) else 1.0

        model = make_model(spw)
        model.fit(X_tr, y_tr)

        if tune_threshold:
            val_prob = model.predict_proba(X_all[val_mask])[:, 1]
            threshold, _ = best_f1_threshold(y_all[val_mask], val_prob)

        test_prob = model.predict_proba(X_all[fold.test_mask])[:, 1]
        metrics = classification_metrics(y_all[fold.test_mask], test_prob, threshold)
        metrics.update({"model": label, "fold": fold.index, "n_train": int(train_mask.sum())})
        rows.append(metrics)

        if hasattr(model, "feature_importances_"):
            importances.append(pd.Series(model.feature_importances_, index=list(feature_cols)))
        if keep_models:
            models.append(model)

        # Free the memory before starting the next round. Each round holds
        # several hundred thousand rows, and without this the notebook runs out
        # of memory partway through.
        del X_tr, y_tr, test_prob
        gc.collect()

        if verbose:
            print(
                f"  fold {fold.index}: F1={metrics['f1']:.3f} "
                f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
                f"AUC={metrics['roc_auc']:.3f} AP={metrics['avg_precision']:.3f} "
                f"thr={threshold:.2f}",
                flush=True,
            )

    return pd.DataFrame(rows), importances, models


def evaluate_regressor(
    frame: pd.DataFrame,
    folds: list[Fold],
    feature_cols: list[str],
    target_col: str,
    make_model: Callable[[], Any],
    encode: Callable[[pd.DataFrame, list[str]], pd.DataFrame],
    *,
    design_matrix: pd.DataFrame | None = None,
    group_col: str | None = "legId",
    label: str = "model",
    keep_models: bool = False,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[pd.Series], list[Any]]:
    """Fit and score a regressor across folds, for the fare-level task."""
    _check_guards(frame, folds, list(feature_cols), group_col)

    X_all = encode(frame, list(feature_cols)) if design_matrix is None else design_matrix
    y_all = frame[target_col].to_numpy(dtype="float64")

    rows: list[dict[str, Any]] = []
    importances: list[pd.Series] = []
    models: list[Any] = []

    for fold in folds:
        model = make_model()
        model.fit(X_all[fold.train_mask], y_all[fold.train_mask])

        pred = model.predict(X_all[fold.test_mask])
        metrics = regression_metrics(y_all[fold.test_mask], pred)
        metrics.update(
            {"model": label, "fold": fold.index, "n_train": int(fold.train_mask.sum())}
        )
        rows.append(metrics)

        if hasattr(model, "feature_importances_"):
            importances.append(pd.Series(model.feature_importances_, index=list(feature_cols)))
        if keep_models:
            models.append(model)

        del pred
        gc.collect()

        if verbose:
            print(
                f"  fold {fold.index}: RMSE={metrics['rmse']:.2f} "
                f"MAE={metrics['mae']:.2f} R2={metrics['r2']:.3f}",
                flush=True,
            )

    return pd.DataFrame(rows), importances, models


def mean_importance(importances: list[pd.Series]) -> pd.Series:
    """Average feature importance across folds, descending.

    Averaging over folds is more trustworthy than any single fold: gain-based
    importance is unstable between fits, particularly among correlated features.
    """
    if not importances:
        return pd.Series(dtype="float64")
    return pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
