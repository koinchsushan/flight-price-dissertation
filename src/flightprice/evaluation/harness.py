"""Fold-running harness shared by every model family.

The point of a shared harness is comparability. If XGBoost, SARIMA and the LSTM
were each evaluated by their own bespoke loop, a difference in scores could come
from a difference in the evaluation rather than from the models, and the
comparison that research questions 1 and 2 rest on would not be sound. Every
model is therefore passed through the same splits, the same guards and the same
metrics.

A model is supplied as a *factory* — a zero-argument callable returning an
unfitted estimator — so that each fold trains a fresh model rather than
continuing to fit one that has already seen later data.
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
    """Run both leakage guards. Called before any fit, never skipped."""
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
    """Fit and score a classifier across folds.

    Args:
        frame: Modelling frame, one row per observation.
        folds: Folds from :mod:`flightprice.evaluation.splitting`.
        feature_cols: Predictor names. Checked against the leakage guard.
        label_col: Binary target.
        make_model: Called with ``scale_pos_weight`` and returns an unfitted
            estimator exposing ``fit`` and ``predict_proba``.
        encode: Converts a frame and column list to a numeric design matrix.
        use_class_weight: Pass the negative/positive ratio as
            ``scale_pos_weight``; otherwise pass 1.0.
        tune_threshold: Choose the decision threshold by maximising F1 on a
            validation slice taken from the **end of the training window**. The
            test set is never used to select it.
        validation_days: Length of that validation slice.
        design_matrix: A design matrix already produced by ``encode``. Supplying it
            avoids re-encoding for every configuration, which matters when several
            are compared on the same data.
        keep_models: Retain fitted estimators. Off by default to limit memory.

    Returns:
        ``(metrics_per_fold, importances_per_fold, models)``.
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
            # Carve the tail of the training window as validation. Selecting the
            # threshold on test would use the answers to pick the answer.
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
