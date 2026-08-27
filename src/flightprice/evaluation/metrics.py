"""Metrics for the two prediction tasks.

Accuracy is deliberately absent from the classification metrics. At a spike rate
of roughly 8%, a classifier that never predicts a spike scores 92% accurate
while being useless, so accuracy would flatter every model and separate none of
them. Precision, recall and F1 are reported instead, with ROC AUC and average
precision as threshold-independent summaries.

Average precision matters more than ROC AUC under class imbalance: it summarises
the precision-recall curve, which reflects performance on the minority class,
whereas ROC AUC is buoyed by the large number of easy negatives.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Precision, recall, F1 and threshold-independent summaries.

    Args:
        y_true: Binary labels.
        y_prob: Predicted probability of the positive class.
        threshold: Decision boundary applied to ``y_prob``.
    """
    y_pred = (y_prob >= threshold).astype(int)
    positives = int(y_true.sum())

    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if positives else float("nan"),
        "avg_precision": average_precision_score(y_true, y_prob) if positives else float("nan"),
        "threshold": threshold,
        "n_test": len(y_true),
        "n_positive": positives,
        "positive_rate": positives / len(y_true) if len(y_true) else float("nan"),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """RMSE, MAE and R² for the fare-level task.

    MAE is reported alongside RMSE because fares are heavy-tailed: RMSE is
    dominated by a few large misses, while MAE describes the typical error.
    """
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mean_actual": float(np.mean(y_true)),
        "n_test": len(y_true),
    }


def best_f1_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, grid: np.ndarray | None = None
) -> tuple[float, float]:
    """Find the decision threshold maximising F1.

    **Must be called on validation data, never on the test set.** The harness
    carves a validation slice from the end of each training window for this
    purpose; tuning on test would report a threshold chosen with knowledge of
    the answers.

    Returns:
        ``(threshold, f1_at_that_threshold)``.
    """
    if grid is None:
        grid = np.arange(0.05, 0.96, 0.01)

    scores = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in grid]
    best = int(np.argmax(scores))
    return float(grid[best]), float(scores[best])


def summarise_folds(results: pd.DataFrame, metrics: tuple[str, ...]) -> pd.DataFrame:
    """Mean and standard deviation of each metric across folds.

    The spread is the point of running multiple folds: a model that wins on
    average but with a spread overlapping its rival has not been shown to be
    better, and the write-up should say so rather than ranking on the mean.
    """
    available = [m for m in metrics if m in results.columns]
    stats = results[available].agg(["mean", "std"]).T
    stats.columns = ["mean", "std"]
    return stats.round(4)
