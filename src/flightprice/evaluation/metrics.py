"""How we mark the models.

WHY THERE IS NO ACCURACY SCORE ANYWHERE IN THIS PROJECT
This is worth being ready to explain, because it is the first thing someone
usually asks.

Only about 8% of days are spikes. So a model that lazily says "no spike" every
single time, and never gets anything right, is still correct 92% of the time.
Accuracy would make that useless model look excellent, and would make every
model look roughly the same. It is the wrong ruler for a rare event, so it is
left out entirely.

WHAT WE USE INSTEAD
  precision  Of the times we shouted "spike!", how often were we right?
  recall     Of the spikes that actually happened, how many did we catch?
  F1         One number balancing the two, since they pull against each other.
             Shout constantly and recall is great but precision is awful.
             Shout almost never and precision is great but recall is awful.

  ROC AUC and average precision are summaries that do not depend on where we
  draw the "probably yes" line. Of the two, average precision is the more
  honest here, because ROC AUC gets flattered by the huge number of easy,
  obviously-not-a-spike days. Saito and Rehmsmeier (2015) show exactly this.
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
    """Mark a spike-prediction model.

    Args:
        y_true: What actually happened (1 = spike, 0 = no spike).
        y_prob: The model's confidence, between 0 and 1.
        threshold: Where "probably not" becomes "probably yes". Default 0.5.
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
    """Mark a fare-predicting model. Both error measures, always.

    RMSE and MAE both measure "how far off were we", but they disagree on
    purpose, and reporting only one would have misled this entire project:

      MAE  the plain average miss. $20 out on average means $20.
      RMSE squares the errors first, so a few enormous misses count for far
           more than lots of small ones.

    Fares sit still most of the time and then jump. So a model can win on RMSE
    (it handles the jumps less badly) while LOSING on MAE (it is worse on the
    ordinary days). That is exactly what happened here: XGBoost beat the naive
    baseline on RMSE and lost to it on MAE in all ten rounds. Reporting RMSE on
    its own would have hidden that completely.
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
    """Find the best place to draw the "probably yes" line.

    Tries every cut-off from 0.05 to 0.95 and keeps whichever gives the best F1.

    IMPORTANT: this must only ever be run on validation data, never on the test
    data. The harness sets aside the last two weeks of each training window for
    exactly this. Choosing the line on the test set would mean looking at the
    answers before deciding how to answer.

    Returns:
        The chosen cut-off, and the F1 score it achieved.
    """
    if grid is None:
        grid = np.arange(0.05, 0.96, 0.01)

    scores = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in grid]
    best = int(np.argmax(scores))
    return float(grid[best]), float(scores[best])


def summarise_folds(results: pd.DataFrame, metrics: tuple[str, ...]) -> pd.DataFrame:
    """Average the five rounds, and report how much they varied.

    The variation is the whole reason for running five rounds rather than one.

    If model A averages 0.377 and model B averages 0.350, that looks like a win
    for A -- until you notice that A itself bounced between 0.29 and 0.46 across
    the five rounds. A gap smaller than a model's own round-to-round wobble has
    not been demonstrated at all. Nine of the nineteen comparisons in this
    project turned out to be exactly that, and the write-up reports them as
    undecided rather than pretending there was a winner.
    """
    available = [m for m in metrics if m in results.columns]
    stats = results[available].agg(["mean", "std"]).T
    stats.columns = ["mean", "std"]
    return stats.round(4)
