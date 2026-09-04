"""The XGBoost models, and the two "do nothing clever" baselines they must beat.

WHAT XGBOOST IS, IN ONE PARAGRAPH
It builds a lot of small decision trees, one after another. Each new tree is
trained to fix the mistakes the previous ones made. Individually the trees are
weak; added together they are strong. It cannot use time order directly, which
is why we hand-build features like "the fare last time we looked".

WHY THE BASELINES MATTER
An error of $47 sounds bad or good depending entirely on what you compare it
against. So both tasks carry a deliberately dumb benchmark:

  For predicting the fare -- PERSISTENCE. It simply guesses that tomorrow's
  price equals today's. This sounds silly, but fares hold completely still 62%
  of the time, so it is right far more often than not. It is a genuinely tough
  benchmark, not a straw man set up to be knocked down -- and in this project it
  actually BEAT XGBoost on average error in all ten rounds.

  For predicting spikes -- THE BASE RATE. It always answers with the overall
  spike rate and learns nothing at all. It scores an F1 of zero. That is the
  floor any real model has to clear before it deserves attention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from flightprice.config import RANDOM_SEED
from flightprice.features.build import encode_features  # re-exported for notebook 04

#: Settings shared by every XGBoost model in this project.
#:
#: These are sensible defaults, deliberately NOT tuned. Every model family here
#: runs at sensible defaults, so the comparison is between the families
#: themselves rather than between "the one I spent a weekend tuning" and the
#: rest. It is written up as a limitation: any of the three would score better
#: with effort spent on it.
#:
#: In plain terms:
#:   n_estimators      how many trees to build (400)
#:   max_depth         how many questions deep each tree may go (6)
#:   learning_rate     how much each new tree is allowed to change the answer
#:   subsample         each tree sees 80% of the rows, which prevents
#:   colsample_bytree  and 80% of the columns -- memorising the training data
#:   min_child_weight  refuse to split a branch covering fewer than 5 rows
XGB_PARAMS: dict = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.08,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "tree_method": "hist",
    # Use exactly 4 processor cores, rather than "all of them". Fixing this
    # means the timings quoted in the write-up are repeatable, instead of
    # depending on whatever else the laptop happened to be doing at the time.
    "n_jobs": 4,
    "random_state": RANDOM_SEED,
}


def make_classifier(scale_pos_weight: float = 1.0, **overrides):
    """Build a spike-predicting model.

    scale_pos_weight controls how seriously the model takes a missed spike:
        1.0  treat both mistakes equally -- our unweighted version
        12   treat one miss as being as bad as twelve false alarms
    """
    import xgboost as xgb

    params = {**XGB_PARAMS, "eval_metric": "logloss", **overrides}
    return xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, **params)


def make_regressor(**overrides):
    """Build a fare-predicting model. Same settings, different goal."""
    import xgboost as xgb

    params = {**XGB_PARAMS, "eval_metric": "rmse", **overrides}
    return xgb.XGBRegressor(**params)


class PersistenceRegressor:
    """The "nothing will change" baseline: guess that the fare stays where it is.

    There is no learning here at all. It just repeats the last observed price.

    It is written to look like a real model from the outside -- same .fit() and
    .predict() -- so the testing harness can put it through exactly the same
    process as XGBoost, with no special-casing. That is what makes the
    comparison fair.
    """

    def __init__(self, lag_col: str = "fareLag1"):
        self.lag_col = lag_col
        self._fallback = 0.0

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "PersistenceRegressor":
        # Nothing to learn. We only note the average fare, to have something
        # sensible to say on the very first observation of a flight, where there
        # is no previous price to repeat.
        self._fallback = float(np.nanmean(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.lag_col not in X.columns:
            raise KeyError(f"{self.lag_col!r} must be among the features for persistence")
        return X[self.lag_col].fillna(self._fallback).to_numpy(dtype="float64")


class BaseRateClassifier:
    """The "learn nothing" baseline for spikes: always answer with the overall rate.

    If 8% of days are spikes, this says "8% chance" to every single day, forever.
    It never commits to a yes, so it never catches a spike and scores an F1 of
    zero. That zero is the floor. Any model worth reporting has to beat it.
    """

    def __init__(self):
        self.rate_ = 0.0

    def fit(self, X, y) -> "BaseRateClassifier":
        self.rate_ = float(np.mean(y))
        return self

    def predict_proba(self, X) -> np.ndarray:
        p = np.full(len(X), self.rate_, dtype="float64")
        return np.column_stack([1 - p, p])
