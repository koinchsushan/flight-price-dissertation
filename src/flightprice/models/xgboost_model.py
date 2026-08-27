"""XGBoost estimators and the naive baselines they are judged against.

A model's RMSE or F1 means little in isolation. Both tasks therefore carry a
naive baseline that any useful model must beat:

- **Fare regression** — persistence, i.e. predict that the fare will be whatever
  it was at the previous observation. Fares are step functions that hold flat
  62% of the time (notebook 02), so persistence is a genuinely strong baseline
  here, not a straw man.
- **Spike classification** — predicting the positive class at its base rate,
  which is what any classifier that has learnt nothing would achieve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from flightprice.config import RANDOM_SEED

#: Untuned defaults, shared by every XGBoost fit so that differences between
#: folds and routes reflect the data rather than the settings.
XGB_PARAMS: dict = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.08,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "tree_method": "hist",
    # A fixed thread count rather than -1, so the reported timings are
    # reproducible and do not depend on what else the machine is running.
    "n_jobs": 4,
    "random_state": RANDOM_SEED,
}


def encode_features(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Categorical columns to integer codes, everything to float32.

    Integer codes rather than one-hot: the categoricals here are low-cardinality
    (four routes, six carriers) and trees split on codes without needing the
    expanded representation.
    """
    out = frame[list(cols)].copy()
    for col in out.columns:
        if str(out[col].dtype) in {"category", "object", "string", "bool"}:
            out[col] = out[col].astype("category").cat.codes
    return out.astype("float32")


def make_classifier(scale_pos_weight: float = 1.0, **overrides):
    """Spike classifier. ``scale_pos_weight`` of 1.0 gives the unweighted baseline."""
    import xgboost as xgb

    params = {**XGB_PARAMS, "eval_metric": "logloss", **overrides}
    return xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, **params)


def make_regressor(**overrides):
    """Fare regressor."""
    import xgboost as xgb

    params = {**XGB_PARAMS, "eval_metric": "rmse", **overrides}
    return xgb.XGBRegressor(**params)


class PersistenceRegressor:
    """Predict the previous observed fare.

    The naive forecast for a series that mostly does not move. Implements the
    minimal fit/predict interface the harness expects.
    """

    def __init__(self, lag_col: str = "fareLag1"):
        self.lag_col = lag_col
        self._fallback = 0.0

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "PersistenceRegressor":
        self._fallback = float(np.nanmean(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.lag_col not in X.columns:
            raise KeyError(f"{self.lag_col!r} must be among the features for persistence")
        return X[self.lag_col].fillna(self._fallback).to_numpy(dtype="float64")


class BaseRateClassifier:
    """Predict the training base rate as a constant probability.

    Gives the precision and recall a classifier achieves by learning nothing,
    which is the floor the real models must clear.
    """

    def __init__(self):
        self.rate_ = 0.0

    def fit(self, X, y) -> "BaseRateClassifier":
        self.rate_ = float(np.mean(y))
        return self

    def predict_proba(self, X) -> np.ndarray:
        p = np.full(len(X), self.rate_, dtype="float64")
        return np.column_stack([1 - p, p])
