"""SARIMA / SARIMAX for the daily mean-fare series.

**This model works at a different granularity from the others, and the
comparison has to account for it.** XGBoost and the LSTM predict the fare of an
individual itinerary at an individual search date. SARIMA is a univariate time
series model: it needs one ordered series with a fixed interval, which the raw
observations are not -- they are roughly 63,000 short, overlapping per-flight
trajectories. Fitting a seasonal ARIMA to each is not viable, since the median
trajectory is 20 observations, far too few to identify a weekly seasonal term.

The series modelled here is therefore the **daily mean fare per route**: 174
consecutive days with no gaps. That is a legitimate target -- it is the
route-level price signal a revenue manager would watch -- but it is not the same
target the tree and network families predict. To keep the comparison honest,
notebook 05 also evaluates persistence and XGBoost *on this same daily series*,
so that the three-way comparison is made on like for like.

Forecasts are **one step ahead**: each day in the test window is predicted from
actuals up to the previous day, with the model fitted once on the training
window and never refitted inside the test window. Every model family is
evaluated the same way, so no family is given information the others lack.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from flightprice.evaluation.metrics import regression_metrics
from flightprice.evaluation.splitting import Fold

#: Weekly seasonality. Motivated by measurement, not convention: lag-7
#: autocorrelation of the daily mean fare runs 0.65-0.81 across the four routes,
#: and exceeds lag-1 on the short-haul pair (notebook 01).
SEASONAL_PERIOD: int = 7


def daily_series(
    frame: pd.DataFrame,
    route: str,
    date_col: str = "flightDate",
    value_col: str = "totalFare",
) -> pd.Series:
    """Mean fare per departure date for one route, as a daily-frequency series.

    A gap in the index would break the seasonal term, so the result is reindexed
    onto a complete daily range and interpolated. On this dataset there are no
    gaps, but the guarantee is worth making explicit.
    """
    sub = frame[frame[route_column(frame)] == route]
    series = sub.groupby(date_col, observed=True)[value_col].mean().sort_index()
    series.index = pd.DatetimeIndex(series.index)

    full = pd.date_range(series.index.min(), series.index.max(), freq="D")
    return series.reindex(full).interpolate("time").asfreq("D")


def route_column(frame: pd.DataFrame) -> str:
    """Name of the route column, tolerating either spelling."""
    return "route" if "route" in frame.columns else "Route"


def daily_exog(
    frame: pd.DataFrame,
    route: str,
    columns: tuple[str, ...],
    date_col: str = "flightDate",
) -> pd.DataFrame:
    """Daily exogenous regressors aligned to :func:`daily_series`.

    The calendar features are constant within a departure date, so taking the
    first value per day loses nothing.
    """
    sub = frame[frame[route_column(frame)] == route]
    exog = sub.groupby(date_col, observed=True)[list(columns)].first().sort_index()
    exog.index = pd.DatetimeIndex(exog.index)

    full = pd.date_range(exog.index.min(), exog.index.max(), freq="D")
    return exog.reindex(full).ffill().bfill().astype("float64").asfreq("D")


@dataclass(frozen=True)
class SarimaOrder:
    """A (p,d,q)(P,D,Q,s) specification."""

    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]

    def __str__(self) -> str:
        return f"SARIMA{self.order}{self.seasonal_order}"


def fit_sarimax(
    endog: pd.Series,
    spec: SarimaOrder,
    exog: pd.DataFrame | None = None,
):
    """Fit one SARIMAX model, with convergence warnings suppressed.

    ``enforce_stationarity`` and ``enforce_invertibility`` are left off: with a
    seasonal term on a short series the optimiser otherwise fails outright on
    some candidate orders, which would silently remove them from a grid search
    rather than letting them be judged on fit.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            endog,
            exog=exog,
            order=spec.order,
            seasonal_order=spec.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        return model.fit(disp=False)


def select_order(
    endog: pd.Series,
    candidates: list[SarimaOrder],
    exog: pd.DataFrame | None = None,
) -> tuple[SarimaOrder, pd.DataFrame]:
    """Choose a specification by AIC on the training data only.

    **Selection uses the first fold's training window and nothing later.**
    Choosing an order on the full series would let the test periods influence
    the model specification, which is the same leak as tuning on test.

    Returns:
        ``(best_spec, table_of_all_candidates)``.
    """
    rows = []
    for spec in candidates:
        try:
            res = fit_sarimax(endog, spec, exog)
            rows.append({"spec": str(spec), "aic": res.aic, "bic": res.bic,
                         "converged": bool(res.mle_retvals.get("converged", True))})
        except Exception as exc:  # noqa: BLE001 - a failed candidate is data, not an error
            rows.append({"spec": str(spec), "aic": np.nan, "bic": np.nan,
                         "converged": False, "error": type(exc).__name__})

    table = pd.DataFrame(rows).set_index("spec").sort_values("aic")
    best_label = table["aic"].idxmin()
    best = next(s for s in candidates if str(s) == best_label)
    return best, table


def one_step_forecasts(
    endog: pd.Series,
    train_index: pd.DatetimeIndex,
    test_index: pd.DatetimeIndex,
    spec: SarimaOrder,
    exog: pd.DataFrame | None = None,
) -> np.ndarray:
    """One-step-ahead forecasts across a test window.

    The model is fitted once on the training window. Each test day is then
    forecast one step ahead, and only afterwards is the actual value appended to
    the state, so no forecast ever uses its own target. The parameters are not
    re-estimated inside the window -- that would be refitting on test data.
    """
    train_exog = exog.loc[train_index] if exog is not None else None
    res = fit_sarimax(endog.loc[train_index], spec, train_exog)

    predictions = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for timestamp in test_index:
            step_exog = exog.loc[[timestamp]] if exog is not None else None
            predictions.append(float(res.forecast(steps=1, exog=step_exog).iloc[0]))
            res = res.append(endog.loc[[timestamp]], exog=step_exog, refit=False)

    return np.asarray(predictions)


def evaluate_sarima_folds(
    endog: pd.Series,
    folds: list[Fold],
    fold_dates: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]],
    spec: SarimaOrder,
    exog: pd.DataFrame | None = None,
    label: str = "sarima",
    verbose: bool = True,
) -> pd.DataFrame:
    """Score a specification across folds, using the same windows as every other model.

    Args:
        endog: The daily series.
        folds: Folds, used only for their index numbers.
        fold_dates: ``(train_start, train_end, test_end)`` per fold, so the
            windows match those used for the per-observation models exactly.
    """
    rows = []
    for fold, (train_start, train_end, test_end) in zip(folds, fold_dates):
        train_index = pd.date_range(train_start, train_end, freq="D").intersection(endog.index)
        test_index = pd.date_range(
            train_end + pd.Timedelta(days=1), test_end, freq="D"
        ).intersection(endog.index)

        if len(test_index) == 0 or len(train_index) < 30:
            continue

        predictions = one_step_forecasts(endog, train_index, test_index, spec, exog)
        metrics = regression_metrics(endog.loc[test_index].to_numpy(), predictions)
        metrics.update({"model": label, "fold": fold.index, "n_train": len(train_index)})
        rows.append(metrics)

        if verbose:
            print(
                f"  fold {fold.index}: RMSE={metrics['rmse']:.2f} "
                f"MAE={metrics['mae']:.2f} R2={metrics['r2']:.3f}",
                flush=True,
            )

    return pd.DataFrame(rows)


def default_candidates(seasonal_period: int = SEASONAL_PERIOD) -> list[SarimaOrder]:
    """A small, deliberately restrained grid.

    Kept small because the series is 174 days: an exhaustive search over a short
    series finds orders that fit the training window and generalise poorly, and
    every extra candidate is another chance to overfit the selection criterion.
    """
    specs = []
    for p in (0, 1, 2):
        for q in (0, 1, 2):
            for d in (0, 1):
                specs.append(SarimaOrder((p, d, q), (1, 0, 0, seasonal_period)))
                specs.append(SarimaOrder((p, d, q), (1, 1, 0, seasonal_period)))
    return specs
