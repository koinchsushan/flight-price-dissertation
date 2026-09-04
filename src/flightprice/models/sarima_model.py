"""The classical statistics model, working on each route's daily average fare.

WHAT SARIMA IS
The oldest of the three approaches, and the most explainable. It predicts
tomorrow from a formula combining recent values, recent errors, and a repeating
weekly pattern. The letters stand for Seasonal AutoRegressive Integrated Moving
Average. SARIMAX is the same thing with extra columns bolted on -- here, holiday
and season flags.

THE ONE THING TO BE READY TO EXPLAIN
This model predicts something DIFFERENT from the other two, and the write-up is
careful about it.

XGBoost and the LSTM predict one specific flight's price on one specific day.
SARIMA cannot do that. It needs a single, evenly spaced series -- one number per
day, no gaps. Our data is nothing like that: it is 63,000 short, overlapping
price histories, and the typical one is only 20 entries long. That is far too
short to detect a weekly rhythm.

So SARIMA is given the DAILY AVERAGE FARE for each route instead: 174
consecutive days, one number per day. That is a perfectly sensible thing to
forecast -- it is what an airline analyst would actually watch -- but it is an
easier target, because averaging smooths away most of the variation.

This is why SARIMA's error looks so much smaller ($19 against $47) and why the
dissertation refuses to put those two numbers side by side. To make the
comparison fair, notebook 05 also runs XGBoost and the naive baseline on this
same daily series, so the three are judged on like for like.

HOW THE FORECASTS ARE MADE
One day at a time. The model is fitted once on the training period, then
predicts the first test day. Only AFTER that prediction is recorded does it get
told what actually happened, before predicting the next day. It is never
re-fitted during testing -- that would be learning from the answers.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from flightprice.evaluation.metrics import regression_metrics
from flightprice.evaluation.splitting import Fold

#: The repeating pattern is weekly (7 days).
#:
#: This was measured rather than assumed. We checked how strongly each day's
#: average fare relates to the fare seven days earlier, and got 0.65 to 0.81
#: across the four routes -- strong. On the short shuttle routes that link is
#: even stronger than the link to YESTERDAY, which says those fares are driven
#: more by which day of the week it is than by recent drift.
SEASONAL_PERIOD: int = 7


def daily_series(
    frame: pd.DataFrame,
    route: str,
    date_col: str = "flightDate",
    value_col: str = "totalFare",
) -> pd.Series:
    """Collapse one route down to a single number per day: its average fare.

    This is the series SARIMA actually models.

    A missing day would break the weekly pattern, so afterwards we lay the
    result on a complete run of dates and fill any hole by interpolating between
    its neighbours. As it happens this dataset has no gaps, but the code
    guarantees it rather than trusting it.
    """
    sub = frame[frame[route_column(frame)] == route]
    series = sub.groupby(date_col, observed=True)[value_col].mean().sort_index()
    series.index = pd.DatetimeIndex(series.index)

    full = pd.date_range(series.index.min(), series.index.max(), freq="D")
    return series.reindex(full).interpolate("time").asfreq("D")


def route_column(frame: pd.DataFrame) -> str:
    """Find the route column, whether it was saved as "route" or "Route".

    A small convenience so the notebooks work either way, rather than failing on
    a capital letter.
    """
    return "route" if "route" in frame.columns else "Route"


def daily_exog(
    frame: pd.DataFrame,
    route: str,
    columns: tuple[str, ...],
    date_col: str = "flightDate",
) -> pd.DataFrame:
    """Line up the extra columns (holidays, season) with the daily series.

    "Exogenous" just means extra inputs that come from outside the fare series
    itself. These are what turn SARIMA into SARIMAX.

    Whether a date is a holiday is the same for every flight on that date, so
    taking the first value of the day loses nothing at all.
    """
    sub = frame[frame[route_column(frame)] == route]
    exog = sub.groupby(date_col, observed=True)[list(columns)].first().sort_index()
    exog.index = pd.DatetimeIndex(exog.index)

    full = pd.date_range(exog.index.min(), exog.index.max(), freq="D")
    return exog.reindex(full).ffill().bfill().astype("float64").asfreq("D")


@dataclass(frozen=True)
class SarimaOrder:
    """One SARIMA configuration, written the standard way: (p,d,q)(P,D,Q,s).

    Roughly what the numbers mean:
        p  how many recent days to look back at
        d  how many times to difference the series to remove a trend
           (differencing = model the CHANGE rather than the level)
        q  how many recent errors to carry forward
        P,D,Q  the same three ideas again, but for the weekly pattern
        s  the length of that pattern, which is 7 here

    Our chosen configuration for JFK-LAX came out as SARIMA(2,1,1)(1,1,0,7).
    """

    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]

    def __str__(self) -> str:
        return f"SARIMA{self.order}{self.seasonal_order}"


def fit_sarimax(
    endog: pd.Series,
    spec: SarimaOrder,
    exog: pd.DataFrame | None = None,
):
    """Fit one SARIMA configuration to a series.

    Two settings are deliberately switched off. They are mathematical tidiness
    checks, and on a short series with a weekly pattern they make the fitting
    routine give up entirely on some perfectly reasonable configurations. That
    would quietly drop those candidates from the search instead of letting them
    compete on how well they actually fit -- so we turn the checks off and judge
    every candidate on results.

    Warnings are silenced because a grid search over 36 configurations produces
    pages of "this one did not converge nicely" messages, which drown out the
    output that matters.
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
    """Try every candidate configuration and keep the best one.

    "Best" is judged by AIC, a standard score that rewards fitting the data well
    while penalising unnecessary complexity. It stops us picking an elaborate
    model that has simply memorised the training period.

    IMPORTANT: the choice is made using the FIRST round's training data only,
    and nothing later. Choosing the configuration by looking at the whole series
    would let the test periods influence the model -- the same kind of cheating
    as tuning on the test set, just one step removed.

    Returns:
        The winning configuration, and the full table so the choice is auditable.
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
    """Walk through the test period one day at a time, predicting each day ahead.

    The order of operations here is what keeps it honest, so it is worth
    spelling out:

        1. Fit the model once, on the training period only.
        2. Predict day 1 of the test period. Write the prediction down.
        3. NOW tell the model what day 1 actually was.
        4. Predict day 2. Write it down. Then reveal day 2. And so on.

    No prediction is ever made with knowledge of the day it is predicting. And
    the model's internal settings are frozen after step 1 -- it absorbs each new
    day's value, but it is never re-fitted, which would amount to training on
    the test data.
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
    """Score SARIMA over the same five rounds used for every other model.

    The dates are passed in explicitly rather than recalculated, which guarantees
    SARIMA is tested on exactly the same calendar windows as XGBoost and the
    LSTM. If the windows drifted even slightly, the comparison would be between
    different test periods rather than between different models.

    Args:
        endog: The daily average fare series ("endogenous" = the thing being
            predicted, as opposed to the extra inputs).
        folds: The rounds. Only their numbers are used here.
        fold_dates: Start and end dates for each round.
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
    """The list of configurations to try. Deliberately short, and here is why.

    The series is only 174 days long. Search through hundreds of configurations
    on a series that short and you will certainly find one that fits the
    training period beautifully and then falls apart on new data. Every extra
    candidate is another lottery ticket in a lottery you do not want to win.

    So we try 36: a few sensible values for the recent-days and recent-errors
    settings, with and without differencing, each with a weekly pattern.
    """
    specs = []
    for p in (0, 1, 2):
        for q in (0, 1, 2):
            for d in (0, 1):
                specs.append(SarimaOrder((p, d, q), (1, 0, 0, seasonal_period)))
                specs.append(SarimaOrder((p, d, q), (1, 1, 0, seasonal_period)))
    return specs
