"""Comparing model families over paired folds, and appraising the trade-off.

Rolling-origin validation produces one score per fold per model, and because
every model saw the identical folds those scores are **paired**. That pairing is
what makes a difference testable: comparing two means while ignoring it would
throw away the fact that fold 1 is hard for everyone and fold 4 easy for
everyone.

**A difference smaller than the fold-to-fold variation is not a difference.**
This is the whole reason the project chose rolling-origin over a single split
(García Crespi et al., 2026): a single split reports one number per model and
offers no way to tell a real gap from a lucky cut date. Every comparative claim
in the write-up is therefore run through :func:`paired_comparison` and reported
with its verdict, including the ones that fail.

A caveat stated rather than hidden: with five folds the paired t-test has very
little power. It can confirm a large, consistent difference; it cannot establish
that two models are equivalent. "Within noise" here means *not shown to differ*,
never *shown to be the same*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Metrics where a larger value is better.
HIGHER_IS_BETTER: frozenset[str] = frozenset(
    {"f1", "precision", "recall", "roc_auc", "avg_precision", "r2"}
)


@dataclass(frozen=True)
class Comparison:
    """Result of comparing two models on one metric across paired folds."""

    metric: str
    name_a: str
    name_b: str
    mean_a: float
    mean_b: float
    difference: float          # a - b, signed
    folds_won_by_a: int
    n_folds: int
    sd_of_differences: float
    t_statistic: float
    p_value: float
    better: str
    verdict: str

    def __str__(self) -> str:
        return (
            f"{self.metric}: {self.name_a} {self.mean_a:.3f} vs "
            f"{self.name_b} {self.mean_b:.3f} — {self.verdict}"
        )


def paired_comparison(
    scores_a: pd.Series,
    scores_b: pd.Series,
    metric: str,
    name_a: str = "A",
    name_b: str = "B",
    alpha: float = 0.05,
) -> Comparison:
    """Compare two models on one metric, over folds they both ran.

    Args:
        scores_a, scores_b: Per-fold scores, indexed by fold so that they align.
        metric: Used to decide the direction of "better".
        alpha: Significance level for the paired t-test.

    Returns:
        A :class:`Comparison`. ``verdict`` is either "distinguishable" or
        "within fold noise" — the latter meaning the difference was not shown,
        not that the models were shown to be equal.
    """
    from scipy import stats

    common = scores_a.index.intersection(scores_b.index)
    a = scores_a.loc[common].to_numpy(dtype="float64")
    b = scores_b.loc[common].to_numpy(dtype="float64")

    higher_better = metric in HIGHER_IS_BETTER
    differences = a - b
    wins_a = int((differences > 0).sum() if higher_better else (differences < 0).sum())

    if len(common) > 1 and np.std(differences) > 0:
        t_statistic, p_value = stats.ttest_rel(a, b)
    else:
        t_statistic, p_value = float("nan"), float("nan")

    a_is_better = (a.mean() > b.mean()) if higher_better else (a.mean() < b.mean())
    significant = bool(p_value == p_value and p_value < alpha)

    return Comparison(
        metric=metric,
        name_a=name_a,
        name_b=name_b,
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
        difference=float(a.mean() - b.mean()),
        folds_won_by_a=wins_a,
        n_folds=len(common),
        sd_of_differences=float(np.std(differences, ddof=1)) if len(common) > 1 else float("nan"),
        t_statistic=float(t_statistic),
        p_value=float(p_value),
        better=(name_a if a_is_better else name_b) if significant else "—",
        verdict="distinguishable" if significant else "within fold noise",
    )


def comparison_table(comparisons: list[Comparison]) -> pd.DataFrame:
    """Tabulate several comparisons, most clearly separated first."""
    rows = [
        {
            "metric": c.metric,
            "model A": c.name_a,
            "model B": c.name_b,
            "mean A": round(c.mean_a, 4),
            "mean B": round(c.mean_b, 4),
            "difference": round(c.difference, 4),
            "A wins": f"{c.folds_won_by_a}/{c.n_folds}",
            "sd of diffs": round(c.sd_of_differences, 4),
            "p": round(c.p_value, 4) if c.p_value == c.p_value else np.nan,
            "better": c.better,
            "verdict": c.verdict,
        }
        for c in comparisons
    ]
    table = pd.DataFrame(rows)
    return table.sort_values("p", na_position="last").reset_index(drop=True)


def confusion_from_metrics(
    precision: float, recall: float, n_positive: int, n_total: int
) -> dict[str, float]:
    """Recover a confusion matrix from reported precision, recall and counts.

    The model notebooks stored metrics rather than raw predictions, and the
    counts are recoverable exactly:

    ``TP = recall x P``, ``FN = P - TP``, ``FP = TP(1 - precision)/precision``.
    """
    true_positive = recall * n_positive
    false_negative = n_positive - true_positive
    false_positive = (
        true_positive * (1.0 - precision) / precision if precision > 0 else 0.0
    )
    true_negative = n_total - true_positive - false_negative - false_positive

    return {
        "TP": true_positive,
        "FP": false_positive,
        "FN": false_negative,
        "TN": max(true_negative, 0.0),
    }


def expected_cost(
    confusion: dict[str, float], miss_cost_ratio: float, false_alarm_cost: float = 1.0
) -> float:
    """Expected cost of an operating point, per test window.

    Absolute currency values are deliberately avoided. What matters is the
    *ratio* between the two errors, and that ratio is a property of the
    application rather than of the data:

    - A **missed spike** means the traveller is not warned and pays the higher
      fare.
    - A **false alarm** means they are told to book when they need not have, and
      may forgo a later saving.

    Sweeping the ratio shows which configuration a product should choose, and at
    what point that choice changes.
    """
    return (
        confusion["FN"] * miss_cost_ratio * false_alarm_cost
        + confusion["FP"] * false_alarm_cost
    )
