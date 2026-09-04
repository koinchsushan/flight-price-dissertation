"""Deciding whether one model is genuinely better than another.

THE RULE THIS FILE ENFORCES
A difference smaller than a model's own round-to-round wobble is not a
difference at all.

WHY THE COMPARISON IS "PAIRED"
Every model was tested on exactly the same five rounds. That matters. Round 1
happens to be hard for every model, round 4 easy for every model. So instead of
comparing two averages and ignoring that, we compare them round by round:

    round   model A   model B   A minus B
      1      0.32      0.30       +0.02
      2      0.35      0.33       +0.02
      3      0.39      0.36       +0.03
      4      0.39      0.38       +0.01
      5      0.42      0.39       +0.03

Consistently ahead by a small amount is far stronger evidence than being ahead
on average once. This is what a paired t-test measures.

THE HONEST CAVEAT, STATED RATHER THAN BURIED
Five rounds is not many, so this test is weak. It can confirm a big, consistent
gap. It cannot prove two models are equally good. So when we say "within fold
noise" we mean "we could not show a difference" -- never "we showed they are
the same". Nine of our nineteen comparisons landed there, and all nine are
reported rather than quietly dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Scores where bigger is better (F1, recall and so on). Everything else --
#: RMSE, MAE -- is an error measurement, where smaller is better. The code needs
#: to know which is which before it can say who won.
HIGHER_IS_BETTER: frozenset[str] = frozenset(
    {"f1", "precision", "recall", "roc_auc", "avg_precision", "r2"}
)


@dataclass(frozen=True)
class Comparison:
    """The answer to "is model A better than model B?", with the workings shown.

    Everything needed to defend the claim is kept here: both averages, the gap,
    how many rounds each model won, how much the gap itself varied, and the
    p-value. The 'verdict' field is the plain-English bottom line.
    """

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
    """Compare two models on one score, round by round.

    Args:
        scores_a, scores_b: Each model's score in each round, labelled by round
            so that round 1 is compared against round 1 and not round 3.
        metric: Which score. Used to work out whether bigger or smaller wins.
        alpha: How sure we insist on being. 0.05 is the usual academic standard:
            we only call it a real difference if there is under a 5% chance of
            seeing a gap this consistent by luck alone.

    Returns:
        The verdict, which is either "distinguishable" (a real difference) or
        "within fold noise" (we could not show one -- which is NOT the same as
        showing the two models are equal).
    """
    from scipy import stats

    # Only compare rounds both models actually ran, and line them up so round 1
    # is set against round 1.
    common = scores_a.index.intersection(scores_b.index)
    a = scores_a.loc[common].to_numpy(dtype="float64")
    b = scores_b.loc[common].to_numpy(dtype="float64")

    higher_better = metric in HIGHER_IS_BETTER
    differences = a - b

    # Simple, readable tally first: how many rounds did A actually win?
    # "9 out of 10 rounds" is often more persuasive to a reader than a p-value.
    wins_a = int((differences > 0).sum() if higher_better else (differences < 0).sum())

    # The formal test. It asks: if the two models were really equally good, how
    # often would we see a gap this consistent purely by chance? A small p-value
    # means "rarely, so the gap is probably real".
    #
    # Guarded because the test is undefined with one round, or if every round
    # produced an identical gap.
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
    """Lay several comparisons out as a table, clearest results at the top.

    Sorted by p-value, so the differences we are most confident about appear
    first and the undecided ones sink to the bottom. This produces the
    nineteen-row table reported in the Results chapter.
    """
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
    """Work backwards from the scores to the raw counts of each kind of mistake.

    The four possible outcomes, in plain terms:
        TP  we said "spike" and there was one          (correct warning)
        FP  we said "spike" and there was not          (false alarm)
        FN  we said nothing and a spike happened       (missed it)
        TN  we said nothing and nothing happened       (correct silence)

    The model notebooks saved scores rather than every individual prediction, so
    the counts are rebuilt from the scores. This is exact arithmetic, not an
    estimate -- precision and recall are defined from these counts, so the
    definitions can simply be rearranged:

        recall    = TP / (all real spikes)   ->  TP = recall x real spikes
        precision = TP / (all warnings)      ->  FP = TP x (1 - precision) / precision

    We need these counts because the cost analysis below has to weigh the two
    kinds of mistake against each other, and scores alone cannot do that.
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
    """Work out what a model's mistakes would cost, given how bad each kind is.

    This function produces the most interesting finding in the project, so it is
    worth being comfortable explaining.

    The two mistakes do not hurt equally:
      - A MISSED spike leaves the traveller unwarned. They pay the higher fare.
      - A FALSE ALARM tells them to book when waiting would have been fine. They
        lose a saving they might have got.

    Which is worse depends entirely on the product, not on our data. So instead
    of inventing pound values -- which would be made up -- we sweep the RATIO.
    "What if a miss is twice as bad as a false alarm? Ten times? Thirty?"

    Doing that revealed three bands. Below 5x, the unweighted model is cheapest.
    Between 5x and 27x, the weighted one. Above 28x, the LSTM -- the model that
    lost every single statistical comparison in this project -- becomes the
    right one to deploy, because it catches the most spikes.

    Being statistically best and being commercially right are two different
    things, and this function is what demonstrates it.
    """
    return (
        confusion["FN"] * miss_cost_ratio * false_alarm_cost
        + confusion["FP"] * false_alarm_cost
    )
