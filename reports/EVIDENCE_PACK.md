# Evidence Pack — CS7P01 Dissertation

**Generated from `reports/results/` and the processed data by `scripts/build_evidence_pack.py`. Do not edit by hand — regenerate.**

Purpose: the write-up is written *from this document*, so that no chapter asserts something the results do not support. Every number below is read from a results file rather than transcribed. Where a claim is not supported, that is stated explicitly, and those cases matter as much as the supported ones.

Written for the write-up phase, and usable standalone — a chapter can be drafted from this document without opening the repository.

## 1. Answers to the research questions

| Question | Answer | Evidence |
|---|---|---|
| RQ1 per-observation | **XGBoost = LSTM (not separable)** | RMSE 47.0 vs 48.1, p > 0.05 |
| RQ1 daily series | **SARIMA** | median RMSE 19.4 vs 29.9, p < 0.05 over 20 cells |
| RQ1 vs naive | **split decision** | XGBoost wins RMSE, persistence wins MAE, 10/10 folds |
| RQ2 best family | **XGBoost** | F1 0.350 vs 0.265, 5/5 folds, p < 0.01 |
| RQ2 best configuration | **not separable** | F1 0.350 / 0.374 / 0.377, all p > 0.05 |
| RQ2 deployment choice | **depends on cost ratio** | unweighted cheapest below 5x, weighted 5-27x, LSTM above 28x |
| RQ3 strongest factor | **daysBeforeDeparture** | ~2x more important on short-haul |
| RQ3 holidays | **not estimable** | 4 holidays in window; SARIMAX degrades on the only holiday fold |

## 2. Statistical testing — what may be claimed

**10 of 19 comparisons are distinguishable at the 5% level; 9 fall within fold noise.**

Paired *t*-test over folds, since every model saw identical folds. With five folds the test has little power: *within fold noise* means **not shown to differ**, never shown to be equal.

### Differences that survive

| Metric | A | B | A wins | p | Better |
|---|---|---|---|---|---|
| rmse | SARIMA | persistence | 18/20 | 0.0000 | **SARIMA** |
| mae | SARIMA | persistence | 18/20 | 0.0001 | **SARIMA** |
| f1 | XGB unweighted | LSTM | 5/5 | 0.0002 | **XGB unweighted** |
| f1 | XGB weighted | LSTM | 5/5 | 0.0018 | **XGB weighted** |
| roc_auc | XGB weighted | LSTM | 5/5 | 0.0025 | **XGB weighted** |
| rmse | XGBoost (both routes) | persistence (both routes) | 9/10 | 0.0091 | **XGBoost (both routes)** |
| rmse | SARIMA | xgboost (daily) | 15/20 | 0.0155 | **SARIMA** |
| mae | SARIMA | xgboost (daily) | 17/20 | 0.0200 | **SARIMA** |
| mae | LSTM | persistence | 0/5 | 0.0217 | **persistence** |
| mae | XGBoost (both routes) | persistence (both routes) | 0/10 | 0.0456 | **persistence (both routes)** |

### Differences that do NOT survive — do not claim these

| Metric | A | B | Means | p |
|---|---|---|---|---|
| mae | XGBoost | persistence | 27.886 vs 22.463 | 0.1311 |
| f1 | XGB unweighted | XGB weighted | 0.374 vs 0.350 | 0.1541 |
| rmse | SARIMA | sarimax | 25.064 vs 29.268 | 0.1778 |
| mae | SARIMA | sarimax | 19.614 vs 22.287 | 0.1914 |
| rmse | XGBoost | persistence | 46.986 vs 50.442 | 0.2275 |
| rmse | LSTM | persistence | 48.066 vs 50.442 | 0.2551 |
| mae | XGBoost | LSTM | 27.886 vs 29.614 | 0.3622 |
| rmse | XGBoost | LSTM | 46.986 vs 48.066 | 0.6229 |
| f1 | XGB wtd+tuned | XGB unweighted | 0.377 vs 0.374 | 0.8891 |

## 3. Model results

### Spike classification — per-observation, long-haul

| Model | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|
| XGBoost unweighted | 0.707 | 0.255 | 0.374 | 0.864 |
| XGBoost weighted | 0.235 | 0.690 | 0.350 | 0.860 |
| XGBoost weighted + tuned | 0.420 | 0.375 | 0.377 | 0.825 |
| LSTM | 0.161 | 0.754 | 0.265 | 0.824 |
| Base rate (floor) | 0.000 | 0.000 | 0.000 | 0.500 |

### Fare regression — per-observation, long-haul

| Model | RMSE | MAE | R² |
|---|---|---|---|
| XGBoost | 46.99 | 27.89 | 0.921 |
| LSTM | 48.07 | 29.61 | 0.918 |
| Persistence (naive) | 50.44 | 22.46 | 0.905 |

### Fare regression — daily mean fare, all four routes

| Model | RMSE (mean) | RMSE (median) | MAE | R² |
|---|---|---|---|---|
| sarima | 25.06 | 19.40 | 19.61 | 0.532 |
| sarimax | 29.27 | 19.89 | 22.29 | 0.275 |
| xgboost (daily) | 35.24 | 29.86 | 29.24 | 0.095 |
| persistence | 44.00 | 40.24 | 33.83 | -0.302 |

### Short-haul

| Model | Task | Score |
|---|---|---|
| XGBoost | spike F1 | 0.461 |
| LSTM | spike F1 | 0.216 |
| XGBoost | fare RMSE | 21.84 |
| LSTM | fare RMSE | 51.71 |
| Persistence | fare RMSE | 26.44 |

## 4. Dataset facts

| Fact | Value | Verified from |
|---|---|---|
| Rows after coach filter | 2,152,628 | `coach_filtered.parquet` |
| Maximum fare (coach only) | $2,281.61 | `coach_filtered.parquet` |
| Median fare | $261.11 | `coach_filtered.parquet` |
| Rows after censoring cutoff | 1,943,082 | `spikes_labelled.parquet` |
| Labelable observations | 1,389,510 (71.5% of rows) | `spikes_labelled.parquet` |
| Spike events | 108,495 | `spikes_labelled.parquet` |
| Spike rate (of labelable) | 7.81% | computed |
| Class imbalance | 1 : 11.8 | computed |
| Modelling rows | 1,389,510 | `model_frame.parquet` |
| Distinct flights | 63,301 | `model_frame.parquet` |

## 5. Figure index

| Figure | Chapter | RQ | Caption |
|---|---|---|---|
| `01_fare_by_cabin_class.png` | Methodology | — | Fare distribution and spread by cabin group. Non-coach itineraries are 0.17% of rows but occupy the upper tail, which is the justification for the coach-only filter. |
| `01_fare_by_route.png` | Results | RQ3 | Fare distribution and observation counts per route. The long-haul and short-haul pairs separate clearly in both level and spread. |
| `01_fare_distribution_before_after.png` | Methodology | — | Fare distribution before and after the coach-only filter. The maximum falls from $4,782.60 to $2,281.61 while the median is unchanged to the cent. |
| `01_right_censoring.png` | Methodology | — | Closest observation to departure, by departure date. Flat at one day until 12 October, then rising linearly — the evidence for the censoring cutoff. |
| `01_temporal_coverage.png` | Methodology | — | Mean fare by departure date with the four in-window federal holidays and the summer-season proxy. Shows coverage running to 19 November 2022. |
| `02_example_trajectories.png` | Methodology | — | Four individual fare trajectories. Fares hold flat then step between booking buckets, which is why the borrowed 2-sigma definition needed re-examination. |
| `02_spike_rate_by_horizon_and_route.png` | Results | RQ2/RQ3 | Spike event rate by booking horizon and route. Rates concentrate near departure (16.0% at 1-3 days against 5.2% at 22-30), independently validating the censoring cutoff. |
| `02_window_sweep.png` | Methodology | — | Labelable coverage and spike rate against rolling-window size. Coverage saturates near ten observations, which is the window adopted. |
| `03_calendar_features.png` | Results | RQ3 | Mean fare by day of week and by distance from the nearest federal holiday, per route. |
| `03_feature_importance.png` | Results | RQ3 | Feature importance by group and individually, fold 1 — diagnostic only. Superseded for reporting by the fold-averaged version in figure 04. |
| `03_itinerary_features.png` | Results | RQ3 | Mean fare by scheduled departure hour and by operating carrier. |
| `04_classification_folds.png` | Results | RQ2 | Spike classification across rolling-origin folds for three imbalance configurations. The fold-to-fold spread exceeds the differences between configurations. |
| `04_feature_importance_by_route.png` | Results | RQ3 | Feature importance averaged over five folds, long-haul against short-haul. The primary evidence for research question 3. |
| `04_regression_folds.png` | Results | RQ1 | Fare regression against the persistence baseline by fold. XGBoost leads on RMSE while persistence leads on MAE. |
| `05_daily_series.png` | Methodology | — | Daily mean fare per route with the five test windows shaded. The series SARIMA models. |
| `05_diagnostics.png` | Methodology | — | Box-Jenkins diagnostics for JFK-LAX: level, first difference, ACF and PACF. Spikes at lags 7, 14 and 21 justify the weekly seasonal term. |
| `05_model_comparison.png` | Results | RQ1 | Daily mean fare: SARIMA, SARIMAX, XGBoost and persistence under one-step-ahead forecasting, with fold-level detail. |
| `06_family_comparison.png` | Results | RQ1/RQ2 | LSTM against XGBoost across folds on RMSE, MAE and spike F1, at per-observation granularity. |
| `06_sequence_lengths.png` | Methodology | — | Per-flight trajectory length distribution. Determines how much history the recurrent state has, and motivates length-bucketed batching. |
| `07_commercial.png` | Discussion | RQ2 | Expected cost by operating point across the missed-spike to false-alarm ratio, with the precision/recall positions of each configuration. |
| `07_rq1_granularity.png` | Results | RQ1 | The two granularities rank the families differently and are not comparable to each other. Central to the RQ1 answer. |
| `07_significance.png` | Results | all | Every model comparison against the paired fold test. Bars past the dashed line are distinguishable; those short of it are within fold noise. |

## 6. Methodology decisions, and the evidence for each

Each was decided by measurement. Written up as *evidence, then decision* rather than decision alone.

| Decision | Evidence | Notebook |
|---|---|---|
| Coach-only filter | 0.17% of rows hold the fare tail; median unchanged to the cent | 01 |
| Censoring cutoff at 12 Oct | Departures after it are never observed near departure, where spikes concentrate (16.0% vs 5.2%) | 01, 02 |
| Spike window of 10 observations | The inherited 20 would be unusable for 65% of flights; coverage saturates at 10 | 02 |
| Spikes labelled as events, not states | One in three naive labels is a continuation of a spike already under way | 02 |
| No SMOTE/ADASYN | Minority class is 108,495 at 1:11.8 — far outside the regime that guidance addresses | 02 |
| Rolling-origin, 5 x 14 days | Positive rate drifts 6.70% to 8.91% across the period; every fold carries 9,600+ events | 03 |
| Split on departure date, not search date | Cutting on searchDate leaves 61.8% of test flights in training | 03 |
| totalFare excluded from features | Including it lifts ROC AUC 0.815 to 0.997 — the signature of leakage | 03 |
| SARIMA on daily means | 63,301 trajectories with median length 20 cannot identify a weekly seasonal term | 05 |
| Weekly seasonal period | Lag-7 autocorrelation 0.65-0.81 across routes | 01, 05 |
| LSTM on per-flight sequences | Flattened rows would discard the structure the architecture exists for | 06 |
| Feature clipping for the LSTM | Unclipped, time-index features extrapolate: predicted $238.6 against actual $148.7 | 06 |

## 7. Limitations, each tied to a measurement

1. **The spike definition partly measures fare-bucket transitions.** 62% of consecutive observations repeat the previous fare exactly, so the rolling standard deviation collapses inside a flat run and any bucket change clears 2σ regardless of economic size. Adapted from a commodities paper whose series do not behave this way. *(nb 02)*
2. **The window is too short to estimate holiday effects.** Four federal holidays fall inside it, so a model forecasting the fourth has seen at most three; the only test window containing a holiday was the only one where SARIMAX degraded, by 47.9 RMSE. *(nb 05)*
3. **Thanksgiving is missed by five days.** Departures end 19 November 2022. Seasonal claims are limited to summer and the four in-window holidays. *(nb 01)*
4. **Late departures are right-censored and excluded**, retaining 90.3% of rows. *(nb 01)*
5. **Five folds give the significance test little power.** Nine comparisons are 'not shown to differ', which is not the same as equivalent. *(nb 07)*
6. **The feature set is not neutral between families.** Designed for trees; the LSTM needed clipping that the trees did not. *(nb 06)*
7. **No model was tuned.** Sensible defaults throughout — the fair basis for comparing families, but an understatement of what any one could achieve. *(nb 04-06)*
8. **Two US route pairs over six months.** Not a general claim about airline pricing.

## 8. Claims that must NOT be made

The most likely way to lose marks is to overstate. Each of these is contradicted by the project's own results.

| Do not claim | Why |
|---|---|
| "XGBoost is the best model" | Only on spike classification. It ties the LSTM on fare regression and loses to SARIMA on the daily series. |
| "The LSTM performed worst" | It is the cheapest operating point when a missed spike costs more than 28x a false alarm. |
| "Model X beats model Y" without a test | Nine of nineteen comparisons fall within fold noise. |
| "XGBoost beats the naive baseline" | On RMSE yes; on MAE persistence wins every single fold. |
| Comparing SARIMA's RMSE with XGBoost's per-observation RMSE | Different targets. Daily averaging removes most of the variance. |
| "Holidays raise fares by X" | Not estimable from four holidays. |
| "Season drives spikes" | `isSummerSeason` scores exactly zero importance for spike timing. |
| "Spikes can be predicted reliably" | F1 ≈ 0.35. At useful recall most warnings are false alarms. |
| Any accuracy figure as a headline | A do-nothing classifier is over 90% accurate here. |
