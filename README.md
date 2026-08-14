# Predicting Seasonal Demand and Price Spikes in Flight Ticket Pricing

MSc Dissertation (CS7P01) — London Metropolitan University.

A three-way comparison of model families (SARIMA/SARIMAX, XGBoost, LSTM) on the tasks of
**forecasting flight fares** and **predicting price spikes before they occur**, evaluated on a
long-haul and a short-haul US domestic route.

---

## Research questions

1. How well can three different model families predict flight prices on these routes?
2. Can a price spike be predicted before it happens, and which model does this best?
3. Which factors (timing, holidays, airline, stops) affect price the most, and does this
   differ between a long-haul and a short-haul route?

## Data

**Source:** [`dilwong/flightprices`](https://www.kaggle.com/datasets/dilwong/flightprices) —
one-way US domestic itineraries scraped from Expedia, **16 Apr – 5 Oct 2022**.

The full release is 82,138,753 rows / ~31 GB and will exhaust memory under a naive
`pandas.read_csv`. This project works from a pre-extracted subset of four directional
routes, `data/jfk_lax_bos_lga.csv` (2,156,316 rows, ~662 MB), which loads normally.

| Route | Role | Rows (post-filter) |
|---|---|---:|
| LAX → JFK | long-haul, primary | 624,358 |
| JFK → LAX | long-haul, primary | 603,909 |
| BOS → LGA | short-haul, secondary | 483,087 |
| LGA → BOS | short-haul, secondary | 441,274 |

**Cabin filter.** 99.83% of the subset is pure coach. The 0.17% of business/first/premium
and mixed-cabin itineraries occupy the extreme tail of the fare distribution and are removed
before modelling — this drops the maximum fare from \$4,782.60 to \$2,281.61 without
materially changing the quartiles.

> The dataset is not committed to this repository (it exceeds GitHub's 100 MB file limit).
> See [`data/README.md`](data/README.md) for how to obtain it and reproduce the subset.

## Method

**Features.** Days before departure (the primary temporal driver, per revenue-management
theory), day of week, non-stop flag, US federal holiday flag, and a summer-season proxy
(Memorial Day → Labor Day, standing in for school-term dates).

**Spike definition.** A fare deviating from its rolling mean by more than 2 rolling standard
deviations, adapted from Lee et al. (2024). The window size is re-tuned empirically rather
than inherited, since the relevant unit here is days-before-departure rather than calendar
days. Framed as **binary classification**, not point-price regression.

**Class imbalance.** SMOTE / ADASYN / undersampling, selected against the observed
minority-class count rather than assumed in advance.

**Validation.** Rolling-origin (walk-forward). García Crespi et al. (2026) show model
rankings can *reverse* between a single chronological split and rolling-origin validation on
a structurally similar comparison — a single split is not sufficient to support a
"which model wins" claim.

**Metrics.** RMSE and MAE for price level; precision, recall and F1 for spike
classification — deliberately **not accuracy**, which is trivially high under this class
distribution.

## Repository layout

```
├── src/flightprice/        Reusable pipeline code
│   ├── config.py           Paths, seeds, dataset constants, spike parameters
│   ├── data/               Loading and cleaning
│   ├── features/           Feature engineering
│   ├── spikes/             Spike labelling and window tuning
│   ├── models/             SARIMA, XGBoost, LSTM
│   └── evaluation/         Metrics and model comparison
├── notebooks/              One notebook per phase — narrative, analysis, figures
├── reports/figures/        Generated figures (300 dpi PNG)
└── data/                   Not committed; see data/README.md
```

Logic lives in `src/`; notebooks import it and carry the narrative. This keeps the analysis
readable end-to-end while avoiding copy-pasted code between phases.

## Setup

Requires Python ≥ 3.12. On macOS, XGBoost additionally needs the OpenMP runtime.

```bash
brew install libomp          # macOS only
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Then place `jfk_lax_bos_lga.csv` in `data/` (see [`data/README.md`](data/README.md)) and run
the notebooks in numerical order.

## Reproducibility

All stochastic components share a single seed, exposed as `RANDOM_SEED` in
[`src/flightprice/config.py`](src/flightprice/config.py) and applied via `set_seeds()` at the
top of each notebook. Dependencies are pinned in `requirements.txt`.

## Scope and limitations

- The observation window is **April–October 2022 only**. There is no November or December
  data, so no Thanksgiving or Christmas signal exists. Seasonality findings are scoped to
  summer travel and the four in-window federal holidays, and are not claims about full-year
  seasonality.
- The short-haul route (BOS–LGA) receives the same pipeline and feature set as the long-haul
  route but is **not independently tuned** — a deliberate scope decision given the project
  timeline, stated rather than hidden.
- The routes are US domestic rather than the UK-originating routes of the original proposal.
  No free longitudinal fare dataset with the required repeated-search structure exists for
  any UK route; the research questions are unchanged.
- SMOTE interpolates in feature space and does not inherently respect temporal ordering.
  Where plain SMOTE is used, this is recorded as a stated simplification against
  time-series-aware alternatives.

## Key references

- Abdella, J.A., Zaki, N.M., Shuaib, K. and Khan, F. (2021) 'Airline ticket price and demand
  prediction: A survey', *Journal of King Saud University – Computer and Information
  Sciences*, 33, pp. 375–391.
- Chawla, N.V., Bowyer, K.W., Hall, L.O. and Kegelmeyer, W.P. (2002) 'SMOTE: Synthetic
  Minority Over-sampling Technique', *JAIR*, 16, pp. 321–357.
- Chen, T. and Guestrin, C. (2016) 'XGBoost: A Scalable Tree Boosting System', *KDD 2016*,
  pp. 785–794.
- Degife, W.A. and Lin, B.-S. (2023) 'Deep-Learning-Powered GRU Model for Flight Ticket Fare
  Forecasting', *Applied Sciences*, 13(10), 6032.
- García Crespi, F., Yubero Funes, E. and Alfosea Simón, M. (2026) 'Rolling-Origin Validation
  Reverses Model Rankings in Multi-Step PM10 Forecasting', arXiv:2603.20315.
- He, H., Bai, Y., Garcia, E.A. and Li, S. (2008) 'ADASYN: Adaptive Synthetic Sampling
  Approach for Imbalanced Learning', *IJCNN 2008*, pp. 1322–1328.
- Hochreiter, S. and Schmidhuber, J. (1997) 'Long Short-Term Memory', *Neural Computation*,
  9(8), pp. 1735–1780.
- Lee, N. *et al.* (2024) 'Metal Price Spike Prediction via a Neurosymbolic Ensemble
  Approach', arXiv:2410.12785.
- Wong, P. *et al.* (2023) 'Using Spark Machine Learning Models to Perform Predictive Analysis
  on Flight Ticket Pricing Data', arXiv:2310.07787.
