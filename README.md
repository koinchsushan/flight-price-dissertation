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
one-way US domestic itineraries scraped from Expedia.

Two windows matter and are easily conflated. Fares were **scraped** between 16 Apr and
5 Oct 2022, but each search returned flights departing up to two months later, so **departure
dates run from 17 Apr to 19 Nov 2022** — 45 days beyond the final scrape.

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

**Spike definition.** A fare deviating from the rolling mean of its own trajectory by more
than 2 rolling standard deviations, adapted from Lee et al. (2024). A trajectory is one
itinerary (`legId`) observed across successive search dates. Rolling statistics are
**causal** — computed from prior observations only — so that labels remain usable at
prediction time. The window was tuned empirically to **10 observations**: the 20 used for
daily commodity closes would have been unusable for 65% of flights, whose median trajectory
is 9 observations. Framed as **binary classification**, not point-price regression.

Spikes are labelled as **events** (the first observation of each run) rather than as
sustained states. One fare rise remains elevated against a trailing window for several
subsequent observations, so a per-observation label counts a single price change many times
and would train a model to detect "this fare is currently high" instead of "this fare is
about to jump".

**Class imbalance.** The observed minority class is 108,495 spike events against 1,281,015
negatives — a ratio of **1:11.8**, moderate rather than severe. Synthetic oversampling
(SMOTE/ADASYN) is therefore not used: it exists to manufacture signal for minority classes
too small to learn from, which is not the situation here, and it would distort temporal
ordering. Class weighting and decision-threshold tuning are used instead, measured against an
unweighted baseline.

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

- **No Thanksgiving or Christmas signal is observable.** Departure dates end on 19 Nov 2022,
  five days short of Thanksgiving. Seasonality findings are scoped to summer travel and the
  four in-window federal holidays, and are not claims about full-year seasonality.
- **Late departures are right-censored.** Scraping stopped on 5 Oct, so a flight departing in
  mid-November was never observed closer than 27 days out. Because the final approach to
  departure is where revenue-management fare surges occur, spike analysis is restricted to
  departures on or before **12 Oct 2022** — the last date whose price trajectory is observable
  down to one day before departure. This retains 90.3% of rows. Including the censored
  flights would bias spike rates downward, since the window in which spikes occur is
  systematically unobserved for them.
- The short-haul route (BOS–LGA) receives the same pipeline and feature set as the long-haul
  route but is **not independently tuned** — a deliberate scope decision given the project
  timeline, stated rather than hidden.
- The routes are US domestic rather than the UK-originating routes of the original proposal.
  No free longitudinal fare dataset with the required repeated-search structure exists for
  any UK route; the research questions are unchanged.
- **Flight fares are step functions, not continuous series.** 61.8% of consecutive
  observations of the same flight repeat the previous fare exactly, and a typical flight
  shows only 4 distinct fares across 9 observations — fares sit in quantised booking buckets
  and jump between them. Inside a flat run the rolling standard deviation collapses toward
  zero, so any bucket change clears a 2σ threshold regardless of its economic size. The
  threshold therefore partly measures fare-bucket transitions rather than unusual price
  movements. This is a genuine difference from the commodity-price series the definition was
  adapted from, and it qualifies the spike results rather than invalidating them.

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
