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

**Validation.** Rolling-origin (walk-forward), five folds of 14 days, on the primary route.
García Crespi et al. (2026) show model rankings can *reverse* between a single chronological
split and rolling-origin validation on a structurally similar comparison, so a single split
cannot support a "which model wins" claim. The secondary route uses a single chronological
split, consistent with its lighter validation pass.

Splits are taken on **departure date, not search date**. A single itinerary is searched
repeatedly over weeks, so cutting on `searchDate` leaves 61.8% of test flights also present
in training, and the model would be scored partly on trajectories it had already memorised.
Cutting on departure date keeps each trajectory wholly on one side; `assert_no_group_leakage`
verifies this and is run before any model is fitted.

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
├── scripts/                verify_setup.py (environment check), build_subset.py
├── reports/figures/        Generated figures (300 dpi PNG)
└── data/                   Not committed; see data/README.md
```

Logic lives in `src/`; notebooks import it and carry the narrative. This keeps the analysis
readable end-to-end while avoiding copy-pasted code between phases.

## Setup

Works on macOS, Windows and Linux. The only prerequisite is **Python 3.12 or newer** —
check with `python --version` (or `python3 --version` on macOS/Linux), and install it from
[python.org](https://www.python.org/downloads/) if needed. On Windows, tick
*"Add Python to PATH"* in the installer.

Each step below gives the macOS/Linux command first, then the Windows equivalent. Run them
from the repository root.

### 1. Create and activate a virtual environment

This keeps the project's packages separate from the rest of your system.

**macOS / Linux**
```bash
python3 -m venv .venv && source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
py -m venv .venv; .venv\Scripts\Activate.ps1
```

Your prompt should now start with `(.venv)`. Re-run the activate command in any new terminal.

> If PowerShell blocks the script with an execution-policy error, run
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and try again. This applies
> to the current window only.

### 2. Install the OpenMP runtime — macOS only

XGBoost does not bundle this on macOS and will fail to import without it. Skip this step on
Windows and Linux, where the wheels normally work as-is — if XGBoost does fail to import
there, see Troubleshooting below.

```bash
brew install libomp
```

If you do not have Homebrew, install it from [brew.sh](https://brew.sh).

### 3. Install the dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

The second command installs this project's own code so the notebooks can import
`flightprice` without any path juggling. Expect a few minutes — PyTorch is a large download.

### 4. Register the Jupyter kernel

The notebooks are saved against a kernel named `flightprice`. Without this step they will
open with no kernel attached and will not run.

```bash
python -m ipykernel install --user --name flightprice --display-name "Python (flightprice)"
```

### 5. Get the data

The dataset is not in this repository — it is far too large for GitHub. Follow
[`data/README.md`](data/README.md), which covers both downloading the ready-made subset and
rebuilding it from the full Kaggle release.

### 6. Check everything worked

```bash
python scripts/verify_setup.py
```

This checks the Python version, every required package, the compute device, the project
import, the data files and the Jupyter kernel. Each failure prints the exact command that
fixes it. When it reports all checks passed, you are ready.

### 7. Run the notebooks

Open the `notebooks/` folder in VS Code or JupyterLab and run them **in numerical order** —
each writes files the next one reads. Select the **Python (flightprice)** kernel when
prompted.

```bash
jupyter lab            # or: code .
```

To run one non-interactively instead:

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=flightprice notebooks/01_data_cleaning.ipynb
```

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `XGBoostError: ... libomp.dylib ... not loaded` | macOS is missing the OpenMP runtime. Run `brew install libomp`. |
| `XGBoostError: ... vcomp140.dll ... not loaded` | Windows is missing the Visual C++ runtime. Install the [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist) and restart the terminal. |
| `ModuleNotFoundError: No module named 'flightprice'` | The project was not installed. Run `pip install -e .` from the repository root with the venv active. |
| Notebook shows "no kernel" or "kernel not found" | Step 4 was skipped. Register the kernel, then reopen the notebook. |
| `ModuleNotFoundError: No module named 'polars'` | Only needed to rebuild the subset. Run `pip install polars`. |
| `FileNotFoundError: ... jfk_lax_bos_lga.csv` | The dataset is missing. See [`data/README.md`](data/README.md). |
| PowerShell: "running scripts is disabled on this system" | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again. |
| `jupyter nbconvert` runs the wrong Python (pyenv users) | The pyenv shim intercepts the subcommand. Call the venv binary directly: `.venv/bin/jupyter-nbconvert`. |
| Notebook 02 is slow | Expected. It computes rolling statistics over ~1.9 M rows for seven window sizes; several minutes is normal. |

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
