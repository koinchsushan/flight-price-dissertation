# Data directory

The data files themselves are **not committed** — the working subset is ~662 MB and the full
source release is ~31 GB, both well over GitHub's 100 MB per-file limit. This file documents
how to obtain and reproduce them.

## Expected contents

```
data/
├── jfk_lax_bos_lga.csv    the working subset (2,156,316 rows, ~662 MB)  [required]
└── processed/             derived outputs written by the notebooks       [generated]
```

## Source

[`dilwong/flightprices`](https://www.kaggle.com/datasets/dilwong/flightprices) — one-way US
domestic itineraries scraped from Expedia between **16 April and 5 October 2022**.
The full file, `itineraries.csv`, is 82,138,753 rows across 27 columns.

## Getting the data

You need `jfk_lax_bos_lga.csv` in this folder. There are two ways to get it.

### Option A — use a copy of the subset (recommended)

If you have been given the ~662 MB `jfk_lax_bos_lga.csv` directly, just place it in this
folder. Nothing else is required. This is by far the quicker route: it skips a 31 GB download
entirely.

### Option B — rebuild it from the full Kaggle release

Only necessary if you are reproducing the extraction from scratch.

**Before you start, be aware of what this involves:**

- A **free Kaggle account**, and accepting the dataset's terms on its page.
- A **~31 GB download**, plus room for the output — budget **40 GB of free disk space**.
- **Tens of minutes to a few hours**, depending on your connection and disk speed.

**Steps**

1. Download `itineraries.csv` from
   [kaggle.com/datasets/dilwong/flightprices](https://www.kaggle.com/datasets/dilwong/flightprices)
   and unzip it. You can use the website's Download button — no API setup needed.

2. Install `polars`, which streams the file rather than loading it into memory. It is not in
   `requirements.txt` because it is only needed for this one step:

   ```bash
   pip install polars
   ```

3. Run the extraction script, pointing it at wherever you saved the file:

   ```bash
   python scripts/build_subset.py --input /path/to/itineraries.csv
   ```

   On Windows the path looks like `C:\Users\you\Downloads\itineraries.csv`.

   The script filters to the four directional airport pairs `JFK→LAX`, `LAX→JFK`, `BOS→LGA`
   and `LGA→BOS`, writes the result to `data/jfk_lax_bos_lga.csv`, and reports the row count.
   It streams throughout, so memory use stays low regardless of the input size.

4. Confirm it reports **2,156,316 rows**. If it does not, the script says so — most likely
   the source file differs from the published release.

> **Do not try this with a plain `pandas.read_csv`.** The full file exhausts memory even with
> `usecols` column trimming — this was confirmed on Kaggle's own hosted notebooks, not
> assumed.

Once the file is in place, run `python scripts/verify_setup.py` from the repository root to
confirm everything is ready.

## Expected row counts

Verified against the extracted subset:

| Route | Rows (raw) | Rows (coach-only) |
|---|---:|---:|
| LAX → JFK | 625,496 | 624,358 |
| JFK → LAX | 605,017 | 603,909 |
| BOS → LGA | 483,784 | 483,087 |
| LGA → BOS | 442,019 | 441,274 |
| **Total** | **2,156,316** | **2,152,628** |

## Data quality notes

Verified on the 2,156,316-row subset:

- `baseFare`, `totalFare`, `searchDate`, `flightDate`, `startingAirport`,
  `destinationAirport` — **zero nulls**, no imputation required.
- `totalTravelDistance` — 12,865 nulls (0.6%), evenly spread across all four routes
  (0.49%–0.69%), so no route-specific handling is needed.
- `segmentsEquipmentDescription` — 130,256 nulls (6.0%); aircraft type, dropped as not
  relevant to fare modelling.
- `segmentsDistance` — 5,968 nulls (0.3%).
