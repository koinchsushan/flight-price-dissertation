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

## Reproducing the subset

The subset is the full file filtered to four directional airport pairs: `JFK→LAX`, `LAX→JFK`,
`BOS→LGA`, `LGA→BOS`.

**Do not attempt this with a plain `pandas.read_csv`** — the full file exhausts memory even
with `usecols` column trimming. Stream it instead:

```python
import polars as pl

ROUTES = [("JFK", "LAX"), ("LAX", "JFK"), ("BOS", "LGA"), ("LGA", "BOS")]

(
    pl.scan_csv("itineraries.csv")
    .filter(
        pl.concat_str([pl.col("startingAirport"), pl.col("destinationAirport")])
        .is_in([a + b for a, b in ROUTES])
    )
    .collect(engine="streaming")
    .write_csv("jfk_lax_bos_lga.csv")
)
```

> `polars` 1.25.0 renamed the streaming parameter from `streaming=True` to
> `engine="streaming"`. The older form still runs but emits a deprecation warning.

A chunked `csv.reader` loop achieves the same result more slowly and without the extra
dependency.

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
