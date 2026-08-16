"""Rebuild the four-route working subset from the full Kaggle release.

The full ``itineraries.csv`` is ~31 GB and will exhaust memory under a plain
``pandas.read_csv``, so this streams it with polars and never holds the whole
file in memory.

    pip install polars
    python scripts/build_subset.py --input /path/to/itineraries.csv

You only need this if you are reproducing the subset from scratch. Most people
should download the pre-built subset instead - see data/README.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from flightprice.config import RAW_SUBSET, ROUTES
except ImportError:  # allow running before `pip install -e .`
    RAW_SUBSET = Path(__file__).resolve().parents[1] / "data" / "jfk_lax_bos_lga.csv"
    ROUTES = ("JFK-LAX", "LAX-JFK", "BOS-LGA", "LGA-BOS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Path to the full itineraries.csv from Kaggle",
    )
    parser.add_argument(
        "--output", type=Path, default=RAW_SUBSET,
        help=f"Where to write the subset (default: {RAW_SUBSET})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import polars as pl
    except ImportError:
        print("polars is required for this script but is not installed.", file=sys.stderr)
        print("  pip install polars", file=sys.stderr)
        return 1

    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        return 1

    # "JFK-LAX" -> "JFKLAX", matching startingAirport + destinationAirport concatenated
    pairs = [route.replace("-", "") for route in ROUTES]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    size_gb = args.input.stat().st_size / 1e9
    print(f"Reading  : {args.input}  ({size_gb:.1f} GB)")
    print(f"Keeping  : {', '.join(ROUTES)}")
    print(f"Writing  : {args.output}")
    print("\nStreaming - this takes a while on the full file and uses little memory.")

    query = pl.scan_csv(args.input, infer_schema_length=10_000).filter(
        pl.concat_str(
            [pl.col("startingAirport"), pl.col("destinationAirport")]
        ).is_in(pairs)
    )

    try:
        query.sink_csv(args.output)
    except Exception:
        # Older polars, or a plan that cannot be streamed to disk: fall back to
        # collecting in streaming mode, then writing.
        try:
            frame = query.collect(engine="streaming")
        except TypeError:
            # polars < 1.25.0 spelled this differently
            frame = query.collect(streaming=True)
        frame.write_csv(args.output)

    rows = pl.scan_csv(args.output).select(pl.len()).collect().item()
    out_mb = args.output.stat().st_size / 1e6
    print(f"\nDone: {rows:,} rows, {out_mb:,.0f} MB")

    if rows != 2_156_316:
        print(
            f"NOTE: expected 2,156,316 rows for the documented subset but got {rows:,}. "
            "If your source file differs from the published release, that is why.",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
