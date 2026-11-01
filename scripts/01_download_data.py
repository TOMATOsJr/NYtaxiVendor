"""
01_download_data.py
-------------------
1. Downloads NYC TLC Yellow Taxi Trip Records (Jan 2024 Parquet, ~50 MB)
   and the official NYC TLC Taxi Zone Lookup CSV if not already present.
2. Inspects schema, null rates, and descriptive statistics.
3. Saves a reproducible 100,000-row raw sample (data/processed/sample_100k.parquet)
   and a machine-readable data dictionary (data/processed/data_dictionary.csv).

Usage:
    uv run python scripts/01_download_data.py
"""

from __future__ import annotations

import pathlib
import shutil
import urllib.request

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

PARQUET_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
ZONES_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"

PARQUET_PATH = RAW_DIR / "yellow_tripdata_2024-01.parquet"
ZONES_PATH = RAW_DIR / "taxi_zone_lookup.csv"
SAMPLE_OUT = PROC_DIR / "sample_100k.parquet"
DICT_OUT = PROC_DIR / "data_dictionary.csv"

SAMPLE_N = 100_000
RANDOM_SEED = 42

COLUMN_DESCRIPTIONS = {
    "VendorID": "TPEP provider: 1=Creative Mobile Technologies (CMT), 2=VeriFone",
    "tpep_pickup_datetime": "Timestamp when the taximeter was engaged",
    "tpep_dropoff_datetime": "Timestamp when the taximeter was disengaged",
    "passenger_count": "Driver-entered passenger count (NULL on ~4.7% of CMT trips)",
    "trip_distance": "Elapsed trip distance in miles reported by taximeter",
    "RatecodeID": "Final rate code: 1=Standard, 2=JFK, 3=Newark, 4=Nassau/Westchester, 5=Negotiated, 6=Group",
    "store_and_fwd_flag": "Y=trip held in vehicle memory before sending to vendor, N=not",
    "PULocationID": "TLC Taxi Zone ID where meter was engaged (pickup)",
    "DOLocationID": "TLC Taxi Zone ID where meter was disengaged (dropoff)",
    "payment_type": "1=Credit card, 2=Cash, 3=No charge, 4=Dispute, 5=Unknown, 6=Voided",
    "fare_amount": "Metered time-and-distance fare in USD",
    "extra": "Rush-hour ($0.50/$1.00) and overnight surcharges",
    "mta_tax": "$0.50 MTA tax automatically triggered based on metered rate",
    "tip_amount": "Credit card tip amount in USD (cash tips are not recorded)",
    "tolls_amount": "Total bridge and tunnel tolls paid during trip",
    "improvement_surcharge": "$0.30 improvement surcharge assessed at flag drop",
    "total_amount": "Total amount charged to rider (excludes cash tips)",
    "congestion_surcharge": "$2.50 NY State congestion surcharge South of 96th St",
    "Airport_fee": "$1.25 fee for pickups at LaGuardia (LGA) and JFK airports",
}


def download_if_missing(url: str, dest: pathlib.Path) -> None:
    if dest.exists() and dest.stat().st_size > 1_000:
        size_mb = dest.stat().st_size / 1e6
        print(f"✓ Found existing {dest.name} ({size_mb:.2f} MB)")
        return

    print(f"⬇ Downloading {dest.name} from {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out)
    size_mb = dest.stat().st_size / 1e6
    print(f"✅ Saved {dest.name} ({size_mb:.2f} MB)")


def print_schema_and_stats(df: pd.DataFrame) -> None:
    print("\n── Schema & Completeness ─────────────────────────────────────────")
    print(f"{'Column':<26} {'Dtype':<18} {'Non-Null':>11} {'Null %':>8}")
    print("─" * 67)
    n = len(df)
    for col in df.columns:
        nulls = int(df[col].isna().sum())
        pct = f"{nulls / n * 100:.1f}%"
        print(f"{col:<26} {str(df[col].dtype):<18} {n - nulls:>11,} {pct:>8}")

    print("\n── Descriptive Summary ───────────────────────────────────────────")
    print(f"  Total records : {len(df):,}")
    print(f"  Pickup window : {df['tpep_pickup_datetime'].min()} → {df['tpep_pickup_datetime'].max()}")
    print(f"  Trip distance : mean={df['trip_distance'].mean():.2f} mi | median={df['trip_distance'].median():.2f} mi")
    print(f"  Total amount  : mean=${df['total_amount'].mean():.2f} | median=${df['total_amount'].median():.2f}")


def write_data_dictionary(df: pd.DataFrame, out_path: pathlib.Path) -> None:
    rows = []
    for col in df.columns:
        rows.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "null_pct": f"{df[col].isna().mean() * 100:.2f}%",
            "sample_values": str(df[col].dropna().unique()[:4].tolist()),
            "description": COLUMN_DESCRIPTIONS.get(col, ""),
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"📖 Data dictionary written → {out_path.relative_to(ROOT)}")


def main() -> None:
    print("=" * 67)
    print("  NYC TLC Yellow Taxi — Data Acquisition & Profiling")
    print("=" * 67)

    download_if_missing(PARQUET_URL, PARQUET_PATH)
    download_if_missing(ZONES_URL, ZONES_PATH)

    df = pd.read_parquet(PARQUET_PATH)
    zones = pd.read_csv(ZONES_PATH)
    print(f"🗺  Loaded {len(zones):,} official NYC TLC taxi zones across {zones['Borough'].nunique()} boroughs.")

    print_schema_and_stats(df)

    sample = df.sample(n=min(SAMPLE_N, len(df)), random_state=RANDOM_SEED)
    sample.to_parquet(SAMPLE_OUT, index=False)
    print(f"\n✅ Saved {len(sample):,}-row sample → {SAMPLE_OUT.relative_to(ROOT)}")

    write_data_dictionary(df, DICT_OUT)


if __name__ == "__main__":
    main()
