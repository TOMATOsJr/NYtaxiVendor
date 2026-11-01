"""
02_spark_etl_and_sql.py
-----------------------
Distributed ETL, Feature Engineering, and Spark SQL Analytics Pipeline
using PySpark (Apache Spark local[*] mode).

What it does:
1. Initializes a local SparkSession (automatically binding jdk4py's Java 17/21
   if the host OS has an incompatible newer JDK such as JDK 25).
2. Ingests the 2.96M-row NYC Yellow Taxi Parquet dataset + Taxi Zone Lookup CSV.
3. Filters corrupted outlier records and engineers analytical & experimental features
   (trip_minutes, speed_mph, distance_bucket, time_bucket, is_airport_trip,
   tip_pct, is_generous_tip).
4. Registers Spark SQL views (`trips`, `zones`) and executes all `.sql` scripts in `sql/`
   (or runs a custom `--file` / `--query` via CLI).
5. Materializes cleaned Parquet aggregates + a 100k stratified sample for downstream
   statistical inference and ML prototyping.
6. Generates Charts 01–03 in `charts/`.

Usage:
    uv run python scripts/02_spark_etl_and_sql.py
    uv run python scripts/02_spark_etl_and_sql.py --file sql/03_window_analytics.sql
    uv run python scripts/02_spark_etl_and_sql.py --query "SELECT vendor_id, COUNT(*) FROM trips GROUP BY vendor_id"
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import time
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW_PARQUET = ROOT / "data" / "raw" / "yellow_tripdata_2024-01.parquet"
ZONES_CSV = ROOT / "data" / "raw" / "taxi_zone_lookup.csv"
AGG_DIR = ROOT / "data" / "processed" / "spark_aggregates"
CLEAN_SAMPLE_PATH = ROOT / "data" / "processed" / "sample_100k_clean.parquet"
SQL_DIR = ROOT / "sql"
CHART_DIR = ROOT / "charts"

AGG_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

# ── Visual Theme (Dark Portfolio Style) ──────────────────────────────────────
BG = "#0f1117"
PANEL = "#1a1d27"
BORDER = "#3a3d4d"
TEXT = "#e0e3f0"
SUBTEXT = "#8b8fa8"
PAL = ["#7c6af7", "#f7706a", "#4ecdc4", "#ffe66d", "#a8e6cf", "#ff8b94"]

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": PANEL,
    "axes.edgecolor": BORDER,
    "axes.labelcolor": SUBTEXT,
    "xtick.color": SUBTEXT,
    "ytick.color": SUBTEXT,
    "text.color": TEXT,
    "grid.color": BORDER,
    "grid.linestyle": "--",
    "grid.alpha": 0.55,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.titlesize": 14,
})


def configure_java_home() -> None:
    """Use the full system OpenJDK if available; fall back to jdk4py only if no system java exists."""
    import shutil
    if shutil.which("java"):
        return
    try:
        import jdk4py  # type: ignore
        java_home = str(jdk4py.JAVA_HOME)
        if os.path.isdir(java_home):
            os.environ["JAVA_HOME"] = java_home
            os.environ["PATH"] = f"{os.path.join(java_home, 'bin')}:{os.environ.get('PATH', '')}"
    except Exception:
        pass


def create_spark_session():
    configure_java_home()
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder
        .appName("NYCTaxi-SparkSQL-AB-Analytics")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def build_enriched_views(spark):
    """Ingest raw Parquet & Zone CSV, clean outliers, engineer features, and register SQL views."""
    from pyspark.sql import functions as F

    t0 = time.perf_counter()
    raw_df = spark.read.parquet(str(RAW_PARQUET))
    zones_df = (
        spark.read.option("header", "true")
        .option("inferSchema", "true")
        .csv(str(ZONES_CSV))
    )
    raw_count = raw_df.count()

    # Standardize column names & compute derived temporal/kinematic/experiment features
    df = (
        raw_df
        .withColumnRenamed("VendorID", "vendor_id")
        .withColumnRenamed("tpep_pickup_datetime", "pickup_datetime")
        .withColumnRenamed("tpep_dropoff_datetime", "dropoff_datetime")
        .withColumnRenamed("RatecodeID", "ratecode_id")
        .withColumnRenamed("PULocationID", "pickup_location_id")
        .withColumnRenamed("DOLocationID", "dropoff_location_id")
        .withColumnRenamed("Airport_fee", "airport_fee")
        .withColumn("pickup_date", F.to_date("pickup_datetime").cast("string"))
        .withColumn("pickup_hour", F.hour("pickup_datetime"))
        .withColumn("pickup_dow", F.date_format("pickup_datetime", "EEEE"))
        .withColumn("dow_num", F.dayofweek("pickup_datetime"))  # 1=Sun ... 7=Sat
        .withColumn("is_weekend", F.when(F.col("dow_num").isin(1, 7), 1).otherwise(0))
        .withColumn(
            "trip_minutes",
            F.round(
                (F.unix_timestamp("dropoff_datetime") - F.unix_timestamp("pickup_datetime")) / 60.0,
                2,
            ),
        )
    )

    # Filter data quality anomalies (strictly Jan 2024, positive distance/duration/fare)
    clean_df = df.filter(
        (F.col("pickup_date").between("2024-01-01", "2024-01-31"))
        & (F.col("trip_minutes").between(1.0, 180.0))
        & (F.col("trip_distance").between(0.1, 100.0))
        & (F.col("total_amount").between(3.0, 300.0))
        & (F.col("fare_amount") >= 2.5)
    )

    # Feature engineering for SQL analytics, A/B stratification, and ML prototype
    enriched_df = (
        clean_df
        .withColumn(
            "speed_mph",
            F.round(F.col("trip_distance") / (F.col("trip_minutes") / 60.0), 2),
        )
        .withColumn(
            "distance_bucket",
            F.when(F.col("trip_distance") < 1.0, "< 1 mi")
            .when(F.col("trip_distance") < 3.0, "1-3 mi")
            .when(F.col("trip_distance") < 5.0, "3-5 mi")
            .when(F.col("trip_distance") < 10.0, "5-10 mi")
            .otherwise("10+ mi"),
        )
        .withColumn(
            "time_bucket",
            F.when(F.col("pickup_hour").between(6, 9), "AM Rush")
            .when(F.col("pickup_hour").between(10, 15), "Midday")
            .when(F.col("pickup_hour").between(16, 19), "PM Rush")
            .when(F.col("pickup_hour").between(20, 23), "Night")
            .otherwise("Overnight"),
        )
        .withColumn(
            "is_airport_trip",
            F.when(
                (F.coalesce(F.col("ratecode_id"), F.lit(1)).isin(2, 3))
                | (F.coalesce(F.col("airport_fee"), F.lit(0.0)) > 0)
                | (F.col("pickup_location_id").isin(1, 132, 138))
                | (F.col("dropoff_location_id").isin(1, 132, 138)),
                1,
            ).otherwise(0),
        )
        .withColumn(
            "tip_pct",
            F.round((F.col("tip_amount") / F.col("fare_amount")) * 100.0, 2),
        )
        .withColumn(
            "is_generous_tip",
            F.when((F.col("payment_type") == 1) & (F.col("tip_pct") >= 20.0), 1).otherwise(0),
        )
        .cache()
    )

    clean_count = enriched_df.count()
    elapsed = time.perf_counter() - t0
    print(f"⚡ PySpark ETL complete in {elapsed:.1f}s")
    print(f"   Raw rows     : {raw_count:,}")
    print(f"   Cleaned rows : {clean_count:,} ({raw_count - clean_count:,} outlier/corrupt rows removed)")

    enriched_df.createOrReplaceTempView("trips")
    zones_df.createOrReplaceTempView("zones")
    return enriched_df, zones_df


def split_sql_file(path: pathlib.Path) -> list[tuple[str, str]]:
    text = path.read_text()
    raw_blocks = re.split(r"\n(?=-- ── )", text)
    queries = []
    for block in raw_blocks:
        lines = block.strip().splitlines()
        label_lines = [l for l in lines if l.startswith("--")]
        sql_lines = [l for l in lines if not l.startswith("--") and l.strip()]
        sql = "\n".join(sql_lines).strip().rstrip(";")
        if not sql:
            continue
        label = next(
            (l.lstrip("- ─").strip() for l in label_lines if "Q" in l and ":" in l),
            f"Query from {path.name}",
        )
        queries.append((label, sql))
    return queries


def execute_sql_suite(spark, target_file: pathlib.Path | None = None, limit: int = 15) -> None:
    files = [target_file] if target_file else sorted(SQL_DIR.glob("*.sql"))
    for path in files:
        print(f"\n{'═' * 72}")
        print(f"  SPARK SQL FILE: {path.name}")
        print(f"{'═' * 72}")
        for label, sql in split_sql_file(path):
            print(f"\n── {label} " + "─" * max(1, 66 - len(label)))
            t0 = time.perf_counter()
            pdf = spark.sql(sql).limit(limit).toPandas()
            ms = (time.perf_counter() - t0) * 1000
            print(pdf.to_string(index=False))
            print(f"   [Spark SQL executed in {ms:.0f} ms]")


def materialize_datasets_and_charts(spark) -> None:
    print(f"\n{'═' * 72}")
    print("  MATERIALIZING SPARK AGGREGATES & GENERATING CHARTS 01–03")
    print(f"{'═' * 72}")

    # 1. Daily + Window Metrics (Chart 01)
    daily_win_sql = split_sql_file(SQL_DIR / "03_window_analytics.sql")[0][1]
    daily_df = spark.sql(daily_win_sql).toPandas()
    daily_df.to_csv(AGG_DIR / "daily_window_metrics.csv", index=False)
    daily_df["pickup_date"] = pd.to_datetime(daily_df["pickup_date"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True, gridspec_kw={"hspace": 0.10})
    fig.patch.set_facecolor(BG)

    ax1.fill_between(daily_df["pickup_date"], daily_df["daily_revenue"] / 1e6, alpha=0.18, color=PAL[0])
    ax1.plot(daily_df["pickup_date"], daily_df["daily_revenue"] / 1e6, color=PAL[0], lw=2.0, marker="o", ms=4, label="Daily Revenue")
    ax1.plot(daily_df["pickup_date"], daily_df["rolling_7d_revenue"] / 1e6, color=PAL[3], lw=2.6, ls="-", label="7-Day Rolling Avg (Spark Window)")
    ax1.set_ylabel("Revenue ($M)")
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1fM"))
    ax1.set_title("NYC Yellow Taxi — Daily Revenue, 7-Day Rolling Trend & Cumulative Growth (Spark SQL)", fontweight="bold", color=TEXT, pad=12)
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True)

    ax2.bar(daily_df["pickup_date"], daily_df["daily_trips"] / 1e3, color=PAL[2], alpha=0.45, width=0.75, label="Daily Trips (K)")
    ax2.plot(daily_df["pickup_date"], daily_df["rolling_7d_trips"] / 1e3, color=PAL[1], lw=2.5, label="7-Day Rolling Trips (K)")
    ax2.set_ylabel("Trips (Thousands)")
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0fK"))
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(True)

    for ax in (ax1, ax2):
        for _, r in daily_df.iterrows():
            if r["pickup_date"].weekday() >= 5:
                ax.axvspan(r["pickup_date"] - pd.Timedelta("0.5d"), r["pickup_date"] + pd.Timedelta("0.5d"), alpha=0.07, color="#ffffff")

    out1 = CHART_DIR / "01_daily_revenue_rolling.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"   💾 Saved {out1.relative_to(ROOT)}")

    # 2. Demand & Traffic Speed Heatmap (Chart 02)
    hm_sql = split_sql_file(SQL_DIR / "02_demand_and_velocity.sql")[0][1]
    hm_df = spark.sql(hm_sql).toPandas()
    hm_df.to_csv(AGG_DIR / "hourly_dow_matrix.csv", index=False)

    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    piv_trips = hm_df.pivot(index="pickup_hour", columns="pickup_dow", values="trips").reindex(columns=dow_order)
    piv_speed = hm_df.pivot(index="pickup_hour", columns="pickup_dow", values="avg_speed_mph").reindex(columns=dow_order)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.patch.set_facecolor(BG)

    sns.heatmap(piv_trips / 1e3, ax=ax1, cmap="magma", linewidths=0.3, linecolor=BG, cbar_kws={"label": "Trips (Thousands)"})
    ax1.set_title("Demand Intensity (Trips in K) — Hour × Day", fontweight="bold", color=TEXT)
    ax1.set_xlabel("Day of Week")
    ax1.set_ylabel("Hour of Day (0–23)")

    sns.heatmap(piv_speed, ax=ax2, cmap="viridis", linewidths=0.3, linecolor=BG, cbar_kws={"label": "Mean Traffic Speed (mph)"})
    ax2.set_title("Traffic Velocity (mph) — Midday Manhattan Gridlock", fontweight="bold", color=TEXT)
    ax2.set_xlabel("Day of Week")
    ax2.set_ylabel("Hour of Day (0–23)")

    plt.tight_layout()
    out2 = CHART_DIR / "02_demand_speed_heatmap.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"   💾 Saved {out2.relative_to(ROOT)}")

    # 3. Top Pickup Zones & Origin-Destination Corridors (Chart 03)
    zones_sql = split_sql_file(SQL_DIR / "01_revenue_and_unit_economics.sql")[3][1]
    corridors_sql = split_sql_file(SQL_DIR / "02_demand_and_velocity.sql")[3][1]
    top_zones_df = spark.sql(zones_sql).toPandas()
    corridors_df = spark.sql(corridors_sql).toPandas()
    top_zones_df.to_csv(AGG_DIR / "top_zones.csv", index=False)
    corridors_df.to_csv(AGG_DIR / "top_corridors.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor(BG)

    top_z = top_zones_df.iloc[::-1]
    norm = plt.Normalize(top_z["avg_fare"].min(), top_z["avg_fare"].max())
    colors = plt.cm.plasma(norm(top_z["avg_fare"]))
    bars = ax1.barh(top_z["zone_name"], top_z["trips"] / 1e3, color=colors, edgecolor=BG, height=0.68)
    ax1.set_xlabel("Trips (Thousands)")
    ax1.set_title("Top 15 Pickup Zones (Colored by Mean Fare $)", fontweight="bold", color=TEXT)
    ax1.grid(True, axis="x")
    for b, fare in zip(bars, top_z["avg_fare"]):
        ax1.text(b.get_width() + 1.5, b.get_y() + b.get_height() / 2, f"${fare:.0f}", va="center", fontsize=8, color=TEXT)

    corr = corridors_df.head(10).iloc[::-1].copy()
    corr["pair"] = corr["pickup_zone"].str.slice(0, 18) + " → " + corr["dropoff_zone"].str.slice(0, 18)
    ax2.barh(corr["pair"], corr["trips"] / 1e3, color=PAL[2], edgecolor=BG, height=0.65)
    ax2.set_xlabel("Trips (Thousands)")
    ax2.set_title("Top 10 Inter-Zone Corridors (Spark SQL Join)", fontweight="bold", color=TEXT)
    ax2.grid(True, axis="x")

    plt.tight_layout()
    out3 = CHART_DIR / "03_borough_corridors.png"
    fig.savefig(out3, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"   💾 Saved {out3.relative_to(ROOT)}")

    # 4. Save A/B Stratification & Subgroup Aggregates + 100k Enriched Sample
    ab_strata_sql = split_sql_file(SQL_DIR / "04_ab_experiment_cohorts.sql")[2][1]
    ab_sub_sql = split_sql_file(SQL_DIR / "04_ab_experiment_cohorts.sql")[3][1]
    spark.sql(ab_strata_sql).toPandas().to_csv(AGG_DIR / "ab_strata_summary.csv", index=False)
    spark.sql(ab_sub_sql).toPandas().to_csv(AGG_DIR / "ab_subgroup_summary.csv", index=False)

    sample_sql = """
        SELECT
            vendor_id,
            pickup_date,
            pickup_hour,
            pickup_dow,
            is_weekend,
            time_bucket,
            distance_bucket,
            is_airport_trip,
            payment_type,
            COALESCE(passenger_count, 1.0) AS passenger_count,
            trip_distance,
            trip_minutes,
            speed_mph,
            fare_amount,
            tip_amount,
            tolls_amount,
            COALESCE(congestion_surcharge, 0.0) AS congestion_surcharge,
            COALESCE(airport_fee, 0.0) AS airport_fee,
            total_amount,
            tip_pct,
            is_generous_tip,
            COALESCE(z.Borough, 'Unknown') AS pickup_borough,
            COALESCE(z.Zone, 'Unknown') AS pickup_zone
        FROM trips t
        LEFT JOIN zones z ON t.pickup_location_id = z.LocationID
        WHERE vendor_id IN (1, 2)
          AND speed_mph BETWEEN 1.0 AND 65.0
    """
    clean_sample_pdf = (
        spark.sql(sample_sql)
        .sample(withReplacement=False, fraction=0.045, seed=42)
        .limit(100_000)
        .toPandas()
    )
    clean_sample_pdf.to_parquet(CLEAN_SAMPLE_PATH, index=False)
    print(f"   📦 Saved enriched 100k analytical sample → {CLEAN_SAMPLE_PATH.relative_to(ROOT)} ({len(clean_sample_pdf):,} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description="PySpark ETL & Spark SQL Analytics Runner")
    parser.add_argument("--file", "-f", type=pathlib.Path, help="Run a specific .sql file")
    parser.add_argument("--query", "-q", type=str, help="Run a single inline Spark SQL query")
    parser.add_argument("--limit", "-l", type=int, default=15, help="Max rows to display per query")
    args = parser.parse_args()

    spark = create_spark_session()
    try:
        build_enriched_views(spark)
        if args.query:
            print(f"\n── Inline Spark SQL Query ────────────────────────────────────────")
            print(spark.sql(args.query).limit(args.limit).toPandas().to_string(index=False))
        elif args.file:
            execute_sql_suite(spark, target_file=args.file, limit=args.limit)
        else:
            execute_sql_suite(spark, limit=args.limit)
            materialize_datasets_and_charts(spark)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
