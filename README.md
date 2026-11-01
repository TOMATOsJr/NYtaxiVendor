# 🚕 NYC Yellow Taxi: PySpark Analytics, Statistical Inference & ML Prototype

An end-to-end data science portfolio project built on **2,964,624 real NYC Yellow Taxi trips** (January 2024) and the official **NYC TLC Taxi Zone Lookup** (265 zones across 6 boroughs).

Designed to demonstrate:
- **Distributed Data Processing & SQL (`PySpark` & `Spark SQL`):** Parquet ingestion, data cleaning, feature engineering, multi-table joins, and analytical window functions (`LAG`, `RANK`, `NTILE`, `ROW_NUMBER`, and 7-day rolling windows).
- **Descriptive & Inferential Statistics + Quasi-Experimental A/B Analysis (`SciPy` & `Statsmodels`):** $\chi^2$ test of independence, two-proportion Z-test, Welch's two-sample t-test, Mann-Whitney U test, Cohen's $d$, non-parametric bootstrap confidence intervals, route-mix stratification, covariate-adjusted OLS (HC3 robust SE), and Holm-Bonferroni / Benjamini-Hochberg multiple testing corrections.
- **Modest Tabular ML Prototype (`Scikit-learn`):** Pre-payment feature pipeline (`ColumnTransformer` with `StandardScaler` + `OneHotEncoder`) comparing **Logistic Regression** and **HistGradientBoostingClassifier** for generous tip propensity ($\text{tip} \ge 20\%$), evaluated with ROC-AUC, PR-AUC, Brier calibration curves, and permutation feature importance.

---

## Quick Start

```bash
cd /home/tomato/Desktop/NYCTaxiAnalysis/mine

# Activate the virtual environment (Python 3.12)
source .venv/bin/activate

# Run the entire 4-stage pipeline end-to-end
python scripts/run_pipeline.py
```

Or run individual stages directly using `.venv/bin/python`:

```bash
# 1. Download raw TLC Parquet (~50 MB) + Taxi Zone CSV & build data dictionary
.venv/bin/python scripts/01_download_data.py

# 2. Run PySpark ETL, execute all 4 Spark SQL files, and save Charts 01–03
.venv/bin/python scripts/02_spark_etl_and_sql.py

#    (Optional) Run a single Spark SQL file or custom inline query:
.venv/bin/python scripts/02_spark_etl_and_sql.py --file sql/03_window_analytics.sql
.venv/bin/python scripts/02_spark_etl_and_sql.py --query "SELECT vendor_id, COUNT(*) AS trips FROM trips GROUP BY vendor_id"

# 3. Run A/B & quasi-experimental statistical inference suite → Charts 04–06
.venv/bin/python scripts/03_ab_statistical_inference.py

# 4. Train & evaluate Scikit-learn tip propensity prototype → Chart 07
.venv/bin/python scripts/04_ml_tip_prototype.py
```

---

## Project Structure

```text
mine/
├── data/
│   ├── raw/
│   │   ├── yellow_tripdata_2024-01.parquet     # 2,964,624 raw trip records (49.96 MB)
│   │   └── taxi_zone_lookup.csv                # 265 official NYC TLC zones & boroughs
│   └── processed/
│       ├── data_dictionary.csv                 # Automated schema & null-rate dictionary
│       ├── sample_100k.parquet                 # Reproducible 100k raw sample
│       ├── sample_100k_clean.parquet           # Spark-cleaned & enriched 100k sample
│       ├── ab_test_results.csv                 # Statistical test estimates, CIs & p-values
│       ├── ml_metrics.csv                      # Held-out test metrics for ML models
│       └── spark_aggregates/                   # Materialized Spark SQL analytical tables
│
├── sql/                                        # 15 Spark SQL queries across 4 files
│   ├── 01_revenue_and_unit_economics.sql       # Daily revenue, fee split, borough joins
│   ├── 02_demand_and_velocity.sql              # Hour × DOW matrix, traffic speed, corridors
│   ├── 03_window_analytics.sql                 # SUM/AVG OVER, LAG, RANK, ROW_NUMBER, NTILE
│   └── 04_ab_experiment_cohorts.sql            # Vendor A/B cohorts & route stratification
│
├── scripts/
│   ├── 01_download_data.py                     # Downloader, schema profiler & dictionary builder
│   ├── 02_spark_etl_and_sql.py                 # PySpark ETL, Spark SQL runner & Charts 01–03
│   ├── 03_ab_statistical_inference.py          # Statistical inference suite & Charts 04–06
│   ├── 04_ml_tip_prototype.py                  # Scikit-learn prototype & Chart 07
│   └── run_pipeline.py                         # One-command orchestrator
│
├── charts/                                     # 7 dark-themed PNG visualizations (150 DPI)
│   ├── 01_daily_revenue_rolling.png
│   ├── 02_demand_speed_heatmap.png
│   ├── 03_borough_corridors.png
│   ├── 04_ab_vendor_distributions_ci.png
│   ├── 05_ab_confounder_stratification.png
│   ├── 06_ab_subgroups_multiple_testing.png
│   └── 07_ml_prototype_evaluation.png
│
├── docs/
│   └── methodology.md                          # Statistical & modeling methodology notes
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Key Empirical Results

### 1. PySpark ETL & Unit Economics (`scripts/02_spark_etl_and_sql.py`)
- **Dataset Scale:** Ingested **2,964,624** raw trips; after removing out-of-bounds dates, negative fares, and extreme duration/distance anomalies, retained **2,855,498** clean January 2024 trips totaling **$77.87M** in gross revenue.
- **Revenue Decomposition:** Base metered fares account for **67.5% ($52.60M)** of revenue, credit-card tips **12.5% ($9.71M)**, congestion surcharges **$6.38M**, tolls **$1.54M**, and airport fees **$0.40M**.
- **Borough & Zone Concentration:**
  - **Manhattan** originates **89.76%** of pickups (2.56M trips, mean fare **$22.74**, mean distance **2.31 mi**).
  - **Queens** originates **8.92%** of pickups (254.7K trips, dominated by **JFK Airport** at **$80.60** avg fare and **LaGuardia Airport** at **$66.23** avg fare).
  - Top inter-zone corridor: **Upper East Side South ↔ Upper East Side North** (40,784 combined trips).
- **Traffic Congestion Signal:** Midday weekday speeds in Manhattan drop to **10.21 mph** (median **8.57 mph** at 2 PM), compared to **19.70 mph** at 5 AM.

---

### 2. Statistical Inference & Quasi-Experimental Vendor A/B Analysis (`scripts/03_ab_statistical_inference.py`)

Compares **Group A — Creative Mobile Technologies (Vendor 1, 24.31% share)** vs. **Group B — VeriFone (Vendor 2, 75.69% share)**:

| Test / Estimator | Metric | CMT (V1) | VeriFone (V2) | Difference (V2 − V1) | 95% Confidence Interval | Test Statistic | p-value |
|---|---|---|---|---|---|---|---|
| **Pearson $\chi^2$ Independence** | Payment Type Mix | — | — | Dependent | — | $\chi^2 = 343.42$ ($\text{dof}=3$) | $3.97 \times 10^{-74}$ |
| **Two-Proportion Z-Test** | Generous Tip Rate ($\ge 20\%$) | 74.27% | 77.33% | **+3.06 pp** | $[+2.38, +3.75]\text{ pp}$ | $z = 8.933$ | $4.15 \times 10^{-19}$ |
| **Welch Two-Sample t-Test** | Unadjusted Mean Fare ($) | $25.80 | $27.94 | **+$2.13** | $[+\$1.70, +\$2.54]$ (boot) | $t = 14.463$ ($d=0.099$) | $2.61 \times 10^{-47}$ |
| **Mann-Whitney U + Bootstrap** | Median Fare ($) | $19.65 | $20.16 | **+$0.51** | $[+\$0.36, +\$0.82]$ (boot) | $U = 9.56 \times 10^8$ | $1.43 \times 10^{-25}$ |
| **Route-Stratified ATE** | Airport × Distance Strata | $25.80 | $26.56 | **+$0.76** | $[+\$0.64, +\$0.88]$ | $z = 12.069$ | $< 10^{-30}$ |
| **Covariate-Adjusted OLS (HC3)** | Controlled for Distance, Airport & Time | — | — | **+$0.60** | $[+\$0.51, +\$0.70]$ | $t = 12.956$ | $2.50 \times 10^{-38}$ |

**Key Experimentation Insight:**
- A naive Welch t-test shows VeriFone fares are **+$2.13 higher** on average ($p = 2.61 \times 10^{-47}$).
- However, checking baseline route covariates reveals strong **observational confounding**: VeriFone has a **44% higher airport trip share** (11.62% vs. 8.07% in the sample; 11.07% vs. 7.90% in the full population) and longer average trip distances (3.39 mi vs. 3.09 mi).
- Stratifying on `(is_airport_trip × distance_bucket)` shrinks the estimated vendor gap by **64%** (to **+$0.76**), and covariate-adjusted OLS with HC3 robust standard errors reduces it by **72%** (to **+$0.60**, 95% CI $[+\$0.51, +\$0.70]$).
- Across all 5 time-of-day subgroups (`AM Rush`, `Midday`, `PM Rush`, `Night`, `Overnight`), differences remain statistically significant after **Holm-Bonferroni** family-wise error rate correction ($\alpha = 0.05$).

---

### 3. Tabular ML Prototype: Generous Tip Propensity (`scripts/04_ml_tip_prototype.py`)

Predicts whether a credit-card trip yields a **generous tip ($\text{tip\_pct} \ge 20\%$)** using strictly pre-payment features (excluding `tip_amount`, `tip_pct`, and `total_amount` to prevent target leakage):

| Model Pipeline | ROC-AUC | PR-AUC | Brier Score (↓ better) | Accuracy | F1 Score |
|---|---|---|---|---|---|
| **Logistic Regression** (`StandardScaler` + `OneHotEncoder`) | 0.6071 | 0.8183 | 0.1711 | 0.7748 | 0.8712 |
| **HistGradientBoostingClassifier** (Prototype) | **0.6123** | **0.8231** | **0.1694** | **0.7772** | **0.8728** |

- **Top Predictive Features (Permutation $\Delta\text{AUC}$):** `fare_amount`, `trip_distance`, `congestion_surcharge`, `speed_mph`, and `is_airport_trip`.
