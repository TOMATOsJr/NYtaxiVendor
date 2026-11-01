# Methodology & Statistical Notes

## 1. Why PySpark & Spark SQL?
Rather than loading 2.96M trip rows into a single-threaded embedded SQLite file, this pipeline uses **Apache Spark (`pyspark.sql`)** in local multi-core execution mode (`local[*]`) with **Adaptive Query Execution (AQE)** enabled:
- Reads columnar **Apache Parquet** files directly with predicate pushdown and column pruning.
- Performs distributed hash joins against the official **NYC TLC Taxi Zone Lookup** (`265` zones across `6` borough categories).
- Executes analytical **Spark SQL** queries (`sql/01_*.sql` – `sql/04_*.sql`) using window functions (`LAG`, `RANK`, `NTILE`, `ROW_NUMBER`, and sliding windows `ROWS BETWEEN 6 PRECEDING AND CURRENT ROW`).

---

## 2. Rigorous A/B & Quasi-Experimental Inference (`scripts/03_ab_statistical_inference.py`)

### Observational vs. Randomized Experiments
In NYC Yellow Taxi operations, trips are recorded by two TPEP technology vendors:
- **Group A (Vendor 1):** Creative Mobile Technologies (CMT)
- **Group B (Vendor 2):** VeriFone

Because riders do not randomly assign themselves to a TPEP vendor in a controlled lab setting, a naive comparison of raw average fares is an **observational A/B comparison** vulnerable to **confounding** (e.g., if one vendor has a higher proportion of JFK/LaGuardia airport trips or longer trips).

To address this rigorously, the project implements a 4-stage statistical inference workflow:

1. **Categorical & Proportion Tests ($\chi^2$ & Two-Proportion Z-test):**
   - Tests whether payment method choice (`payment_type`) is independent of vendor using Pearson's $\chi^2$ test.
   - Tests whether the **Generous Tipper Rate** ($\text{tip\_pct} \ge 20\%$) differs across vendors using a two-proportion Z-test and 95% Wald/Wilson confidence intervals.

2. **Parametric vs. Non-Parametric Continuous Tests:**
   - **Welch's Two-Sample t-test** (unequal variances) tests the difference in means $\Delta \mu = \bar{Y}_{\text{VeriFone}} - \bar{Y}_{\text{CMT}}$, accompanied by **Cohen's $d$** standardized effect size.
   - **Mann-Whitney U test** (Wilcoxon rank-sum) provides a rank-based non-parametric test insensitive to extreme right-tail fare outliers.
   - **Non-Parametric Bootstrap (1,500 resamples)** constructs empirical 95% confidence intervals for both the mean difference ($\Delta \mu$) and median difference ($\Delta \text{median}$).

3. **Confounder Exploration via Stratification & Covariate-Adjusted Regression:**
   - **Route-Mix Stratification:** Computes the weighted Average Treatment Effect across `(is_airport_trip × distance_bucket)` strata so airport trips are only compared against airport trips of similar distance.
   - **Covariate-adjusted OLS:** Regresses fare on vendor, trip distance, airport-trip indicator, and time-of-day bucket, reporting an HC3 heteroskedasticity-robust confidence interval. This is an adjusted association, not a causal effect.
   - **Important terminology:** This is **not CUPED**. Trip distance is measured as part of the trip and is not a genuine pre-experiment baseline covariate. Formal CUPED is appropriate for randomized experiments with pre-treatment covariates.

4. **Subgroup Analysis & Multiple Hypothesis Testing Correction:**
   - Evaluates the vendor effect across 5 time-of-day cohorts (`AM Rush`, `Midday`, `PM Rush`, `Night`, `Overnight`).
   - Applies **Holm-Bonferroni** (family-wise error rate) and **Benjamini-Hochberg** (false discovery rate) adjustments to prevent false positives from multiple comparisons.

---

## 3. Modest Tabular ML Prototype (`scripts/04_ml_tip_prototype.py`)
- **Objective:** Predict whether a credit-card rider leaves a **generous tip ($\ge 20\%$ of base fare)** using only pre-payment trip context (`fare_amount`, `trip_distance`, `trip_minutes`, `speed_mph`, `tolls_amount`, `congestion_surcharge`, `airport_fee`, `pickup_hour`, `passenger_count`, `is_weekend`, `is_airport_trip`, `vendor_id`, `pickup_borough`, `time_bucket`, `distance_bucket`).
- **Leakage Prevention:** `tip_amount`, `tip_pct`, and `total_amount` are strictly excluded from the feature matrix `X`.
- **Architecture:** A `scikit-learn` `ColumnTransformer` (`StandardScaler` + `OneHotEncoder`) feeding both an interpretable **Logistic Regression** baseline and a **HistGradientBoostingClassifier** prototype, evaluated via **ROC-AUC**, **PR-AUC**, **Brier Score / Calibration Curve**, and **Permutation Feature Importance**.
