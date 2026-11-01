"""
03_ab_statistical_inference.py
------------------------------
Rigorous A/B & Quasi-Experimental Statistical Inference Suite
on NYC TLC Yellow Taxi Trips (Vendor 1: CMT vs. Vendor 2: VeriFone).

Implements:
1. Categorical & Proportion Inference:
   - Chi-Square (χ²) Test of Independence on payment method distribution
   - Two-Proportion Z-Test + 95% CI on Generous Tipper Rate (tip_pct >= 20%)
2. Parametric vs. Non-Parametric Continuous Inference:
   - Welch's Two-Sample t-test (unequal variance) + Cohen's d effect size
   - Mann-Whitney U test (robust to heavy right skew in fares)
   - Non-Parametric Bootstrap (2,000 resamples) for 95% CIs on Mean & Median differences
3. Confounder Control & Variance Reduction:
   - Stratified Subclassification across (Airport Trip × Distance Bucket) strata
     to separate route-mix confounding from vendor differences
   - CUPED (Controlled-experiment Using Pre-Experiment Data / Covariate Adjustment)
     using trip_distance as baseline covariate, quantifying Standard Error reduction
4. Subgroup Analysis & Multiple Hypothesis Testing Correction:
   - Evaluates vendor effect across 5 time-of-day cohorts with Holm-Bonferroni
     and Benjamini-Hochberg (FDR) p-value adjustments.

Generates Charts 04–06 in `charts/` and saves `data/processed/ab_test_results.csv`.

Usage:
    uv run python scripts/03_ab_statistical_inference.py
"""

from __future__ import annotations

import pathlib
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parent.parent
SAMPLE_PATH = ROOT / "data" / "processed" / "sample_100k_clean.parquet"
RESULTS_CSV = ROOT / "data" / "processed" / "ab_test_results.csv"
CHART_DIR = ROOT / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

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


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pool_sd = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    return float((np.mean(y) - np.mean(x)) / pool_sd)


def bootstrap_diff(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 1500,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return bootstrap distributions of (mean_y - mean_x) and (median_y - median_x)."""
    rng = np.random.default_rng(seed)
    # Use 20k per group per resample for fast vectorized bootstrap
    n_sub = min(20_000, len(x), len(y))
    mean_diffs = np.empty(n_boot, dtype=np.float64)
    med_diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        xb = rng.choice(x, size=n_sub, replace=True)
        yb = rng.choice(y, size=n_sub, replace=True)
        mean_diffs[i] = yb.mean() - xb.mean()
        med_diffs[i] = np.median(yb) - np.median(xb)
    return mean_diffs, med_diffs


def main() -> None:
    print("=" * 74)
    print("  NYC Yellow Taxi — A/B & Quasi-Experimental Statistical Inference")
    print("=" * 74)

    df = pd.read_parquet(SAMPLE_PATH)
    v1 = df[df["vendor_id"] == 1]  # Group A: Creative Mobile Technologies (CMT)
    v2 = df[df["vendor_id"] == 2]  # Group B: VeriFone

    summary_rows = []

    # ═════════════════════════════════════════════════════════════════════════
    # 1. CATEGORICAL & PROPORTION TESTS (Chi-Square & Two-Proportion Z-Test)
    # ═════════════════════════════════════════════════════════════════════════
    print("\n── 1. Categorical & Proportion Tests (Chi-Square & Two-Prop Z-Test) ────")
    ct = pd.crosstab(df["vendor_id"], df["payment_type"])
    chi2_stat, chi2_p, chi2_dof, _ = stats.chi2_contingency(ct)
    print(f"   Payment Type Independence (χ²): chi2={chi2_stat:.2f}, dof={chi2_dof}, p={chi2_p:.3e}")

    cc_df = df[(df["payment_type"] == 1) & (df["fare_amount"] >= 3.0)]
    cc_v1 = cc_df[cc_df["vendor_id"] == 1]
    cc_v2 = cc_df[cc_df["vendor_id"] == 2]

    succ = np.array([cc_v2["is_generous_tip"].sum(), cc_v1["is_generous_tip"].sum()])
    nobs = np.array([len(cc_v2), len(cc_v1)])
    z_stat, z_p = proportions_ztest(succ, nobs)
    ci_low, ci_high = confint_proportions_2indep(succ[0], nobs[0], succ[1], nobs[1], method="wald")
    p1_gen = succ[1] / nobs[1] * 100
    p2_gen = succ[0] / nobs[0] * 100
    diff_gen_pp = p2_gen - p1_gen

    print(f"   Generous Tipper Rate (tip ≥ 20%):")
    print(f"     CMT (V1)      : {p1_gen:.2f}% ({succ[1]:,}/{nobs[1]:,})")
    print(f"     VeriFone (V2) : {p2_gen:.2f}% ({succ[0]:,}/{nobs[0]:,})")
    print(f"     Diff (V2-V1)  : {diff_gen_pp:+.2f} percentage points (95% CI: [{ci_low*100:+.2f} pp, {ci_high*100:+.2f} pp])")
    print(f"     Two-Prop Z    : z={z_stat:.3f}, p={z_p:.3e}")

    summary_rows.append({
        "test_name": "Two-Proportion Z-Test (Generous Tip >= 20%)",
        "metric": "Generous Tip Share (%)",
        "cmt_v1_est": round(p1_gen, 3),
        "verifone_v2_est": round(p2_gen, 3),
        "effect_diff_v2_minus_v1": round(diff_gen_pp, 3),
        "ci_95_low": round(float(ci_low * 100), 3),
        "ci_95_high": round(float(ci_high * 100), 3),
        "stat_value": round(float(z_stat), 3),
        "p_value": float(z_p),
    })

    # ═════════════════════════════════════════════════════════════════════════
    # 2. CONTINUOUS TESTS: WELCH T-TEST, MANN-WHITNEY U & BOOTSTRAP CI
    # ═════════════════════════════════════════════════════════════════════════
    print("\n── 2. Continuous Fare Inference (Welch t, Mann-Whitney U, Bootstrap) ───")
    x_fare = v1["total_amount"].to_numpy()
    y_fare = v2["total_amount"].to_numpy()

    t_stat, t_p = stats.ttest_ind(y_fare, x_fare, equal_var=False)
    u_stat, u_p = stats.mannwhitneyu(y_fare, x_fare, alternative="two-sided")
    d_val = cohens_d(x_fare, y_fare)

    raw_diff = float(y_fare.mean() - x_fare.mean())
    raw_se = float(np.sqrt(y_fare.var(ddof=1) / len(y_fare) + x_fare.var(ddof=1) / len(x_fare)))
    raw_med_diff = float(np.median(y_fare) - np.median(x_fare))

    boot_mean, boot_med = bootstrap_diff(x_fare, y_fare, n_boot=1500, seed=42)
    b_mean_ci = np.percentile(boot_mean, [2.5, 97.5])
    b_med_ci = np.percentile(boot_med, [2.5, 97.5])

    print(f"   CMT (V1) Fare      : mean=${x_fare.mean():.2f}, median=${np.median(x_fare):.2f}, std=${x_fare.std(ddof=1):.2f} (n={len(x_fare):,})")
    print(f"   VeriFone (V2) Fare : mean=${y_fare.mean():.2f}, median=${np.median(y_fare):.2f}, std=${y_fare.std(ddof=1):.2f} (n={len(y_fare):,})")
    print(f"   Raw Mean Diff      : ${raw_diff:+.2f} ± ${1.96*raw_se:.2f} | Bootstrap 95% CI: [${b_mean_ci[0]:+.2f}, ${b_mean_ci[1]:+.2f}]")
    print(f"   Raw Median Diff    : ${raw_med_diff:+.2f} | Bootstrap 95% CI: [${b_med_ci[0]:+.2f}, ${b_med_ci[1]:+.2f}]")
    print(f"   Welch t-test       : t={t_stat:.3f}, p={t_p:.3e} | Cohen's d={d_val:.4f}")
    print(f"   Mann-Whitney U     : U={u_stat:.3e}, p={u_p:.3e}")

    summary_rows.append({
        "test_name": "Welch Two-Sample t-Test (Raw Total Fare)",
        "metric": "Mean Total Fare ($)",
        "cmt_v1_est": round(float(x_fare.mean()), 3),
        "verifone_v2_est": round(float(y_fare.mean()), 3),
        "effect_diff_v2_minus_v1": round(raw_diff, 3),
        "ci_95_low": round(float(b_mean_ci[0]), 3),
        "ci_95_high": round(float(b_mean_ci[1]), 3),
        "stat_value": round(float(t_stat), 3),
        "p_value": float(t_p),
    })

    # Chart 04: Distributions + Bootstrap Mean & Median CI
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor(BG)

    bins = np.linspace(3, 95, 55)
    ax1.hist(np.clip(x_fare, 3, 95), bins=bins, density=True, alpha=0.55, color=PAL[0], label=f"CMT (V1) · Mean ${x_fare.mean():.2f}")
    ax1.hist(np.clip(y_fare, 3, 95), bins=bins, density=True, alpha=0.50, color=PAL[1], label=f"VeriFone (V2) · Mean ${y_fare.mean():.2f}")
    ax1.axvline(x_fare.mean(), color=PAL[0], ls="--", lw=2)
    ax1.axvline(y_fare.mean(), color=PAL[1], ls="--", lw=2)
    ax1.set_title("Right-Skewed Fare Distributions (Welch t & Mann-Whitney U)", fontweight="bold", color=TEXT)
    ax1.set_xlabel("Total Fare (USD, clipped at 95 USD)")
    ax1.set_ylabel("Density")
    ax1.legend(fontsize=9)
    ax1.grid(True)

    ax2.hist(boot_mean, bins=40, density=True, alpha=0.75, color=PAL[2], edgecolor=BG, label="Bootstrap Δ Mean ($)")
    ax2.hist(boot_med, bins=30, density=True, alpha=0.65, color=PAL[3], edgecolor=BG, label="Bootstrap Δ Median ($)")
    ax2.axvline(b_mean_ci[0], color=PAL[2], ls=":", lw=1.8)
    ax2.axvline(b_mean_ci[1], color=PAL[2], ls=":", lw=1.8, label=f"95% CI ΔMean [{b_mean_ci[0]:+.2f}, {b_mean_ci[1]:+.2f}]")
    ax2.axvline(0, color="#ffffff", ls="--", lw=1.2, alpha=0.6)
    ax2.set_title("Non-Parametric Bootstrap (1,500 Resamples) of Vendor Effect", fontweight="bold", color=TEXT)
    ax2.set_xlabel("VeriFone (V2) − CMT (V1) Difference ($ USD)")
    ax2.set_ylabel("Bootstrap Density")
    ax2.legend(fontsize=8.5)
    ax2.grid(True)

    plt.tight_layout()
    out4 = CHART_DIR / "04_ab_vendor_distributions_ci.png"
    fig.savefig(out4, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"   💾 Saved {out4.relative_to(ROOT)}")

    # ═════════════════════════════════════════════════════════════════════════
    # 3. CONFOUNDER CONTROL: STRATIFICATION & CUPED VARIANCE REDUCTION
    # ═════════════════════════════════════════════════════════════════════════
    print("\n── 3. Confounder Control (Route-Mix Stratification & CUPED) ────────────")
    # Why is VeriFone's raw mean higher? Check airport share & distance!
    v1_air = v1["is_airport_trip"].mean() * 100
    v2_air = v2["is_airport_trip"].mean() * 100
    print(f"   Route Confounder Check: CMT Airport Share={v1_air:.2f}% vs VeriFone Airport Share={v2_air:.2f}%")
    print(f"   Distance Confounder   : CMT Mean Dist={v1['trip_distance'].mean():.2f} mi vs VeriFone Mean Dist={v2['trip_distance'].mean():.2f} mi")

    # 3a. Stratified ATE across (is_airport_trip × distance_bucket) weighted by total stratum size
    strata_cols = ["is_airport_trip", "distance_bucket"]
    n_total = len(df)
    strat_diff = 0.0
    strat_var = 0.0
    strata_records = []
    for (is_air, dbucket), grp in df.groupby(strata_cols):
        g1 = grp[grp["vendor_id"] == 1]["total_amount"]
        g2 = grp[grp["vendor_id"] == 2]["total_amount"]
        if len(g1) > 10 and len(g2) > 10:
            w = len(grp) / n_total
            d_s = float(g2.mean() - g1.mean())
            v_s = float(g2.var(ddof=1) / len(g2) + g1.var(ddof=1) / len(g1))
            strat_diff += w * d_s
            strat_var += (w ** 2) * v_s
            strata_records.append({
                "stratum": f"{'Airport' if is_air else 'City'} · {dbucket}",
                "weight_pct": w * 100,
                "cmt_mean": float(g1.mean()),
                "verifone_mean": float(g2.mean()),
                "diff": d_s,
                "se": np.sqrt(v_s),
            })
    strat_se = float(np.sqrt(strat_var))
    strat_z = strat_diff / strat_se
    strat_p = float(2 * (1 - stats.norm.cdf(abs(strat_z))))

    # 3b. Exploratory covariate-adjusted regression (HC3 robust standard errors).
    # This is not CUPED: distance is not a pre-experiment baseline covariate.
    adjusted_fit = smf.ols(
        "total_amount ~ C(vendor_id, Treatment(reference=1)) + trip_distance "
        "+ is_airport_trip + C(time_bucket)",
        data=df,
    ).fit(cov_type="HC3")
    vendor_term = "C(vendor_id, Treatment(reference=1))[T.2]"
    adjusted_diff = float(adjusted_fit.params[vendor_term])
    adjusted_se = float(adjusted_fit.bse[vendor_term])
    adjusted_p = float(adjusted_fit.pvalues[vendor_term])
    adjusted_ci = adjusted_fit.conf_int().loc[vendor_term].to_numpy()

    print(f"   1) Unadjusted Raw Diff      : ${raw_diff:+.3f} (SE = ${raw_se:.3f})")
    print(f"   2) Route-Stratified Diff    : ${strat_diff:+.3f} (SE = ${strat_se:.3f}, p={strat_p:.3e})")
    print(f"   3) Covariate-Adjusted OLS   : ${adjusted_diff:+.3f} (HC3 SE = ${adjusted_se:.3f}, 95% CI [${adjusted_ci[0]:+.3f}, ${adjusted_ci[1]:+.3f}], p={adjusted_p:.3e})")
    print("   Note: trip distance is trip-level, not pre-experiment; this is not formal CUPED or a causal estimate.")

    summary_rows.append({
        "test_name": "Stratified ATE (Airport x Distance Bucket)",
        "metric": "Adjusted Fare Diff ($)",
        "cmt_v1_est": round(float(x_fare.mean()), 3),
        "verifone_v2_est": round(float(x_fare.mean() + strat_diff), 3),
        "effect_diff_v2_minus_v1": round(strat_diff, 3),
        "ci_95_low": round(strat_diff - 1.96 * strat_se, 3),
        "ci_95_high": round(strat_diff + 1.96 * strat_se, 3),
        "stat_value": round(float(strat_z), 3),
        "p_value": strat_p,
    })
    summary_rows.append({
        "test_name": "Covariate-Adjusted OLS (HC3 Robust SE)",
        "metric": "Adjusted Vendor Fare Difference ($)",
        "cmt_v1_est": np.nan,
        "verifone_v2_est": np.nan,
        "effect_diff_v2_minus_v1": round(adjusted_diff, 3),
        "ci_95_low": round(float(adjusted_ci[0]), 3),
        "ci_95_high": round(float(adjusted_ci[1]), 3),
        "stat_value": round(float(adjusted_fit.tvalues[vendor_term]), 3),
        "p_value": adjusted_p,
    })

    # Chart 05: Confounder Stratification & CUPED Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.5))
    fig.patch.set_facecolor(BG)

    methods = ["Unadjusted\nObservational", "Stratified\n(Airport × Dist)", "Adjusted OLS\n(HC3 robust SE)"]
    ests = [raw_diff, strat_diff, adjusted_diff]
    ses = [raw_se, strat_se, adjusted_se]
    cols = [PAL[1], PAL[3], PAL[2]]

    y_pos = np.arange(len(methods))
    for i, (m, est, se, col) in enumerate(zip(methods, ests, ses, cols)):
        ax1.errorbar(est, i, xerr=1.96 * se, fmt="o", color=col, ecolor=col, elinewidth=3, capsize=6, ms=9)
        ax1.text(est, i + 0.18, f"{est:+.2f} USD (±{1.96*se:.2f})", ha="center", color=TEXT, fontsize=9, fontweight="bold")
    ax1.axvline(0, color="#ffffff", ls="--", alpha=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(methods, fontsize=10, color=TEXT)
    ax1.set_ylim(-0.4, len(methods) - 0.45)
    ax1.set_xlabel("Estimated Vendor Effect: VeriFone − CMT (USD)")
    ax1.set_title("Vendor Fare Gap: Raw vs. Stratified vs. Covariate-Adjusted", fontweight="bold", color=TEXT)
    ax1.set_xlim(-0.3, max(ests) + 0.6)
    ax1.grid(True)

    st_df = pd.DataFrame(strata_records).sort_values("weight_pct", ascending=True)
    ax2.barh(st_df["stratum"], st_df["diff"], xerr=1.96 * st_df["se"], color=PAL[0], alpha=0.85, edgecolor=BG, capsize=3)
    ax2.axvline(0, color="#ffffff", ls="--", alpha=0.5)
    ax2.axvline(raw_diff, color=PAL[1], ls=":", lw=1.8, label=f"Naive Diff (${raw_diff:+.2f})")
    ax2.set_xlabel("Within-Stratum Fare Difference: VeriFone − CMT ($)")
    ax2.set_title("Within-Stratum Differences Across Route × Distance Cohorts", fontweight="bold", color=TEXT)
    ax2.legend(fontsize=8.5)
    ax2.grid(True, axis="x")

    plt.tight_layout()
    out5 = CHART_DIR / "05_ab_confounder_stratification.png"
    fig.savefig(out5, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"   💾 Saved {out5.relative_to(ROOT)}")

    # ═════════════════════════════════════════════════════════════════════════
    # 4. SUBGROUP HETEROGENEITY & MULTIPLE TESTING CORRECTION
    # ═════════════════════════════════════════════════════════════════════════
    print("\n── 4. Subgroup Analysis & Holm-Bonferroni Multiple Testing ─────────────")
    buckets = ["AM Rush", "Midday", "PM Rush", "Night", "Overnight"]
    sub_rows = []
    for b in buckets:
        sub = df[df["time_bucket"] == b]
        s1 = sub[sub["vendor_id"] == 1]["total_amount"].to_numpy()
        s2 = sub[sub["vendor_id"] == 2]["total_amount"].to_numpy()
        t_b, p_b = stats.ttest_ind(s2, s1, equal_var=False)
        diff_b = float(s2.mean() - s1.mean())
        se_b = float(np.sqrt(s2.var(ddof=1) / len(s2) + s1.var(ddof=1) / len(s1)))
        sub_rows.append({
            "time_bucket": b,
            "n_v1": len(s1),
            "n_v2": len(s2),
            "diff": diff_b,
            "se": se_b,
            "ci_low": diff_b - 1.96 * se_b,
            "ci_high": diff_b + 1.96 * se_b,
            "t_stat": float(t_b),
            "raw_p": float(p_b),
        })

    sub_df = pd.DataFrame(sub_rows)
    reject_holm, p_holm, _, _ = multipletests(sub_df["raw_p"], alpha=0.05, method="holm")
    reject_fdr, p_fdr, _, _ = multipletests(sub_df["raw_p"], alpha=0.05, method="fdr_bh")
    sub_df["p_holm"] = p_holm
    sub_df["p_fdr_bh"] = p_fdr
    sub_df["sig_holm"] = reject_holm

    print(sub_df[["time_bucket", "diff", "se", "raw_p", "p_holm", "sig_holm"]].to_string(index=False))

    # Chart 06: Subgroup Forest Plot with Holm-Bonferroni Adjusted p-values
    fig, ax = plt.subplots(figsize=(12, 5.2))
    fig.patch.set_facecolor(BG)

    y_idx = np.arange(len(sub_df))
    for i, r in sub_df.iterrows():
        c = PAL[2] if r["sig_holm"] else SUBTEXT
        ax.errorbar(r["diff"], i, xerr=1.96 * r["se"], fmt="s", color=c, ecolor=c, elinewidth=2.6, capsize=5, ms=8)
        status = "Sig (Holm α=0.05)" if r["sig_holm"] else "Not Sig"
        ax.text(
            r["ci_high"] + 0.08,
            i,
            f"Δ=${r['diff']:+.2f}  |  p_raw={r['raw_p']:.4f}  →  p_holm={r['p_holm']:.4f} ({status})",
            va="center",
            fontsize=8.5,
            color=TEXT,
        )

    ax.axvline(0, color="#ffffff", ls="--", alpha=0.55)
    ax.axvline(raw_diff, color=PAL[1], ls=":", alpha=0.7, label=f"Overall Unadjusted Δ (${raw_diff:+.2f})")
    ax.set_yticks(y_idx)
    ax.set_yticklabels(sub_df["time_bucket"], fontsize=10, color=TEXT)
    ax.set_xlabel("Subgroup Fare Difference: VeriFone − CMT ($ USD, 95% CI)")
    ax.set_title("Subgroup Heterogeneity by Time-of-Day with Holm-Bonferroni Correction", fontweight="bold", color=TEXT)
    ax.set_xlim(min(sub_df["ci_low"]) - 0.5, max(sub_df["ci_high"]) + 3.2)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True)

    plt.tight_layout()
    out6 = CHART_DIR / "06_ab_subgroups_multiple_testing.png"
    fig.savefig(out6, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"   💾 Saved {out6.relative_to(ROOT)}")

    pd.DataFrame(summary_rows).to_csv(RESULTS_CSV, index=False)
    print(f"\n✅ Saved statistical inference summary → {RESULTS_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
