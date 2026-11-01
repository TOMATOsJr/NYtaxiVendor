"""
04_ml_tip_prototype.py
----------------------
Modest Tabular ML Prototype using Scikit-learn:
Predicting Generous Tippers (tip_pct >= 20%) on NYC Credit-Card Taxi Trips.

Why this is scoped for the JD:
- Demonstrates clean Scikit-learn Pipeline & ColumnTransformer design
  (StandardScaler for numerical features + OneHotEncoder for categorical features).
- Compares an interpretable statistical baseline (Logistic Regression) against
  a gradient-boosted tree model (HistGradientBoostingClassifier).
- Evaluates Discrimination (ROC-AUC, PR-AUC), Probability Calibration (Brier score
  & Reliability curve), and Permutation Feature Importance.
- Saves a 3-panel evaluation figure to `charts/07_ml_prototype_evaluation.png`
  and metrics to `data/processed/ml_metrics.csv`.

Usage:
    uv run python scripts/04_ml_tip_prototype.py
"""

from __future__ import annotations

import pathlib
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parent.parent
SAMPLE_PATH = ROOT / "data" / "processed" / "sample_100k_clean.parquet"
METRICS_CSV = ROOT / "data" / "processed" / "ml_metrics.csv"
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
    "axes.titlesize": 11.5,
    "axes.labelsize": 9.5,
    "figure.titlesize": 14,
})


def main() -> None:
    print("=" * 74)
    print("  NYC Yellow Taxi — Scikit-learn Tip Propensity Prototype (tip >= 20%)")
    print("=" * 74)

    df = pd.read_parquet(SAMPLE_PATH)
    # Filter to credit-card trips where tips are recorded on the meter
    cc = df[(df["payment_type"] == 1) & (df["fare_amount"] >= 3.0)].copy()

    # Pre-payment features only (strictly avoiding target leakage from tip_amount or total_amount)
    num_features = [
        "fare_amount",
        "trip_distance",
        "trip_minutes",
        "speed_mph",
        "tolls_amount",
        "congestion_surcharge",
        "airport_fee",
        "pickup_hour",
        "passenger_count",
        "is_weekend",
        "is_airport_trip",
    ]
    cat_features = ["vendor_id", "pickup_borough", "time_bucket", "distance_bucket"]
    target = "is_generous_tip"

    X = cc[num_features + cat_features].copy()
    X["vendor_id"] = X["vendor_id"].astype(str)
    y = cc[target].astype(int).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print(f"   Credit-card dataset : {len(cc):,} rows (Train: {len(X_train):,} | Test: {len(X_test):,})")
    print(f"   Positive class rate : {y.mean()*100:.2f}% generous tippers (tip >= 20% of base fare)")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
        ]
    )

    models = {
        "Logistic Regression (Baseline)": Pipeline([
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=500, random_state=42)),
        ]),
        "HistGradientBoosting (Prototype)": Pipeline([
            ("prep", preprocessor),
            ("clf", HistGradientBoostingClassifier(max_depth=6, learning_rate=0.07, max_iter=180, random_state=42)),
        ]),
    }

    metrics_rows = []
    preds = {}

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        preds[name] = prob

        roc_auc = roc_auc_score(y_test, prob)
        pr_auc = average_precision_score(y_test, prob)
        brier = brier_score_loss(y_test, prob)
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred)

        metrics_rows.append({
            "model": name,
            "roc_auc": round(float(roc_auc), 4),
            "pr_auc": round(float(pr_auc), 4),
            "brier_score": round(float(brier), 4),
            "accuracy": round(float(acc), 4),
            "f1_score": round(float(f1), 4),
        })

    metrics_df = pd.DataFrame(metrics_rows)
    print("\n── Held-Out Test Set Performance ────────────────────────────────────────")
    print(metrics_df.to_string(index=False))
    metrics_df.to_csv(METRICS_CSV, index=False)

    # Permutation Importance on the HistGradientBoosting prototype
    hgb_pipe = models["HistGradientBoosting (Prototype)"]
    perm = permutation_importance(
        hgb_pipe,
        X_test.iloc[:5000],
        y_test[:5000],
        scoring="roc_auc",
        n_repeats=6,
        random_state=42,
    )
    imp_df = (
        pd.DataFrame({
            "feature": num_features + cat_features,
            "importance": perm.importances_mean,
            "std": perm.importances_std,
        })
        .sort_values("importance", ascending=True)
        .tail(10)
    )

    # Chart 07: 3-Panel ML Prototype Evaluation (ROC, Calibration, Permutation Importance)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.5, 5.2))
    fig.patch.set_facecolor(BG)

    # Panel 1: ROC Curves
    for (name, prob), col in zip(preds.items(), [PAL[0], PAL[2]]):
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc_v = roc_auc_score(y_test, prob)
        ax1.plot(fpr, tpr, lw=2.3, color=col, label=f"{name}\n(AUC = {auc_v:.3f})")
    ax1.plot([0, 1], [0, 1], ls="--", color=SUBTEXT, alpha=0.6)
    ax1.set_title("ROC Curve — Held-Out Test Set", fontweight="bold", color=TEXT)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend(loc="lower right", fontsize=8.5)
    ax1.grid(True)

    # Panel 2: Calibration / Reliability Curve
    for (name, prob), col in zip(preds.items(), [PAL[0], PAL[2]]):
        frac_pos, mean_pred = calibration_curve(y_test, prob, n_bins=10)
        brier_v = brier_score_loss(y_test, prob)
        ax2.plot(mean_pred, frac_pos, marker="o", lw=2.0, color=col, label=f"{name.split()[0]} (Brier={brier_v:.3f})")
    ax2.plot([0, 1], [0, 1], ls="--", color="#ffffff", alpha=0.5, label="Perfect Calibration")
    ax2.set_title("Probability Calibration Curve", fontweight="bold", color=TEXT)
    ax2.set_xlabel("Mean Predicted Probability")
    ax2.set_ylabel("Observed Fraction of Generous Tips")
    ax2.legend(loc="upper left", fontsize=8.5)
    ax2.grid(True)

    # Panel 3: Permutation Feature Importance
    ax3.barh(imp_df["feature"], imp_df["importance"], xerr=imp_df["std"], color=PAL[3], alpha=0.88, edgecolor=BG, capsize=3)
    ax3.set_title("Top Drivers of Generous Tips (Permutation ΔAUC)", fontweight="bold", color=TEXT)
    ax3.set_xlabel("Decrease in ROC-AUC when Permuted")
    ax3.grid(True, axis="x")

    plt.tight_layout()
    out7 = CHART_DIR / "07_ml_prototype_evaluation.png"
    fig.savefig(out7, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"\n   💾 Saved {out7.relative_to(ROOT)}")
    print(f"✅ Saved ML evaluation metrics → {METRICS_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
