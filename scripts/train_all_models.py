"""Train all models and compare: SARIMA, LightGBM Global, LightGBM Local, Prophet."""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from airport_forecast.constants import SHORT_NAMES as SHORT, CORE_AIRPORTS as CORE
from airport_forecast.data import load_enriched
from airport_forecast.features import build_features
from airport_forecast.models import (
    evaluate_sarima,
    evaluate_lightgbm_global,
    evaluate_lightgbm_local,
    evaluate_prophet,
    results_to_dataframe,
)

FIGS = Path(__file__).resolve().parent.parent / "reports" / "figures"
REPORTS = Path(__file__).resolve().parent.parent / "reports"
FIGS.mkdir(parents=True, exist_ok=True)

print("Loading enriched data...")
enriched = load_enriched()
feat = build_features(enriched)
feat_core = feat[feat["airport"].isin(CORE)].copy()
raw = enriched[enriched["airport"].isin(CORE)].copy()

all_results = []

# ──────────────────────────────────────────────
# 1. SARIMA (per airport)
# ──────────────────────────────────────────────
print("\n=== SARIMA ===")
for ap in CORE:
    t0 = time.time()
    try:
        results = evaluate_sarima(raw, ap)
        all_results.extend(results)
        for r in results:
            print(f"  {SHORT[ap]}: MAPE={r.mape:.1f}%, MAE={r.mae:,.0f}, h={r.horizon}")
    except Exception as e:
        print(f"  {SHORT[ap]}: FAILED - {e}")
    print(f"    ({time.time()-t0:.1f}s)")

# ──────────────────────────────────────────────
# 2. LightGBM GLOBAL
# ──────────────────────────────────────────────
print("\n=== LightGBM Global ===")
t0 = time.time()
lgb_model, lgb_results = evaluate_lightgbm_global(feat_core, core_airports=CORE)
all_results.extend(lgb_results)
for r in lgb_results:
    print(f"  {SHORT[r.airport]}: MAPE={r.mape:.1f}%, MAE={r.mae:,.0f}, h={r.horizon}")
print(f"  ({time.time()-t0:.1f}s)")

# Feature importance
import lightgbm as lgb
fi = pd.DataFrame({
    "feature": lgb_model.feature_name_,
    "importance": lgb_model.feature_importances_,
}).sort_values("importance", ascending=False)
print("\n  Top 10 features:")
for _, row in fi.head(10).iterrows():
    print(f"    {row['feature']}: {row['importance']}")

# ──────────────────────────────────────────────
# 3. LightGBM LOCAL (per airport)
# ──────────────────────────────────────────────
print("\n=== LightGBM Local ===")
for ap in CORE:
    t0 = time.time()
    results = evaluate_lightgbm_local(feat_core, ap)
    all_results.extend(results)
    for r in results:
        print(f"  {SHORT[ap]}: MAPE={r.mape:.1f}%, MAE={r.mae:,.0f}, h={r.horizon}")
    print(f"    ({time.time()-t0:.1f}s)")

# ──────────────────────────────────────────────
# 4. Prophet (per airport)
# ──────────────────────────────────────────────
print("\n=== Prophet ===")
for ap in CORE:
    t0 = time.time()
    try:
        results = evaluate_prophet(raw, ap)
        all_results.extend(results)
        for r in results:
            print(f"  {SHORT[ap]}: MAPE={r.mape:.1f}%, MAE={r.mae:,.0f}, h={r.horizon}")
    except Exception as e:
        print(f"  {SHORT[ap]}: FAILED - {e}")
    print(f"    ({time.time()-t0:.1f}s)")

# ──────────────────────────────────────────────
# RESULTS COMPARISON
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("RESULTS COMPARISON")
print("=" * 70)

df_results = results_to_dataframe(all_results)
df_results["airport_name"] = df_results["airport"].map(SHORT)

# Separate val (horizon=12) and test results
# Val results have dates in 2024, test in 2025+
val_results = []
test_results = []
for r in all_results:
    dates = pd.to_datetime(r.dates)
    if dates.min().year == 2024:
        val_results.append(r)
    else:
        test_results.append(r)

print("\n--- VALIDATION (2024) ---")
df_val = results_to_dataframe(val_results)
if not df_val.empty:
    df_val["airport_name"] = df_val["airport"].map(SHORT)
    pivot_val = df_val.pivot_table(index="airport_name", columns="model", values="mape", aggfunc="mean")
    print(pivot_val.round(1).to_string())

print("\n--- TEST (2025+) ---")
df_test = results_to_dataframe(test_results)
if not df_test.empty:
    df_test["airport_name"] = df_test["airport"].map(SHORT)
    pivot_test = df_test.pivot_table(index="airport_name", columns="model", values="mape", aggfunc="mean")
    print(pivot_test.round(1).to_string())

print("\n--- AVERAGE MAPE BY MODEL ---")
avg = df_results.groupby("model")["mape"].mean().sort_values()
for model, mape in avg.items():
    print(f"  {model}: {mape:.1f}%")

# ──────────────────────────────────────────────
# PLOTS
# ──────────────────────────────────────────────

# Plot 1: MAPE comparison bar chart
fig, ax = plt.subplots(figsize=(14, 6))
if not df_test.empty:
    pivot_plot = df_test.pivot_table(index="airport_name", columns="model", values="mape")
    pivot_plot.plot(kind="bar", ax=ax)
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Model Comparison: MAPE on Test Set (2025+)")
    ax.legend(title="Model")
    plt.xticks(rotation=45)
fig.tight_layout()
fig.savefig(FIGS / "23_model_comparison_mape.png", dpi=150)
plt.close()
print("\n  Plot 23 - Model comparison MAPE")

# Plot 2: Predictions vs actual (test set, best model per airport)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for i, ap in enumerate(CORE):
    ax = axes[i // 3, i % 3]
    ap_results = [r for r in test_results if r.airport == ap]
    if not ap_results:
        ap_results = [r for r in val_results if r.airport == ap]

    for r in ap_results:
        dates = pd.to_datetime(r.dates)
        ax.plot(dates, r.y_true / 1e6, "k-", linewidth=2, label="Actual" if r == ap_results[0] else "")
        ax.plot(dates, r.y_pred / 1e6, "--", alpha=0.7, label=f"{r.model_name} ({r.mape:.1f}%)")

    ax.set_title(SHORT[ap])
    ax.legend(fontsize=7)
    ax.set_ylabel("PAX (M)")
    ax.tick_params(axis="x", rotation=45)

fig.suptitle("Predictions vs Actual", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(FIGS / "24_predictions_vs_actual.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Plot 24 - Predictions vs actual")

# Plot 3: Feature importance (LightGBM Global)
fig, ax = plt.subplots(figsize=(10, 8))
fi_top = fi.head(15)
ax.barh(fi_top["feature"], fi_top["importance"], color="steelblue")
ax.set_xlabel("Importance (split count)")
ax.set_title("LightGBM Global - Top 15 Feature Importance")
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(FIGS / "25_feature_importance.png", dpi=150)
plt.close()
print("  Plot 25 - Feature importance")

# Save results
df_results.to_csv(REPORTS / "model_results.csv", index=False)
fi.to_csv(REPORTS / "feature_importance.csv", index=False)
print(f"\nResults saved to {REPORTS}")
print("Done.")
