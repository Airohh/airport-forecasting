"""Honest comparison: LightGBM recursive vs one-step vs SARIMA on the same test horizon."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from airport_forecast.constants import SHORT_NAMES as SHORT, CORE_AIRPORTS as CORE
from airport_forecast.data import load_enriched
from airport_forecast.features import build_features
from airport_forecast.models import (
    evaluate_lightgbm_global,
    evaluate_lightgbm_recursive,
    evaluate_sarima,
    results_to_dataframe,
)

enriched = load_enriched()

# 1. One-step (the OLD, optimistic method — uses real lags from test set)
feat = build_features(enriched)
feat_core = feat[feat["airport"].isin(CORE)].copy()
_, onestep_results = evaluate_lightgbm_global(feat_core, core_airports=CORE)
onestep_test = [r for r in onestep_results if pd.to_datetime(r.dates).min().year >= 2025]

# 2. Recursive (the HONEST method)
print("Running recursive forecast (this rebuilds features per month)...")
_, recursive_results = evaluate_lightgbm_recursive(enriched, val_end="2024-12", core_airports=CORE)

# 3. SARIMA (already honest multi-step)
sarima_results = []
for ap in CORE:
    res = evaluate_sarima(enriched[enriched["airport"].isin(CORE)], ap)
    sarima_results.extend([r for r in res if pd.to_datetime(r.dates).min().year >= 2025])

# Compare
print("\n" + "=" * 70)
print("HONEST COMPARISON (same test horizon 2025+)")
print("=" * 70)
print(f"\n{'Airport':<10} {'LGB 1-step':>12} {'LGB recursive':>14} {'SARIMA':>10}")
print("-" * 50)

onestep_map = {r.airport: r.mape for r in onestep_test}
recursive_map = {r.airport: r.mape for r in recursive_results}
sarima_map = {r.airport: r.mape for r in sarima_results}

for ap in CORE:
    o = onestep_map.get(ap, float("nan"))
    r = recursive_map.get(ap, float("nan"))
    s = sarima_map.get(ap, float("nan"))
    print(f"{SHORT[ap]:<10} {o:>11.1f}% {r:>13.1f}% {s:>9.1f}%")

import numpy as np
print("-" * 50)
print(f"{'AVG':<10} {np.mean(list(onestep_map.values())):>11.1f}% "
      f"{np.mean(list(recursive_map.values())):>13.1f}% "
      f"{np.mean(list(sarima_map.values())):>9.1f}%")

print("\nNote:")
print("  LGB 1-step    = optimistic, uses REAL previous month as lag (not a true forecast)")
print("  LGB recursive = honest, feeds own predictions back as lags")
print("  SARIMA        = honest multi-step baseline")

# Save honest results
df_rec = results_to_dataframe(recursive_results)
df_rec.to_csv(Path(__file__).resolve().parent.parent / "reports" / "recursive_results.csv", index=False)
print("\nRecursive results saved to reports/recursive_results.csv")
