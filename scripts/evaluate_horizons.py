"""Evaluate LightGBM recursive vs SARIMA at multiple forecast horizons (M+1, M+3, M+6, M+12)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from airport_forecast.features import build_features, temporal_train_val_test_split
from airport_forecast.models import (
    ForecastResult,
    recursive_forecast_global,
    results_to_dataframe,
    train_lightgbm_global,
    train_sarima,
    FEATURE_COLS,
)

REPORTS = Path(__file__).resolve().parent.parent / "reports"
SHORT = {
    "FR_LFLL": "Lyon", "FR_LFRS": "Nantes", "HU_LHBP": "Budapest",
    "PT_LPPT": "Lisbon", "PT_LPPR": "Porto", "RS_LYBE": "Belgrade",
}
CORE = list(SHORT.keys())
HORIZONS = [1, 3, 6, 12]
VAL_END = "2024-12"

enriched = pd.read_parquet(
    Path(__file__).resolve().parent.parent / "data" / "processed" / "pax_enriched.parquet"
)
enriched["date"] = pd.to_datetime(enriched["date"])

# Train LightGBM once on data up to VAL_END
print("Building features and training LightGBM Global...")
feat = build_features(enriched)
feat_core = feat[feat["airport"].isin(CORE)].copy()
train, val, test = temporal_train_val_test_split(feat_core, VAL_END, VAL_END)
lag_cols = [c for c in train.columns if "lag" in c or "rolling" in c]
train_clean = train.dropna(subset=lag_cols)
model, feature_cols = train_lightgbm_global(train_clean, None)

# Get test dates sorted
test_dates = sorted(feat_core[feat_core["date"] > pd.Timestamp(VAL_END)]["date"].unique())
if not test_dates:
    print("No test dates available.")
    sys.exit(1)

print(f"Test dates available: {len(test_dates)} months ({test_dates[0].strftime('%Y-%m')} to {test_dates[-1].strftime('%Y-%m')})")

all_results = []

for h in HORIZONS:
    if h > len(test_dates):
        print(f"\nSkipping M+{h}: only {len(test_dates)} test months available")
        continue

    print(f"\n{'='*60}")
    print(f"HORIZON M+{h}")
    print(f"{'='*60}")

    # --- LightGBM Recursive ---
    # Origin = VAL_END, forecast h months, take only first h
    fc = recursive_forecast_global(
        model, feature_cols, enriched,
        origin_date=VAL_END,
        airports=CORE,
    )
    fc = fc.sort_values(["airport", "date"])

    for ap in CORE:
        fc_ap = fc[fc["airport"] == ap].head(h)
        fc_ap = fc_ap.dropna(subset=["pax_actual", "pax_pred"])
        if fc_ap.empty:
            continue
        r = ForecastResult(
            model_name=f"LightGBM_Recursive",
            airport=ap,
            horizon=h,
            y_true=fc_ap["pax_actual"].values.astype(float),
            y_pred=fc_ap["pax_pred"].values.astype(float),
            dates=fc_ap["date"].values,
        )
        all_results.append(r)
        print(f"  LGB Recursive {SHORT[ap]:>10s} M+{h}: MAPE={r.mape:.1f}%")

    # --- SARIMA ---
    raw_core = enriched[enriched["airport"].isin(CORE)].copy()
    for ap in CORE:
        sub = raw_core[raw_core["airport"] == ap].sort_values("date")
        train_series = sub[sub["date"] <= VAL_END].set_index("date")["pax"]
        test_sub = sub[sub["date"] > VAL_END].head(h)
        if len(test_sub) < h or len(train_series) < 24:
            continue
        pred = train_sarima(train_series, h)
        r = ForecastResult(
            model_name="SARIMA",
            airport=ap,
            horizon=h,
            y_true=test_sub["pax"].values,
            y_pred=pred[:len(test_sub)],
            dates=test_sub["date"].values,
        )
        all_results.append(r)
        print(f"  SARIMA         {SHORT[ap]:>10s} M+{h}: MAPE={r.mape:.1f}%")

# Summary table
print(f"\n{'='*60}")
print("SUMMARY: Average MAPE by Model × Horizon")
print(f"{'='*60}")

df = results_to_dataframe(all_results)
pivot = df.pivot_table(index="model", columns="horizon", values="mape", aggfunc="mean")
pivot = pivot[sorted(pivot.columns)]
pivot.columns = [f"M+{c}" for c in pivot.columns]
print(pivot.round(1).to_string())

# Per airport detail
print(f"\n{'='*60}")
print("DETAIL: MAPE per Airport × Model × Horizon")
print(f"{'='*60}")

for h in HORIZONS:
    sub = df[df["horizon"] == h]
    if sub.empty:
        continue
    print(f"\n--- M+{h} ---")
    sub["airport_name"] = sub["airport"].map(SHORT)
    pivot_ap = sub.pivot_table(index="airport_name", columns="model", values="mape")
    print(pivot_ap.round(1).to_string())

# Save
out_path = REPORTS / "horizon_results.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")
