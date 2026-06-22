"""Evaluate LightGBM recursive vs SARIMA vs Naive at multiple horizons.

Includes:
- Naive seasonal baseline (same month last year)
- MASE (vs naive), bias, MAPE
- Expanding window cross-validation (3 folds)
- Quantile regression prediction intervals (10th/90th)
"""

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

enriched = pd.read_parquet(
    Path(__file__).resolve().parent.parent / "data" / "processed" / "pax_enriched.parquet"
)
enriched["date"] = pd.to_datetime(enriched["date"])


def get_naive_seasonal(df: pd.DataFrame, airport: str, test_dates: list) -> np.ndarray:
    """Naive baseline: same month last year."""
    sub = df[df["airport"] == airport].set_index("date")["pax"].sort_index()
    naive = []
    for d in test_dates:
        d = pd.Timestamp(d)
        last_year = d - pd.DateOffset(months=12)
        if last_year in sub.index:
            naive.append(sub[last_year])
        else:
            naive.append(np.nan)
    return np.array(naive, dtype=float)


def evaluate_fold(enriched_df, val_end, horizons, fold_name=""):
    """Run one evaluation fold: train up to val_end, forecast test horizons."""
    feat = build_features(enriched_df)
    feat_core = feat[feat["airport"].isin(CORE)].copy()
    train, _, test = temporal_train_val_test_split(feat_core, val_end, val_end)
    lag_cols = [c for c in train.columns if "lag" in c or "rolling" in c]
    train_clean = train.dropna(subset=lag_cols)
    model, feature_cols = train_lightgbm_global(train_clean, None)

    test_dates = sorted(feat_core[feat_core["date"] > pd.Timestamp(val_end)]["date"].unique())
    if not test_dates:
        return []

    all_results = []
    raw_core = enriched_df[enriched_df["airport"].isin(CORE)].copy()

    for h in horizons:
        if h > len(test_dates):
            continue

        # --- LightGBM Recursive ---
        fc = recursive_forecast_global(
            model, feature_cols, enriched_df,
            origin_date=val_end, airports=CORE,
        )
        fc = fc.sort_values(["airport", "date"])

        for ap in CORE:
            fc_ap = fc[fc["airport"] == ap].head(h)
            fc_ap = fc_ap.dropna(subset=["pax_actual", "pax_pred"])
            if fc_ap.empty:
                continue
            y_naive = get_naive_seasonal(enriched_df, ap, fc_ap["date"].values)
            all_results.append(ForecastResult(
                model_name="LightGBM_Recursive",
                airport=ap, horizon=h,
                y_true=fc_ap["pax_actual"].values.astype(float),
                y_pred=fc_ap["pax_pred"].values.astype(float),
                dates=fc_ap["date"].values,
                y_naive=y_naive,
            ))

        # --- SARIMA ---
        for ap in CORE:
            sub = raw_core[raw_core["airport"] == ap].sort_values("date")
            train_series = sub[sub["date"] <= val_end].set_index("date")["pax"]
            test_sub = sub[sub["date"] > val_end].head(h)
            if len(test_sub) < h or len(train_series) < 24:
                continue
            pred = train_sarima(train_series, h)
            y_naive = get_naive_seasonal(enriched_df, ap, test_sub["date"].values)
            all_results.append(ForecastResult(
                model_name="SARIMA",
                airport=ap, horizon=h,
                y_true=test_sub["pax"].values,
                y_pred=pred[:len(test_sub)],
                dates=test_sub["date"].values,
                y_naive=y_naive,
            ))

        # --- Naive Seasonal baseline ---
        for ap in CORE:
            sub = raw_core[raw_core["airport"] == ap].sort_values("date")
            test_sub = sub[sub["date"] > val_end].head(h)
            if len(test_sub) < h:
                continue
            y_naive = get_naive_seasonal(enriched_df, ap, test_sub["date"].values)
            valid = ~np.isnan(y_naive)
            if valid.sum() == 0:
                continue
            all_results.append(ForecastResult(
                model_name="Naive_Seasonal",
                airport=ap, horizon=h,
                y_true=test_sub["pax"].values[valid],
                y_pred=y_naive[valid],
                dates=test_sub["date"].values[valid],
            ))

    return all_results


# ═══════════════════════════════════════════════
# MAIN EVALUATION (primary fold)
# ═══════════════════════════════════════════════
print("=" * 60)
print("HORIZON EVALUATION (primary fold: train -> 2024-12)")
print("=" * 60)

primary_results = evaluate_fold(enriched, "2024-12", HORIZONS, "primary")
df_primary = results_to_dataframe(primary_results)

print("\nSUMMARY: Avg MAPE by Model × Horizon")
pivot_mape = df_primary.pivot_table(index="model", columns="horizon", values="mape", aggfunc="mean")
pivot_mape = pivot_mape[sorted(pivot_mape.columns)]
pivot_mape.columns = [f"M+{c}" for c in pivot_mape.columns]
print(pivot_mape.round(1).to_string())

print("\nBIAS (avg over/under-estimation in PAX):")
pivot_bias = df_primary.pivot_table(index="model", columns="horizon", values="bias", aggfunc="mean")
pivot_bias = pivot_bias[sorted(pivot_bias.columns)]
pivot_bias.columns = [f"M+{c}" for c in pivot_bias.columns]
print(pivot_bias.round(0).to_string())

print("\nMASE (< 1 = beats naive seasonal):")
mase_data = df_primary[df_primary["mase"] > 0]
if not mase_data.empty:
    pivot_mase = mase_data.pivot_table(index="model", columns="horizon", values="mase", aggfunc="mean")
    pivot_mase = pivot_mase[sorted(pivot_mase.columns)]
    pivot_mase.columns = [f"M+{c}" for c in pivot_mase.columns]
    print(pivot_mase.round(2).to_string())

# ═══════════════════════════════════════════════
# EXPANDING WINDOW CROSS-VALIDATION (3 folds)
# ═══════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXPANDING WINDOW CV (3 folds)")
print("=" * 60)

CV_FOLDS = [
    ("2022-12", "Fold 1: train->2022-12, test 2023"),
    ("2023-12", "Fold 2: train->2023-12, test 2024"),
    ("2024-12", "Fold 3: train->2024-12, test 2025+"),
]

cv_results = []
for val_end, desc in CV_FOLDS:
    print(f"\n--- {desc} ---")
    fold_results = evaluate_fold(enriched, val_end, [1, 3, 6, 12], desc)
    for r in fold_results:
        r.model_name = f"{r.model_name}"
    cv_results.extend(fold_results)
    df_fold = results_to_dataframe(fold_results)
    if not df_fold.empty:
        avg = df_fold.groupby("model")["mape"].mean()
        for m, v in avg.items():
            print(f"  {m}: {v:.1f}%")

df_cv = results_to_dataframe(cv_results)

print("\n--- CV AVERAGE ACROSS 3 FOLDS ---")
pivot_cv = df_cv.pivot_table(index="model", columns="horizon", values="mape", aggfunc="mean")
pivot_cv = pivot_cv[sorted(pivot_cv.columns)]
pivot_cv.columns = [f"M+{c}" for c in pivot_cv.columns]
print(pivot_cv.round(1).to_string())

print("\n--- CV STD (stability) ---")
pivot_std = df_cv.pivot_table(index="model", columns="horizon", values="mape", aggfunc="std")
pivot_std = pivot_std[sorted(pivot_std.columns)]
pivot_std.columns = [f"M+{c}" for c in pivot_std.columns]
print(pivot_std.round(1).to_string())

# ═══════════════════════════════════════════════
# QUANTILE REGRESSION (prediction intervals)
# ═══════════════════════════════════════════════
print("\n" + "=" * 60)
print("PREDICTION INTERVALS (LightGBM quantile regression)")
print("=" * 60)

import lightgbm as lgb

feat = build_features(enriched)
feat_core = feat[feat["airport"].isin(CORE)].copy()
train, _, test = temporal_train_val_test_split(feat_core, "2024-12", "2024-12")
lag_cols = [c for c in train.columns if "lag" in c or "rolling" in c]
train_clean = train.dropna(subset=lag_cols)
test_clean = test.dropna(subset=lag_cols)

feature_cols = [c for c in FEATURE_COLS if c in train_clean.columns]
if "airport" in train_clean.columns:
    train_clean = train_clean.copy()
    test_clean = test_clean.copy()
    train_clean["airport_cat"] = train_clean["airport"].astype("category")
    test_clean["airport_cat"] = test_clean["airport"].astype("category")
    all_features = ["airport_cat"] + feature_cols
else:
    all_features = feature_cols

X_train = train_clean[all_features]
y_train = train_clean["pax"]
X_test = test_clean[all_features]

quantile_preds = {}
for q, label in [(0.1, "p10"), (0.5, "p50"), (0.9, "p90")]:
    model_q = lgb.LGBMRegressor(
        objective="quantile", alpha=q,
        n_estimators=500, max_depth=8, learning_rate=0.05,
        verbose=-1, n_jobs=-1, random_state=42,
    )
    model_q.fit(X_train, y_train, categorical_feature=["airport_cat"])
    quantile_preds[label] = np.maximum(model_q.predict(X_test), 0)

test_clean = test_clean.copy()
test_clean["pred_p10"] = quantile_preds["p10"]
test_clean["pred_p50"] = quantile_preds["p50"]
test_clean["pred_p90"] = quantile_preds["p90"]
test_clean["in_interval"] = (
    (test_clean["pax"] >= test_clean["pred_p10"]) &
    (test_clean["pax"] <= test_clean["pred_p90"])
)

coverage = test_clean["in_interval"].mean() * 100
avg_width = (test_clean["pred_p90"] - test_clean["pred_p10"]).mean()

print(f"\n80% prediction interval (P10–P90):")
print(f"  Coverage: {coverage:.1f}% (target: 80%)")
print(f"  Avg interval width: {avg_width:,.0f} PAX")

print("\nPer airport:")
for ap in CORE:
    sub = test_clean[test_clean["airport"] == ap]
    if sub.empty:
        continue
    cov = sub["in_interval"].mean() * 100
    width = (sub["pred_p90"] - sub["pred_p10"]).mean()
    print(f"  {SHORT[ap]:>10s}: coverage={cov:.0f}%, width={width:,.0f} PAX")

# Save interval predictions
interval_cols = ["airport", "date", "pax", "pred_p10", "pred_p50", "pred_p90", "in_interval"]
test_clean[interval_cols].to_csv(REPORTS / "prediction_intervals.csv", index=False)

# ═══════════════════════════════════════════════
# SAVE ALL RESULTS
# ═══════════════════════════════════════════════
df_primary.to_csv(REPORTS / "horizon_results.csv", index=False)
df_cv.to_csv(REPORTS / "cv_results.csv", index=False)
print(f"\nSaved: horizon_results.csv, cv_results.csv, prediction_intervals.csv")
