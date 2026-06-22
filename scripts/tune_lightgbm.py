"""Optuna hyperparameter tuning for LightGBM Global model."""

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from airport_forecast.constants import (
    SHORT_NAMES as SHORT, CORE_AIRPORTS as CORE, TRAIN_END, VAL_END,
)
from airport_forecast.data import load_enriched
from airport_forecast.features import build_features, temporal_train_val_test_split
from airport_forecast.models import FEATURE_COLS, recursive_forecast_global

enriched = load_enriched()
enriched_core = enriched[enriched["airport"].isin(CORE)].copy()
feat = build_features(enriched)
feat_core = feat[feat["airport"].isin(CORE)].copy()

train, val, test = temporal_train_val_test_split(feat_core)
lag_cols = [c for c in train.columns if "lag" in c or "rolling" in c]
train_clean = train.dropna(subset=lag_cols)
val_clean = val.dropna(subset=lag_cols)
test_clean = test.dropna(subset=lag_cols)

feature_cols = [c for c in FEATURE_COLS if c in train_clean.columns]
train_clean = train_clean.copy()
val_clean = val_clean.copy()
test_clean = test_clean.copy()
train_clean["airport_cat"] = train_clean["airport"].astype("category")
val_clean["airport_cat"] = val_clean["airport"].astype("category")
test_clean["airport_cat"] = test_clean["airport"].astype("category")
all_features = ["airport_cat"] + feature_cols

X_train = train_clean[all_features]
y_train = train_clean["pax"]
X_val = val_clean[all_features]
y_val = val_clean["pax"]
X_test = test_clean[all_features]
y_test = test_clean["pax"]

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
print(f"Features: {len(all_features)}")

# Recursive evaluation windows (honest exog, matches serving regime).
enriched_trainval = enriched_core[enriched_core["date"] <= VAL_END].copy()  # for val (2024)


def recursive_mape(model, enriched_window, origin, lo, hi):
    """Honest recursive MAPE on (lo, hi] using a model trained up to `origin`."""
    fc = recursive_forecast_global(model, all_features, enriched_window, origin, CORE)
    fc = fc.dropna(subset=["pax_actual", "pax_pred"])
    fc = fc[(fc["date"] > pd.Timestamp(lo)) & (fc["date"] <= pd.Timestamp(hi))]
    yt, yp = fc["pax_actual"].values, fc["pax_pred"].values
    mask = yt > 0
    if mask.sum() == 0:
        return float("inf")
    return float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100)


def objective(trial):
    params = {
        "objective": "regression",
        "metric": "mae",
        "verbose": -1,
        "n_jobs": -1,
        "random_state": 42,
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
    }

    # Train on train (<= TRAIN_END), score by recursive forecast over val (2024).
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train, categorical_feature=["airport_cat"])
    return recursive_mape(model, enriched_trainval, TRAIN_END, TRAIN_END, VAL_END)


optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="minimize", study_name="lgbm_tuning")
print("Running 100 Optuna trials...")
study.optimize(objective, n_trials=100, show_progress_bar=True)

print(f"\nBest recursive val MAPE: {study.best_value:.2f}%")
print(f"Best params: {study.best_params}")

# Retrain with best params on train+val, evaluate on test via honest recursive forecast.
best = study.best_params.copy()
best.update({"objective": "regression", "metric": "mae", "verbose": -1,
             "n_jobs": -1, "random_state": 42})

X_trainval = pd.concat([X_train, X_val])
y_trainval = pd.concat([y_train, y_val])

final_model = lgb.LGBMRegressor(**best)
final_model.fit(X_trainval, y_trainval, categorical_feature=["airport_cat"])

# Recursive forecast over test (2025+), trained up to VAL_END.
fc_test = recursive_forecast_global(final_model, all_features, enriched_core, VAL_END, CORE)
fc_test = fc_test.dropna(subset=["pax_actual", "pax_pred"])
yt, yp = fc_test["pax_actual"].values, fc_test["pax_pred"].values
mask = yt > 0
test_mape = float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100)
test_mae = mean_absolute_error(yt, yp)

print(f"\nTest MAPE (tuned, recursive): {test_mape:.2f}%")
print(f"Test MAE (tuned, recursive): {test_mae:,.0f}")

# Per airport
for ap in CORE:
    sub = fc_test[fc_test["airport"] == ap]
    if sub.empty:
        continue
    at, ap_p = sub["pax_actual"].values, sub["pax_pred"].values
    m = at > 0
    if m.sum() == 0:
        continue
    ap_mape = float(np.mean(np.abs((at[m] - ap_p[m]) / at[m])) * 100)
    print(f"  {SHORT[ap]}: {ap_mape:.1f}%")

# Save best params
import json
params_path = Path(__file__).resolve().parent.parent / "reports" / "best_params.json"
with open(params_path, "w") as f:
    json.dump({"best_params": study.best_params, "val_mape": study.best_value,
               "test_mape": test_mape}, f, indent=2)
print(f"\nBest params saved to {params_path}")
