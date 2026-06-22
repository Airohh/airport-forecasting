"""Optuna hyperparameter tuning for LightGBM Global model."""

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from airport_forecast.features import build_features, temporal_train_val_test_split
from airport_forecast.models import FEATURE_COLS

CORE = ["FR_LFLL", "FR_LFRS", "HU_LHBP", "PT_LPPT", "PT_LPPR", "RS_LYBE"]
SHORT = {"FR_LFLL": "Lyon", "FR_LFRS": "Nantes", "HU_LHBP": "Budapest",
         "PT_LPPT": "Lisbon", "PT_LPPR": "Porto", "RS_LYBE": "Belgrade"}

# Load
enriched = pd.read_parquet(
    Path(__file__).resolve().parent.parent / "data" / "processed" / "pax_enriched.parquet"
)
enriched["date"] = pd.to_datetime(enriched["date"])
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

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        categorical_feature=["airport_cat"],
    )

    pred_val = np.maximum(model.predict(X_val), 0)
    mask = y_val > 0
    mape = float(np.mean(np.abs((y_val[mask] - pred_val[mask]) / y_val[mask])) * 100)
    return mape


optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="minimize", study_name="lgbm_tuning")
print("Running 100 Optuna trials...")
study.optimize(objective, n_trials=100, show_progress_bar=True)

print(f"\nBest MAPE (val): {study.best_value:.2f}%")
print(f"Best params: {study.best_params}")

# Retrain with best params on train+val, evaluate on test
best = study.best_params.copy()
best.update({"objective": "regression", "metric": "mae", "verbose": -1,
             "n_jobs": -1, "random_state": 42})

X_trainval = pd.concat([X_train, X_val])
y_trainval = pd.concat([y_train, y_val])

final_model = lgb.LGBMRegressor(**best)
final_model.fit(X_trainval, y_trainval, categorical_feature=["airport_cat"])
pred_test = np.maximum(final_model.predict(X_test), 0)

mask = y_test > 0
test_mape = float(np.mean(np.abs((y_test[mask] - pred_test[mask]) / y_test[mask])) * 100)
test_mae = mean_absolute_error(y_test, pred_test)

print(f"\nTest MAPE (tuned): {test_mape:.2f}%")
print(f"Test MAE (tuned): {test_mae:,.0f}")

# Per airport
for ap in CORE:
    ap_mask = test_clean["airport"] == ap
    if ap_mask.sum() == 0:
        continue
    ap_true = y_test[ap_mask]
    ap_pred = pred_test[ap_mask.values]
    ap_mape = float(np.mean(np.abs((ap_true - ap_pred) / ap_true)) * 100)
    print(f"  {SHORT[ap]}: {ap_mape:.1f}%")

# Save best params
import json
params_path = Path(__file__).resolve().parent.parent / "reports" / "best_params.json"
with open(params_path, "w") as f:
    json.dump({"best_params": study.best_params, "val_mape": study.best_value,
               "test_mape": test_mape}, f, indent=2)
print(f"\nBest params saved to {params_path}")
