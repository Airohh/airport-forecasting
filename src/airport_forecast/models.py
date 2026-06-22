"""Model training and evaluation for airport PAX forecasting."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass
class ForecastResult:
    model_name: str
    airport: str
    horizon: int
    y_true: np.ndarray
    y_pred: np.ndarray
    dates: np.ndarray
    y_naive: np.ndarray | None = None
    mae: float = 0.0
    rmse: float = 0.0
    mape: float = 0.0
    bias: float = 0.0
    mase: float = 0.0

    def __post_init__(self):
        self.mae = mean_absolute_error(self.y_true, self.y_pred)
        self.rmse = float(np.sqrt(mean_squared_error(self.y_true, self.y_pred)))
        self.bias = float(np.mean(self.y_pred - self.y_true))
        mask = self.y_true > 0
        if mask.sum() > 0:
            self.mape = float(np.mean(np.abs(
                (self.y_true[mask] - self.y_pred[mask]) / self.y_true[mask]
            )) * 100)
        if self.y_naive is not None:
            naive_mae = float(np.mean(np.abs(self.y_true - self.y_naive)))
            self.mase = self.mae / naive_mae if naive_mae > 0 else float("inf")


# ─────────────────────────────────────────────
# SARIMA (per airport)
# ─────────────────────────────────────────────
def train_sarima(
    train_series: pd.Series,
    n_forecast: int,
    order: tuple = (1, 1, 1),
    seasonal_order: tuple = (1, 1, 1, 12),
) -> np.ndarray:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            train_series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = model.fit(disp=False, maxiter=200)
    forecast = fit.forecast(steps=n_forecast)
    return np.maximum(forecast.values, 0)


def evaluate_sarima(
    df: pd.DataFrame,
    airport: str,
    train_end: str = "2023-12",
    val_end: str = "2024-12",
) -> list[ForecastResult]:
    sub = df[df["airport"] == airport].sort_values("date")
    train = sub[sub["date"] <= train_end].set_index("date")["pax"]
    val = sub[(sub["date"] > train_end) & (sub["date"] <= val_end)]
    test = sub[sub["date"] > val_end]

    results = []

    # Forecast on validation
    if len(val) > 0:
        pred_val = train_sarima(train, len(val))
        results.append(ForecastResult(
            model_name="SARIMA",
            airport=airport,
            horizon=len(val),
            y_true=val["pax"].values,
            y_pred=pred_val,
            dates=val["date"].values,
        ))

    # Forecast on test (train on train+val)
    if len(test) > 0:
        train_full = sub[sub["date"] <= val_end].set_index("date")["pax"]
        pred_test = train_sarima(train_full, len(test))
        results.append(ForecastResult(
            model_name="SARIMA",
            airport=airport,
            horizon=len(test),
            y_true=test["pax"].values,
            y_pred=pred_test,
            dates=test["date"].values,
        ))

    return results


# ─────────────────────────────────────────────
# LightGBM GLOBAL
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "month_sin", "month_cos", "is_summer", "quarter",
    "n_holidays", "is_school_vacation",
    "is_covid", "event_covid", "event_ukraine_war",
    "event_major_sport", "event_conference",
    "pax_lag_1", "pax_lag_2", "pax_lag_3", "pax_lag_6", "pax_lag_12",
    "pax_rolling_mean_3", "pax_rolling_std_3",
    "pax_rolling_mean_6", "pax_rolling_std_6",
    "pax_rolling_mean_12", "pax_rolling_std_12",
    "pax_yoy_growth",
    "unemployment_rate", "gdp", "oil_price_usd", "exchange_rate",
    "network_total_pax", "country_total_pax", "country_market_share", "network_rank",
    "n_flights", "pax_per_flight",
]


def train_lightgbm_global(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None = None,
    feature_cols: list[str] | None = None,
):
    import lightgbm as lgb

    if feature_cols is None:
        feature_cols = [c for c in FEATURE_COLS if c in train_df.columns]

    # Airport as categorical
    cat_features = []
    if "airport" in train_df.columns:
        train_df = train_df.copy()
        train_df["airport_cat"] = train_df["airport"].astype("category")
        feature_cols = ["airport_cat"] + feature_cols
        cat_features = ["airport_cat"]
        if val_df is not None:
            val_df = val_df.copy()
            val_df["airport_cat"] = val_df["airport"].astype("category")

    X_train = train_df[feature_cols]
    y_train = train_df["pax"]

    params = {
        "objective": "regression",
        "metric": "mae",
        "n_estimators": 500,
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 10,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
    }

    callbacks = [lgb.log_evaluation(0)]
    if val_df is not None and len(val_df) > 0:
        X_val = val_df[feature_cols]
        y_val = val_df["pax"]
        callbacks.append(lgb.early_stopping(50, verbose=False))
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks,
            categorical_feature=cat_features if cat_features else "auto",
        )
    else:
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train, categorical_feature=cat_features if cat_features else "auto")

    return model, feature_cols


def predict_lightgbm(model, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    df_pred = df.copy()
    if "airport_cat" in feature_cols and "airport_cat" not in df_pred.columns:
        df_pred["airport_cat"] = df_pred["airport"].astype("category")
    X = df_pred[feature_cols]
    preds = model.predict(X)
    return np.maximum(preds, 0)


def recursive_forecast_global(
    model,
    feature_cols: list[str],
    enriched_df: pd.DataFrame,
    origin_date: str,
    airports: list[str],
) -> pd.DataFrame:
    """Honest multi-step forecast: predict month by month, feeding each prediction
    back as the lag/rolling input for the next month (no peeking at future actuals).

    enriched_df must contain exogenous columns (macro, calendar, events) for the
    future dates — only PAX-derived features (lags, rolling, yoy, network) are
    recomputed recursively.

    Returns a DataFrame with columns: airport, date, pax_actual, pax_pred.
    """
    from airport_forecast.features import build_features

    origin = pd.Timestamp(origin_date)
    work = enriched_df[enriched_df["airport"].isin(airports)].copy()
    work["date"] = pd.to_datetime(work["date"])
    work = work.sort_values(["airport", "date"]).reset_index(drop=True)

    # Keep ground truth, then blank out future PAX so features are recomputed
    work["pax_actual"] = work["pax"]
    future_mask = work["date"] > origin
    work.loc[future_mask, "pax"] = np.nan

    future_dates = sorted(work.loc[future_mask, "date"].unique())

    for d in future_dates:
        feat = build_features(work)
        rows = feat[feat["date"] == d]
        if rows.empty:
            continue
        preds = predict_lightgbm(model, rows, feature_cols)
        # Write predictions back into the working PAX so they feed next month's lags
        for ap, p in zip(rows["airport"].values, preds):
            work.loc[(work["date"] == d) & (work["airport"] == ap), "pax"] = float(p)

    out = work.loc[future_mask, ["airport", "date", "pax_actual"]].copy()
    out["pax_pred"] = work.loc[future_mask, "pax"].values
    return out


def evaluate_lightgbm_recursive(
    enriched_df: pd.DataFrame,
    val_end: str = "2024-12",
    core_airports: list[str] | None = None,
) -> tuple[object, list[ForecastResult]]:
    """Train on data up to val_end, then recursively forecast the test horizon."""
    from airport_forecast.features import build_features, temporal_train_val_test_split

    airports = core_airports or sorted(enriched_df["airport"].unique().tolist())
    feat = build_features(enriched_df)
    feat_core = feat[feat["airport"].isin(airports)].copy()

    # Train on everything up to val_end (train + val combined)
    train, val, test = temporal_train_val_test_split(feat_core, val_end, val_end)
    lag_cols = [c for c in feat_core.columns if "lag" in c or "rolling" in c]
    trainval = pd.concat([train, val]).dropna(subset=lag_cols)

    model, feature_cols = train_lightgbm_global(trainval, None)

    # Recursive forecast over the test horizon
    fc = recursive_forecast_global(
        model, feature_cols, enriched_df, origin_date=val_end, airports=airports
    )

    results = []
    for ap in airports:
        sub = fc[fc["airport"] == ap].dropna(subset=["pax_actual", "pax_pred"])
        if sub.empty:
            continue
        results.append(ForecastResult(
            model_name="LightGBM_Recursive",
            airport=ap,
            horizon=len(sub),
            y_true=sub["pax_actual"].values.astype(float),
            y_pred=sub["pax_pred"].values.astype(float),
            dates=sub["date"].values,
        ))

    return model, results


def evaluate_lightgbm_global(
    feat: pd.DataFrame,
    train_end: str = "2023-12",
    val_end: str = "2024-12",
    core_airports: list[str] | None = None,
) -> tuple[object, list[ForecastResult]]:
    from airport_forecast.features import temporal_train_val_test_split

    # Filter to core airports
    if core_airports:
        feat = feat[feat["airport"].isin(core_airports)].copy()

    train, val, test = temporal_train_val_test_split(feat, train_end, val_end)

    # Drop rows with NaN in key lag features
    lag_cols = [c for c in train.columns if "lag" in c or "rolling" in c]
    train_clean = train.dropna(subset=lag_cols)
    val_clean = val.dropna(subset=lag_cols)
    test_clean = test.dropna(subset=lag_cols)

    print(f"  Train: {len(train_clean)}, Val: {len(val_clean)}, Test: {len(test_clean)}")

    model, feature_cols = train_lightgbm_global(train_clean, val_clean)

    results = []
    airports = core_airports or feat["airport"].unique().tolist()

    for ap in airports:
        # Validation
        val_ap = val_clean[val_clean["airport"] == ap]
        if len(val_ap) > 0:
            pred_val = predict_lightgbm(model, val_ap, feature_cols)
            results.append(ForecastResult(
                model_name="LightGBM_Global",
                airport=ap,
                horizon=len(val_ap),
                y_true=val_ap["pax"].values,
                y_pred=pred_val,
                dates=val_ap["date"].values,
            ))

        # Test
        test_ap = test_clean[test_clean["airport"] == ap]
        if len(test_ap) > 0:
            pred_test = predict_lightgbm(model, test_ap, feature_cols)
            results.append(ForecastResult(
                model_name="LightGBM_Global",
                airport=ap,
                horizon=len(test_ap),
                y_true=test_ap["pax"].values,
                y_pred=pred_test,
                dates=test_ap["date"].values,
            ))

    return model, results


# ─────────────────────────────────────────────
# LightGBM LOCAL (per airport, for comparison)
# ─────────────────────────────────────────────
def evaluate_lightgbm_local(
    feat: pd.DataFrame,
    airport: str,
    train_end: str = "2023-12",
    val_end: str = "2024-12",
) -> list[ForecastResult]:
    from airport_forecast.features import temporal_train_val_test_split

    sub = feat[feat["airport"] == airport].copy()
    train, val, test = temporal_train_val_test_split(sub, train_end, val_end)

    lag_cols = [c for c in train.columns if "lag" in c or "rolling" in c]
    train_clean = train.dropna(subset=lag_cols)
    val_clean = val.dropna(subset=lag_cols)
    test_clean = test.dropna(subset=lag_cols)

    if len(train_clean) < 20:
        return []

    # No airport feature for local model
    local_features = [c for c in FEATURE_COLS if c in train_clean.columns]
    model, fcols = train_lightgbm_global(train_clean, val_clean, local_features)

    results = []
    if len(val_clean) > 0:
        pred = predict_lightgbm(model, val_clean, fcols)
        results.append(ForecastResult(
            model_name="LightGBM_Local",
            airport=airport,
            horizon=len(val_clean),
            y_true=val_clean["pax"].values,
            y_pred=pred,
            dates=val_clean["date"].values,
        ))
    if len(test_clean) > 0:
        pred = predict_lightgbm(model, test_clean, fcols)
        results.append(ForecastResult(
            model_name="LightGBM_Local",
            airport=airport,
            horizon=len(test_clean),
            y_true=test_clean["pax"].values,
            y_pred=pred,
            dates=test_clean["date"].values,
        ))

    return results


# ─────────────────────────────────────────────
# Prophet (per airport)
# ─────────────────────────────────────────────
def evaluate_prophet(
    df: pd.DataFrame,
    airport: str,
    train_end: str = "2023-12",
    val_end: str = "2024-12",
) -> list[ForecastResult]:
    from prophet import Prophet

    sub = df[df["airport"] == airport].sort_values("date")
    train = sub[sub["date"] <= train_end][["date", "pax"]].rename(columns={"date": "ds", "pax": "y"})
    val = sub[(sub["date"] > train_end) & (sub["date"] <= val_end)]
    test = sub[sub["date"] > val_end]

    results = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Val
        if len(val) > 0:
            m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            m.fit(train)
            future = m.make_future_dataframe(periods=len(val), freq="MS")
            forecast = m.predict(future)
            pred_val = forecast.tail(len(val))["yhat"].values
            pred_val = np.maximum(pred_val, 0)
            results.append(ForecastResult(
                model_name="Prophet",
                airport=airport,
                horizon=len(val),
                y_true=val["pax"].values,
                y_pred=pred_val,
                dates=val["date"].values,
            ))

        # Test (train on train+val)
        if len(test) > 0:
            train_full = sub[sub["date"] <= val_end][["date", "pax"]].rename(
                columns={"date": "ds", "pax": "y"})
            m2 = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            m2.fit(train_full)
            future2 = m2.make_future_dataframe(periods=len(test), freq="MS")
            forecast2 = m2.predict(future2)
            pred_test = forecast2.tail(len(test))["yhat"].values
            pred_test = np.maximum(pred_test, 0)
            results.append(ForecastResult(
                model_name="Prophet",
                airport=airport,
                horizon=len(test),
                y_true=test["pax"].values,
                y_pred=pred_test,
                dates=test["date"].values,
            ))

    return results


# ─────────────────────────────────────────────
# Results summary
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# Chronos (Amazon foundation model, zero-shot)
# ─────────────────────────────────────────────
def evaluate_chronos(
    df: pd.DataFrame,
    airport: str,
    train_end: str = "2023-12",
    val_end: str = "2024-12",
    model_name: str = "amazon/chronos-t5-small",
) -> list[ForecastResult]:
    import torch
    from chronos import ChronosPipeline

    sub = df[df["airport"] == airport].sort_values("date")
    train = sub[sub["date"] <= train_end]
    val = sub[(sub["date"] > train_end) & (sub["date"] <= val_end)]
    test = sub[sub["date"] > val_end]

    pipeline = ChronosPipeline.from_pretrained(
        model_name,
        device_map="cpu",
        torch_dtype=torch.float32,
    )

    results = []

    def _extract_median(forecast):
        # forecast shape: (batch, num_samples, horizon)
        return np.median(forecast.numpy(), axis=1).squeeze()

    # Validation
    if len(val) > 0:
        context = torch.tensor(train["pax"].values, dtype=torch.float32).unsqueeze(0)
        forecast = pipeline.predict(context, prediction_length=len(val))
        pred_val = _extract_median(forecast)
        pred_val = np.maximum(pred_val, 0)
        results.append(ForecastResult(
            model_name="Chronos",
            airport=airport,
            horizon=len(val),
            y_true=val["pax"].values,
            y_pred=pred_val,
            dates=val["date"].values,
        ))

    # Test (context = train + val)
    if len(test) > 0:
        train_full = sub[sub["date"] <= val_end]
        context = torch.tensor(train_full["pax"].values, dtype=torch.float32).unsqueeze(0)
        forecast = pipeline.predict(context, prediction_length=len(test))
        pred_test = _extract_median(forecast)
        pred_test = np.maximum(pred_test, 0)
        results.append(ForecastResult(
            model_name="Chronos",
            airport=airport,
            horizon=len(test),
            y_true=test["pax"].values,
            y_pred=pred_test,
            dates=test["date"].values,
        ))

    return results


# ─────────────────────────────────────────────
# Ensemble (weighted average)
# ─────────────────────────────────────────────
def ensemble_predictions(
    results_by_model: dict[str, list[ForecastResult]],
    weights: dict[str, float] | None = None,
) -> list[ForecastResult]:
    """Weighted average of multiple model predictions per airport."""
    if weights is None:
        weights = {m: 1.0 / len(results_by_model) for m in results_by_model}

    total_w = sum(weights.values())
    weights = {m: w / total_w for m, w in weights.items()}

    # Group by (airport, horizon)
    from collections import defaultdict
    grouped: dict[tuple, dict[str, ForecastResult]] = defaultdict(dict)
    for model_name, results in results_by_model.items():
        for r in results:
            key = (r.airport, r.horizon)
            grouped[key][model_name] = r

    ensemble_results = []
    for (airport, horizon), model_results in grouped.items():
        # Only ensemble if we have all models
        available = [m for m in weights if m in model_results]
        if not available:
            continue

        ref = model_results[available[0]]
        y_pred_ensemble = np.zeros_like(ref.y_pred, dtype=float)
        w_sum = 0.0
        for m in available:
            w = weights[m]
            y_pred_ensemble += w * model_results[m].y_pred
            w_sum += w

        if w_sum > 0:
            y_pred_ensemble /= w_sum

        ensemble_results.append(ForecastResult(
            model_name="Ensemble",
            airport=airport,
            horizon=horizon,
            y_true=ref.y_true,
            y_pred=np.maximum(y_pred_ensemble, 0),
            dates=ref.dates,
        ))

    return ensemble_results


def results_to_dataframe(results: list[ForecastResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "model": r.model_name,
            "airport": r.airport,
            "horizon": r.horizon,
            "mae": r.mae,
            "rmse": r.rmse,
            "mape": r.mape,
            "bias": r.bias,
            "mase": r.mase,
        })
    return pd.DataFrame(rows)
