"""Model training and evaluation for airport PAX forecasting."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

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
    "n_flights_lag1", "pax_per_flight_lag1",
]


def _load_best_params() -> dict:
    """Load Optuna-tuned hyperparameters from reports/best_params.json if present."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent.parent / "reports" / "best_params.json"
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f).get("best_params", {})
    except (json.JSONDecodeError, OSError):
        return {}


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
    # Use Optuna-tuned params (honest recursive objective) when available, so
    # eval / serve / tune all share one configuration.
    params.update(_load_best_params())

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


# ─────────────────────────────────────────────
# LightGBM GROWTH-RATIO (Budapest / high-growth fix)
# ─────────────────────────────────────────────
# A tree model cannot predict above the PAX levels it saw in training, so on a
# fast-growing airport (Budapest, +~20%/yr) the level model plateaus and badly
# under-forecasts forward. Predicting the year-over-year RATIO instead keeps the
# target stationary (~1.0–1.3) and IN the training range, then we rebuild the
# level: pax_t = ratio_t × pax_lag_12. The ratio extrapolates the growth a level
# tree cannot.
GROWTH_RATIO_CLIP = (0.5, 2.5)  # guard against runaway recursive ratios


def train_lightgbm_growth(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None = None,
    feature_cols: list[str] | None = None,
):
    """Global LightGBM whose target is the YoY ratio pax / pax_lag_12 (not the
    level). Same features and tuned params as the level model — only the target
    changes. Returns (model, feature_cols)."""
    import lightgbm as lgb

    train_df = train_df.copy()
    # Target = YoY ratio; need a valid, non-zero same-month-last-year anchor
    train_df = train_df[train_df["pax_lag_12"].fillna(0) > 0]
    y_ratio = (train_df["pax"] / train_df["pax_lag_12"]).clip(*GROWTH_RATIO_CLIP)

    if feature_cols is None:
        feature_cols = [c for c in FEATURE_COLS if c in train_df.columns]

    cat_features = []
    if "airport" in train_df.columns:
        train_df["airport_cat"] = train_df["airport"].astype("category")
        feature_cols = ["airport_cat"] + feature_cols
        cat_features = ["airport_cat"]
        if val_df is not None:
            val_df = val_df.copy()
            val_df["airport_cat"] = val_df["airport"].astype("category")

    X_train = train_df[feature_cols]

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
    params.update(_load_best_params())

    model = lgb.LGBMRegressor(**params)
    if val_df is not None and len(val_df) > 0:
        val_df = val_df[val_df["pax_lag_12"].fillna(0) > 0]
        y_val = (val_df["pax"] / val_df["pax_lag_12"]).clip(*GROWTH_RATIO_CLIP)
        model.fit(
            X_train, y_ratio,
            eval_set=[(val_df[feature_cols], y_val)],
            callbacks=[lgb.log_evaluation(0), lgb.early_stopping(50, verbose=False)],
            categorical_feature=cat_features if cat_features else "auto",
        )
    else:
        model.fit(X_train, y_ratio,
                  categorical_feature=cat_features if cat_features else "auto")

    return model, feature_cols


def predict_growth(model, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Predict YoY ratio then rebuild the PAX level via pax_lag_12. Rows whose
    pax_lag_12 is missing fall back to NaN (caller handles)."""
    df_pred = df.copy()
    if "airport_cat" in feature_cols and "airport_cat" not in df_pred.columns:
        df_pred["airport_cat"] = df_pred["airport"].astype("category")
    ratio = np.clip(model.predict(df_pred[feature_cols]), *GROWTH_RATIO_CLIP)
    anchor = df_pred["pax_lag_12"].to_numpy(dtype=float)
    level = ratio * anchor
    return np.maximum(level, 0)


# Exogenous columns and how they are assumed-known at forecast time.
# seasonal naive (same month last year) — calendar + airline supply (published schedules)
SEASONAL_EXOG_COLS = ["n_holidays", "is_school_vacation", "n_flights", "pax_per_flight"]
# carried forward (last known state) — macro levels + ongoing event flags
CARRY_EXOG_COLS = [
    "unemployment_rate", "oil_price_usd", "exchange_rate", "gdp",
    "event_covid", "event_ukraine_war", "event_major_sport", "event_conference",
]


def forecast_flight_counts(
    df: pd.DataFrame,
    origin: pd.Timestamp,
    airports: list[str],
    future_dates: list,
) -> dict:
    """Stage-1 of the two-stage forecast: predict future `n_flights` per airport.

    `n_flights` is the strongest PAX predictor (corr 0.86–0.98) but unknown at
    forecast time. Instead of a flat seasonal-naive (same month N-1), fit a SARIMA
    on each airport's flight-count history up to `origin` and forecast forward —
    capturing trend + recovery the naive proxy misses. Falls back to seasonal-naive
    when history is too short for a seasonal model or SARIMA fails to converge.

    Honest: only flight history up to `origin` is used. Returns
    dict[(airport, Timestamp)] -> forecast n_flights.
    """
    origin = pd.Timestamp(origin)
    fdates = sorted(pd.Timestamp(d) for d in future_dates if pd.Timestamp(d) > origin)
    out: dict = {}
    if not fdates:
        return out

    for ap in airports:
        sub = df[df["airport"] == ap].set_index("date").sort_index()
        if "n_flights" not in sub.columns:
            continue
        hist = sub.loc[sub.index <= origin, "n_flights"].dropna()

        def _seasonal_naive(d: pd.Timestamp) -> float:
            v = sub["n_flights"].get(d - pd.DateOffset(months=12), np.nan)
            if pd.isna(v):
                v = hist.iloc[-1] if len(hist) else np.nan
            return float(v) if pd.notna(v) else np.nan

        if len(hist) < 36:
            for d in fdates:
                out[(ap, d)] = _seasonal_naive(d)
            continue
        try:
            preds = train_sarima(hist, len(fdates))
            for d, p in zip(fdates, preds):
                out[(ap, d)] = float(p) if pd.notna(p) else _seasonal_naive(d)
        except Exception:
            for d in fdates:
                out[(ap, d)] = _seasonal_naive(d)
    return out


def assume_future_exog(
    df: pd.DataFrame,
    origin: pd.Timestamp,
    airports: list[str],
    flight_forecast: dict | None = None,
    origin_by_airport: dict | None = None,
) -> pd.DataFrame:
    """Overwrite exogenous columns for dates AFTER `origin` with values that would
    actually be available at forecast time — so the model never peeks at real
    future macro/flights. seasonal naive for SEASONAL_EXOG_COLS, last-known carry
    for CARRY_EXOG_COLS. This is what makes recursive forecasts genuinely honest.

    If `flight_forecast` (from forecast_flight_counts) is given, future `n_flights`
    uses those stage-1 model predictions instead of seasonal-naive (two-stage).

    `origin_by_airport` lets each airport keep its own real recent actuals: only
    dates after THAT airport's last actual are treated as future (live forecast,
    where airports have unequal data end-dates). When None, the single `origin`
    applies to all airports (evaluation / backtest path).
    """
    out = df.copy()
    origin = pd.Timestamp(origin)
    for ap in airports:
        ap_origin = (
            pd.Timestamp(origin_by_airport[ap])
            if origin_by_airport and ap in origin_by_airport
            else origin
        )
        m = out["airport"] == ap
        sub = out[m].set_index("date").sort_index()
        fut_idx = sub.index[sub.index > ap_origin]
        if len(fut_idx) == 0:
            continue
        hist = sub[sub.index <= ap_origin]
        for c in CARRY_EXOG_COLS:
            if c in sub.columns:
                last = hist[c].dropna()
                val = last.iloc[-1] if len(last) else np.nan
                out.loc[m & (out["date"] > ap_origin), c] = val
        for c in SEASONAL_EXOG_COLS:
            if c in sub.columns:
                for d in fut_idx:
                    if (
                        c == "n_flights"
                        and flight_forecast is not None
                        and (ap, d) in flight_forecast
                    ):
                        v = flight_forecast[(ap, d)]
                    else:
                        v = sub[c].get(d - pd.DateOffset(months=12), np.nan)
                    if pd.isna(v):
                        last = hist[c].dropna()
                        v = last.iloc[-1] if len(last) else np.nan
                    out.loc[m & (out["date"] == d), c] = v
    return out


def recursive_forecast_global(
    model,
    feature_cols: list[str],
    enriched_df: pd.DataFrame,
    origin_date: str,
    airports: list[str],
    honest_exog: bool = True,
    flight_forecast: bool = False,
    target_mode: str = "level",
    origin_by_airport: dict | None = None,
) -> pd.DataFrame:
    """Honest multi-step forecast: predict month by month, feeding each prediction
    back as the lag/rolling input for the next month (no peeking at future actuals).

    With honest_exog=True (default), exogenous columns (macro, airline supply,
    event flags) for future dates are replaced by assume_future_exog so the
    forecast uses only information available at the origin — not real future
    macro/flight values. Set False only for diagnostics.

    With flight_forecast=True, future n_flights uses a stage-1 SARIMA forecast
    (two-stage) instead of seasonal-naive.

    target_mode="level" uses the standard level model; "growth" uses a model whose
    target is the YoY ratio and rebuilds the level via pax_lag_12 (extrapolates
    growth a level tree cannot — fixes Budapest-style under-forecasting).

    Returns a DataFrame with columns: airport, date, pax_actual, pax_pred.
    """
    from airport_forecast.features import build_features

    origin = pd.Timestamp(origin_date)
    work = enriched_df[enriched_df["airport"].isin(airports)].copy()
    work["date"] = pd.to_datetime(work["date"])
    work = work.sort_values(["airport", "date"]).reset_index(drop=True)

    # Keep ground truth, then blank out future PAX so features are recomputed.
    # With origin_by_airport (live forecast), "future" is per-airport: each airport
    # keeps its own real recent actuals and only its genuinely-future months are
    # blanked/predicted. Without it (eval), a single global origin applies.
    work["pax_actual"] = work["pax"]
    if origin_by_airport:
        thr = work["airport"].map(
            {a: pd.Timestamp(d) for a, d in origin_by_airport.items()}
        ).fillna(origin)
        future_mask = work["date"] > thr
    else:
        future_mask = work["date"] > origin
    work.loc[future_mask, "pax"] = np.nan

    # Replace future exogenous values with assumed-known proxies (no leakage)
    if honest_exog:
        ff = None
        if flight_forecast:
            fut_dates = sorted(work.loc[future_mask, "date"].unique())
            ff = forecast_flight_counts(work, origin, airports, fut_dates)
        work = assume_future_exog(
            work, origin, airports, flight_forecast=ff,
            origin_by_airport=origin_by_airport,
        )

    future_dates = sorted(work.loc[future_mask, "date"].unique())

    for d in future_dates:
        feat = build_features(work)
        rows = feat[feat["date"] == d]
        if rows.empty:
            continue
        if target_mode == "growth":
            preds = predict_growth(model, rows, feature_cols)
        else:
            preds = predict_lightgbm(model, rows, feature_cols)
        # Write predictions back into the working PAX so they feed next month's lags.
        # Never overwrite an airport's real actual: at a shared future date some
        # airports still have ground truth (later data end-date) — skip those.
        for ap, p in zip(rows["airport"].values, preds):
            if np.isnan(p):
                continue
            if origin_by_airport is not None:
                ap_o = pd.Timestamp(origin_by_airport.get(ap, origin))
                if d <= ap_o:
                    continue
            work.loc[(work["date"] == d) & (work["airport"] == ap), "pax"] = float(p)

    out = work.loc[future_mask, ["airport", "date", "pax_actual"]].copy()
    out["pax_pred"] = work.loc[future_mask, "pax"].values
    return out


def make_future_enriched(
    enriched_df: pd.DataFrame,
    airports: list[str],
    horizon: int,
) -> tuple[pd.DataFrame, pd.Timestamp, dict]:
    """Append empty future months so the model can forecast PAST the end of history
    (not backcast it). Future PAX is left NaN and future exogenous values are filled
    later by assume_future_exog inside the recursive forecast — keeping a single
    source of truth for the exog assumptions.

    Airports have UNEQUAL data end-dates (e.g. Budapest runs months ahead of Lyon).
    Each airport gets `horizon` genuine future months PAST ITS OWN last actual — we
    never duplicate or overwrite a month it already has. The shared grid spans
    min(last)+1 .. max(last)+horizon so every airport is present at every future
    month (network features need all airports each month); the dashboard trims each
    airport to its first `horizon` rows.

    Returns (enriched_df + skeleton future rows with pax=NaN, global origin,
    origin_by_airport mapping each airport to its own last-actual date).
    """
    df = enriched_df[enriched_df["airport"].isin(airports)].copy()
    df["date"] = pd.to_datetime(df["date"])

    last = df.groupby("airport")["date"].max()
    origin = last.min()
    origin_by_airport = {ap: last[ap] for ap in airports if ap in last.index}

    grid_end = last.max() + pd.DateOffset(months=horizon)
    all_future = pd.date_range(origin + pd.DateOffset(months=1), grid_end, freq="MS")

    new_rows = []
    for ap in airports:
        sub = df[df["airport"] == ap]
        if sub.empty:
            continue
        ap_last = sub["date"].max()
        name = sub["airport_name"].iloc[-1] if "airport_name" in sub.columns else None
        for d in all_future:
            if d <= ap_last:  # real actual already exists — never duplicate it
                continue
            row = {"airport": ap, "date": d, "pax": np.nan}
            if name is not None:
                row["airport_name"] = name
            new_rows.append(row)

    if not new_rows:
        return df, origin, origin_by_airport
    out = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    out = out.sort_values(["airport", "date"]).reset_index(drop=True)
    return out, origin, origin_by_airport


def forecast_future_global(
    model,
    feature_cols: list[str],
    enriched_df: pd.DataFrame,
    airports: list[str],
    horizon: int,
    flight_forecast: bool = False,
    target_mode: str = "level",
) -> pd.DataFrame:
    """Genuine out-of-sample forecast: extend history with future exog rows, then
    recurse. Unlike a backcast, every returned date is strictly after the last
    available actual. target_mode="growth" uses the YoY-ratio model (Budapest
    fix). Returns columns: airport, date, pax_pred."""
    future_df, origin, origin_by_airport = make_future_enriched(
        enriched_df, airports, horizon
    )
    fc = recursive_forecast_global(
        model, feature_cols, future_df,
        origin_date=origin.strftime("%Y-%m-%d"), airports=airports,
        flight_forecast=flight_forecast, target_mode=target_mode,
        origin_by_airport=origin_by_airport,
    )
    return fc[["airport", "date", "pax_pred"]].reset_index(drop=True)


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
