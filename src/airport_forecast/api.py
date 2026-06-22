"""FastAPI serving endpoint for airport PAX forecasting."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from airport_forecast.constants import VINCI_AIRPORTS, HORIZONS, SHORT_NAMES, CORE_AIRPORTS
from airport_forecast.features import build_features

app = FastAPI(
    title="Airport PAX Forecasting API",
    description="Multi-model forecasting for VINCI Airports network",
    version="0.1.0",
)

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

SHORT = SHORT_NAMES


class PredictRequest(BaseModel):
    airport: str = Field(..., description="Airport code (e.g. FR_LFLL)")
    horizon: int = Field(default=6, ge=1, le=24, description="Months to forecast")
    model: str = Field(default="lightgbm", description="Model: lightgbm, sarima, prophet, chronos")


class PredictionPoint(BaseModel):
    date: str
    pax_predicted: int
    pax_lower: int | None = None
    pax_upper: int | None = None


class PredictResponse(BaseModel):
    airport: str
    airport_name: str
    model: str
    horizon: int
    predictions: list[PredictionPoint]


class AirportInfo(BaseModel):
    code: str
    name: str
    data_start: str
    data_end: str
    avg_monthly_pax: int


class MetricsResponse(BaseModel):
    airport: str
    metrics: list[dict]


# Cache
_data_cache = {}
_model_cache = {}


def _load_data():
    if "feat" not in _data_cache:
        pax = pd.read_parquet(DATA_DIR / "pax_enriched.parquet")
        pax["date"] = pd.to_datetime(pax["date"])
        feat = build_features(pax)
        _data_cache["feat"] = feat
        _data_cache["raw"] = pax
    return _data_cache["feat"], _data_cache["raw"]


def _load_lgb_model():
    if "lgb" not in _model_cache:
        model_path = MODEL_DIR / "lightgbm_global.pkl"
        if model_path.exists():
            with open(model_path, "rb") as f:
                _model_cache["lgb"] = pickle.load(f)
        else:
            from airport_forecast.models import train_lightgbm_global
            from airport_forecast.features import temporal_train_val_test_split
            feat, _ = _load_data()
            feat_core = feat[feat["airport"].isin(CORE_AIRPORTS)]
            train, val, _ = temporal_train_val_test_split(feat_core)
            lag_cols = [c for c in train.columns if "lag" in c or "rolling" in c]
            train_clean = train.dropna(subset=lag_cols)
            val_clean = val.dropna(subset=lag_cols)
            model, fcols = train_lightgbm_global(train_clean, val_clean)
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            with open(model_path, "wb") as f:
                pickle.dump({"model": model, "feature_cols": fcols}, f)
            _model_cache["lgb"] = {"model": model, "feature_cols": fcols}
    return _model_cache["lgb"]["model"], _model_cache["lgb"]["feature_cols"]


@app.get("/")
def root():
    return {
        "service": "Airport PAX Forecasting API",
        "version": "0.1.0",
        "endpoints": ["/airports", "/predict", "/models/{airport}/metrics"],
    }


@app.get("/airports", response_model=list[AirportInfo])
def list_airports():
    _, raw = _load_data()
    result = []
    for code in CORE_AIRPORTS:
        sub = raw[raw["airport"] == code]
        if sub.empty:
            continue
        result.append(AirportInfo(
            code=code,
            name=SHORT.get(code, code),
            data_start=sub["date"].min().strftime("%Y-%m"),
            data_end=sub["date"].max().strftime("%Y-%m"),
            avg_monthly_pax=int(sub["pax"].mean()),
        ))
    return result


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if req.airport not in CORE_AIRPORTS:
        raise HTTPException(404, f"Airport {req.airport} not found. Use /airports for list.")

    feat, raw = _load_data()

    if req.model == "lightgbm":
        model, fcols = _load_lgb_model()
        _, raw = _load_data()

        sub = raw[raw["airport"] == req.airport].sort_values("date")
        if sub.empty:
            raise HTTPException(400, "No data for this airport")

        last_date = sub["date"].max()
        origin = last_date - pd.DateOffset(months=req.horizon)

        from airport_forecast.models import recursive_forecast_global
        fc = recursive_forecast_global(
            model, fcols, raw,
            origin_date=origin.strftime("%Y-%m-%d"),
            airports=CORE_AIRPORTS,
        )

        fc_ap = fc[fc["airport"] == req.airport].sort_values("date")
        if fc_ap.empty:
            raise HTTPException(400, "Not enough data for recursive forecast")

        predictions = [
            PredictionPoint(
                date=pd.Timestamp(d).strftime("%Y-%m"),
                pax_predicted=max(int(p), 0),
            )
            for d, p in zip(fc_ap["date"].values, fc_ap["pax_pred"].values)
        ]

    elif req.model == "sarima":
        from airport_forecast.models import train_sarima
        sub = raw[raw["airport"] == req.airport].sort_values("date")
        series = sub.set_index("date")["pax"]
        preds = train_sarima(series, req.horizon)
        last_date = series.index.max()
        future_dates = pd.date_range(last_date, periods=req.horizon + 1, freq="MS")[1:]
        predictions = [
            PredictionPoint(date=d.strftime("%Y-%m"), pax_predicted=max(int(p), 0))
            for d, p in zip(future_dates, preds)
        ]

    else:
        raise HTTPException(400, f"Model '{req.model}' not supported. Use: lightgbm, sarima")

    return PredictResponse(
        airport=req.airport,
        airport_name=SHORT.get(req.airport, req.airport),
        model=req.model,
        horizon=req.horizon,
        predictions=predictions,
    )


@app.get("/models/{airport}/metrics", response_model=MetricsResponse)
def model_metrics(airport: str):
    if airport not in CORE_AIRPORTS:
        raise HTTPException(404, f"Airport {airport} not found")

    results_path = REPORTS_DIR / "model_results.csv"
    if not results_path.exists():
        raise HTTPException(404, "No results available. Run training first.")

    df = pd.read_csv(results_path)
    sub = df[df["airport"] == airport]
    metrics = sub[["model", "horizon", "mae", "rmse", "mape"]].to_dict("records")
    return MetricsResponse(airport=airport, metrics=metrics)
