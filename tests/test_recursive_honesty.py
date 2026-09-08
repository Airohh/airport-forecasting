"""The claim: future flights / PAX must not leak into the forecast."""

import numpy as np
import pandas as pd

from airport_forecast.models import assume_future_exog, recursive_forecast_global


def _panel(n: int = 24, airport: str = "FR_LFLL") -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=n, freq="MS")
    return pd.DataFrame(
        {
            "airport": airport,
            "date": dates,
            "pax": np.linspace(100_000, 200_000, n),
            "n_flights": 1000.0,
            "pax_per_flight": 150.0,
            "n_holidays": 2,
            "is_school_vacation": 0,
            "unemployment_rate": 7.0,
            "oil_price_usd": 80.0,
            "exchange_rate": 1.0,
            "gdp": 100.0,
            "event_covid": 0,
            "event_ukraine_war": 0,
            "event_major_sport": 0,
            "event_conference": 0,
        }
    )


def test_assume_future_exog_drops_future_flights_and_oil():
    df = _panel()
    origin = df["date"].iloc[11]
    future = df["date"] > origin
    df.loc[future, "n_flights"] = 99_999
    df.loc[future, "oil_price_usd"] = 1.0

    out = assume_future_exog(df, origin, ["FR_LFLL"])
    assert (out.loc[future, "n_flights"] == 1000.0).all()
    assert (out.loc[future, "oil_price_usd"] == 80.0).all()


def test_recursive_forecast_ignores_poisoned_future_pax():
    class LagEcho:
        def predict(self, X):
            return np.asarray(X["pax_lag_1"], dtype=float)

    df = _panel(n=16)
    origin = df["date"].iloc[11]
    poisoned = df.copy()
    poisoned.loc[poisoned["date"] > origin, "pax"] = 1e9

    fc = recursive_forecast_global(
        LagEcho(),
        ["pax_lag_1"],
        poisoned,
        origin_date=origin.strftime("%Y-%m-%d"),
        airports=["FR_LFLL"],
    )
    preds = fc["pax_pred"].dropna().astype(float)
    assert len(preds) >= 2
    assert (preds < 1e6).all()
    last_hist = float(df.loc[df["date"] <= origin, "pax"].iloc[-1])
    assert abs(float(preds.iloc[0]) - last_hist) < 1.0


def test_metrics_endpoint_reads_horizon_results():
    from fastapi.testclient import TestClient

    from airport_forecast.api import app

    r = TestClient(app).get("/models/FR_LFLL/metrics")
    assert r.status_code == 200
    models = {row["model"] for row in r.json()["metrics"]}
    assert "LightGBM_Recursive" in models
    assert "LightGBM_Global" not in models
