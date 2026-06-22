"""Tests for models and API."""

import sys
import numpy as np
import pandas as pd
import pytest

from airport_forecast.models import ForecastResult, ensemble_predictions, results_to_dataframe


def test_forecast_result_metrics():
    r = ForecastResult(
        model_name="test",
        airport="FR_LFLL",
        horizon=3,
        y_true=np.array([100, 200, 300]),
        y_pred=np.array([110, 190, 310]),
        dates=np.array(["2025-01", "2025-02", "2025-03"]),
    )
    assert r.mae > 0
    assert r.rmse > 0
    assert 0 < r.mape < 100


def test_forecast_result_perfect():
    vals = np.array([100, 200, 300])
    r = ForecastResult(
        model_name="perfect",
        airport="FR_LFLL",
        horizon=3,
        y_true=vals,
        y_pred=vals.copy(),
        dates=np.array(["2025-01", "2025-02", "2025-03"]),
    )
    assert r.mae == 0
    assert r.mape == 0


def test_ensemble_predictions():
    dates = np.array(["2025-01", "2025-02"])
    y_true = np.array([100, 200])
    r1 = ForecastResult("A", "FR_LFLL", 2, y_true, np.array([90, 210]), dates)
    r2 = ForecastResult("B", "FR_LFLL", 2, y_true, np.array([110, 190]), dates)

    ensemble = ensemble_predictions(
        {"A": [r1], "B": [r2]},
        weights={"A": 0.5, "B": 0.5},
    )
    assert len(ensemble) == 1
    assert ensemble[0].model_name == "Ensemble"
    np.testing.assert_array_almost_equal(ensemble[0].y_pred, [100, 200])
    assert ensemble[0].mape == 0


def test_results_to_dataframe():
    r = ForecastResult("test", "FR_LFLL", 3, np.array([100]), np.array([110]),
                       np.array(["2025-01"]))
    df = results_to_dataframe([r])
    assert len(df) == 1
    assert "model" in df.columns
    assert "mape" in df.columns


def test_results_to_dataframe_empty():
    df = results_to_dataframe([])
    assert len(df) == 0


class TestAPI:
    @pytest.fixture(autouse=True)
    def setup(self):
        from airport_forecast.api import app
        from fastapi.testclient import TestClient
        self.client = TestClient(app)

    def test_root(self):
        r = self.client.get("/")
        assert r.status_code == 200
        assert "service" in r.json()

    def test_airports(self):
        r = self.client.get("/airports")
        assert r.status_code == 200
        airports = r.json()
        assert len(airports) == 6
        names = [a["name"] for a in airports]
        assert "Lyon" in names

    def test_predict_lightgbm(self):
        r = self.client.post("/predict", json={
            "airport": "FR_LFLL", "horizon": 3, "model": "lightgbm"
        })
        assert r.status_code == 200
        data = r.json()
        assert data["airport"] == "FR_LFLL"
        assert len(data["predictions"]) == 3
        for p in data["predictions"]:
            assert p["pax_predicted"] > 0

    def test_predict_sarima(self):
        r = self.client.post("/predict", json={
            "airport": "HU_LHBP", "horizon": 3, "model": "sarima"
        })
        assert r.status_code == 200

    def test_predict_invalid_airport(self):
        r = self.client.post("/predict", json={
            "airport": "FAKE", "horizon": 3, "model": "lightgbm"
        })
        assert r.status_code == 404

    def test_predict_invalid_model(self):
        r = self.client.post("/predict", json={
            "airport": "FR_LFLL", "horizon": 3, "model": "fake_model"
        })
        assert r.status_code == 400

    def test_metrics(self):
        r = self.client.get("/models/FR_LFLL/metrics")
        assert r.status_code == 200
        assert len(r.json()["metrics"]) > 0

    def test_metrics_invalid_airport(self):
        r = self.client.get("/models/FAKE/metrics")
        assert r.status_code == 404
