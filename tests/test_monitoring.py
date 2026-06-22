"""Tests for drift monitoring."""

import numpy as np
import pandas as pd

from airport_forecast.monitoring import psi, check_prediction_drift, monitor_drift, should_retrain


def test_psi_identical_distributions():
    data = np.random.randn(1000)
    score = psi(data, data)
    assert score < 0.05


def test_psi_different_distributions():
    ref = np.random.randn(1000)
    prod = np.random.randn(1000) + 3
    score = psi(ref, prod)
    assert score > 0.25


def test_psi_moderate_drift():
    ref = np.random.randn(1000)
    prod = np.random.randn(1000) + 0.5
    score = psi(ref, prod)
    assert 0.0 < score < 1.0


def test_check_prediction_no_degradation():
    y_true = np.array([100, 200, 300])
    y_pred = np.array([105, 195, 310])
    result = check_prediction_drift(y_true, y_pred)
    assert result["status"] == "OK"
    assert result["mape"] < 10


def test_check_prediction_with_threshold():
    y_true = np.array([100, 200, 300])
    y_pred = np.array([200, 400, 600])
    result = check_prediction_drift(y_true, y_pred, mae_threshold=50)
    assert result["status"] == "DEGRADED"


def test_should_retrain_critical_triggers():
    report = pd.DataFrame([
        {"feature": "pax_lag_1", "psi": 0.40, "status": "CRITICAL"},
        {"feature": "gdp", "psi": 0.02, "status": "OK"},
    ])
    decision = should_retrain(report)
    assert decision["retrain"] is True
    assert decision["n_critical"] == 1
    assert "pax_lag_1" in decision["drivers"]


def test_should_retrain_single_warning_does_not_trigger():
    report = pd.DataFrame([
        {"feature": "gdp", "psi": 0.15, "status": "WARNING"},
        {"feature": "oil_price_usd", "psi": 0.03, "status": "OK"},
    ])
    decision = should_retrain(report, n_warning_to_retrain=3)
    assert decision["retrain"] is False


def test_should_retrain_warning_cluster_triggers():
    report = pd.DataFrame([
        {"feature": f"f{i}", "psi": 0.15, "status": "WARNING"} for i in range(3)
    ])
    decision = should_retrain(report, n_warning_to_retrain=3)
    assert decision["retrain"] is True
    assert decision["n_warning"] == 3


def test_should_retrain_empty_report():
    decision = should_retrain(pd.DataFrame())
    assert decision["retrain"] is False


def test_monitor_drift_flags_shifted_feature():
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"x": rng.normal(0, 1, 500), "y": rng.normal(0, 1, 500)})
    prod = pd.DataFrame({"x": rng.normal(0, 1, 100), "y": rng.normal(5, 1, 100)})
    report = monitor_drift(train, prod, ["x", "y"])
    statuses = dict(zip(report["feature"], report["status"]))
    assert statuses["y"] == "CRITICAL"
    assert statuses["x"] == "OK"
