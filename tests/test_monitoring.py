"""Tests for drift monitoring."""

import numpy as np
import pytest

from airport_forecast.monitoring import psi, check_prediction_drift


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
