import json
from pathlib import Path

import numpy as np

from airport_forecast.conformal import (
    apply_interval,
    apply_relative_interval,
    coverage_and_width,
    relative_residuals,
    split_conformal_q,
)


def test_split_conformal_q_is_max_when_n_is_small():
    q = split_conformal_q(np.array([1.0, -2.0, 3.0, -4.0]), coverage=0.80)
    assert q == 4.0


def test_apply_interval_clips_at_zero():
    lower, upper = apply_interval(np.array([10.0, 100.0]), q=30.0)
    np.testing.assert_array_equal(lower, [0.0, 70.0])
    np.testing.assert_array_equal(upper, [40.0, 130.0])


def test_relative_residuals_drop_nonpositive_pred():
    r = relative_residuals(np.array([110.0, 90.0, 5.0]), np.array([100.0, 100.0, 0.0]))
    np.testing.assert_allclose(r[:2], [0.1, -0.1])
    assert np.isnan(r[2])
    assert split_conformal_q(r, coverage=0.5) == 0.1


def test_relative_interval_scales_with_prediction():
    lower, upper = apply_relative_interval(np.array([100.0, 1000.0]), q_rel=0.1)
    np.testing.assert_allclose(lower, [90.0, 900.0])
    np.testing.assert_allclose(upper, [110.0, 1100.0])


def test_coverage_and_width():
    y = np.array([10.0, 20.0, 30.0, 40.0])
    lower = np.array([9.0, 19.0, 0.0, 50.0])
    upper = np.array([11.0, 21.0, 31.0, 60.0])
    cov, width = coverage_and_width(y, lower, upper)
    assert cov == 75.0
    assert width == 11.25


def test_committed_conformal_covers_at_least_target():
    summary = Path(__file__).resolve().parent.parent / "reports" / "conformal_summary.json"
    data = json.loads(summary.read_text(encoding="utf-8"))
    assert data["coverage_target"] == 0.8
    assert data["test_coverage_pct"] >= 80
    assert data["min_airport_coverage_pct"] >= 80
    assert 0 < data["q_rel"] < 1


def test_committed_conformal_calibration_stops_before_test():
    """Calibration residuals must not reach into the test window."""
    summary = Path(__file__).resolve().parent.parent / "reports" / "conformal_summary.json"
    data = json.loads(summary.read_text(encoding="utf-8"))
    assert data["cal_end"] <= data["test_origin"]
