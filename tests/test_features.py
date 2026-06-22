"""Tests for feature engineering."""

import numpy as np
import pandas as pd
import pytest

from airport_forecast.features import (
    add_time_features,
    add_lag_features,
    add_rolling_features,
    build_features,
    temporal_train_val_test_split,
)


@pytest.fixture
def sample_df():
    dates = pd.date_range("2020-01-01", periods=36, freq="MS")
    return pd.DataFrame({
        "airport": "FR_LFLL",
        "date": dates,
        "pax": np.random.randint(100000, 1000000, 36),
    })


def test_add_time_features(sample_df):
    result = add_time_features(sample_df)
    assert "month" in result.columns
    assert "month_sin" in result.columns
    assert "month_cos" in result.columns
    assert "is_summer" in result.columns
    assert "is_covid" in result.columns
    assert result["month"].min() >= 1
    assert result["month"].max() <= 12


def test_add_lag_features(sample_df):
    result = add_lag_features(sample_df, lags=[1, 12])
    assert "pax_lag_1" in result.columns
    assert "pax_lag_12" in result.columns
    assert result["pax_lag_1"].isna().sum() == 1
    assert result["pax_lag_12"].isna().sum() == 12


def test_add_rolling_features(sample_df):
    result = add_rolling_features(sample_df, windows=[3])
    assert "pax_rolling_mean_3" in result.columns
    assert "pax_rolling_std_3" in result.columns


def test_no_leakage_in_rolling(sample_df):
    result = add_rolling_features(sample_df, windows=[3])
    for i in range(4, len(result)):
        rolling_val = result.iloc[i]["pax_rolling_mean_3"]
        if np.isnan(rolling_val):
            continue
        # shift(1) + rolling(3) => mean(pax[i-1], pax[i-2], pax[i-3])
        expected = result.iloc[i - 3 : i]["pax"].mean()
        assert abs(rolling_val - expected) < 1e-6, (
            f"Row {i}: rolling_mean_3={rolling_val:.2f} != expected {expected:.2f}"
        )


def test_build_features(sample_df):
    result = build_features(sample_df)
    assert len(result) == len(sample_df)
    assert "pax_lag_1" in result.columns
    assert "pax_yoy_growth" in result.columns


def test_temporal_split(sample_df):
    sample_df["date"] = pd.date_range("2022-01-01", periods=36, freq="MS")
    result = build_features(sample_df)
    train, val, test = temporal_train_val_test_split(result, "2023-12", "2024-06")
    assert train["date"].max() <= pd.Timestamp("2023-12-01")
    assert val["date"].min() > pd.Timestamp("2023-12-01")
    assert val["date"].max() <= pd.Timestamp("2024-06-01")
    assert test["date"].min() > pd.Timestamp("2024-06-01")
    assert len(train) + len(val) + len(test) == len(result)
