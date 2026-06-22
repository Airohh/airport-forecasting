"""Feature engineering for airport PAX time series forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar features from the date column."""
    out = df.copy()
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["quarter"] = out["date"].dt.quarter

    # Cyclical encoding
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)

    # Summer flag (peak travel)
    out["is_summer"] = out["month"].isin([6, 7, 8]).astype(int)

    # COVID period
    out["is_covid"] = (
        (out["date"] >= "2020-03-01") & (out["date"] <= "2022-06-01")
    ).astype(int)

    return out


def add_lag_features(df: pd.DataFrame, lags: list[int] | None = None) -> pd.DataFrame:
    """Add lagged PAX values per airport. Must be sorted by (airport, date)."""
    if lags is None:
        lags = [1, 2, 3, 6, 12]
    out = df.copy()
    for lag in lags:
        out[f"pax_lag_{lag}"] = out.groupby("airport")["pax"].shift(lag)
    return out


def add_rolling_features(
    df: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Rolling mean and std on PAX per airport. Uses shift(1) to avoid leakage."""
    if windows is None:
        windows = [3, 6, 12]
    out = df.copy()
    for w in windows:
        shifted = out.groupby("airport")["pax"].shift(1)
        out[f"pax_rolling_mean_{w}"] = (
            shifted.groupby(out["airport"]).transform(
                lambda s: s.rolling(w, min_periods=1).mean()
            )
        )
        out[f"pax_rolling_std_{w}"] = (
            shifted.groupby(out["airport"]).transform(
                lambda s: s.rolling(w, min_periods=1).std()
            )
        )
    return out


def add_yoy_features(df: pd.DataFrame) -> pd.DataFrame:
    """Year-over-year growth rate (lagged to avoid target leakage)."""
    out = df.copy()
    pax_1 = out.groupby("airport")["pax"].shift(1)
    pax_13 = out.groupby("airport")["pax"].shift(13)
    out["pax_yoy_growth"] = (pax_1 - pax_13) / pax_13.replace(0, np.nan)
    return out


def add_network_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-airport features: total PAX by country, market share, rank."""
    out = df.copy()
    country_map = {
        "FR_LFLL": "FR", "FR_LFRS": "FR",
        "HU_LHBP": "HU", "PT_LPPT": "PT", "PT_LPPR": "PT",
        "RS_LYBE": "RS", "UK_EGKK": "UK", "UK_EGPH": "UK",
    }
    out["_country"] = out["airport"].map(country_map)

    # Total PAX across all airports per month (network-level)
    network_total = out.groupby("date")["pax"].transform("sum")
    out["network_total_pax"] = out.groupby("airport")["pax"].transform(
        lambda s: network_total.loc[s.index].shift(1)
    )

    # Total PAX by country per month
    country_total = out.groupby(["_country", "date"])["pax"].transform("sum")
    out["country_total_pax"] = out.groupby("airport")["pax"].transform(
        lambda s: country_total.loc[s.index].shift(1)
    )

    # Market share within country (lagged)
    out["country_market_share"] = np.where(
        out["country_total_pax"] > 0,
        out.groupby("airport")["pax"].shift(1) / out["country_total_pax"],
        np.nan,
    )

    # Rank within network per month (lagged)
    lagged_pax = out.groupby("airport")["pax"].shift(1)
    out["network_rank"] = lagged_pax.groupby(out["date"]).rank(ascending=False, method="min")

    out = out.drop(columns=["_country"])
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature pipeline: time + lags + rolling + YoY + network. Input must have date + airport + pax."""
    out = df.sort_values(["airport", "date"]).reset_index(drop=True)
    out = add_time_features(out)
    out = add_lag_features(out)
    out = add_rolling_features(out)
    out = add_yoy_features(out)
    out = add_network_features(out)
    return out


def temporal_train_val_test_split(
    df: pd.DataFrame,
    train_end: str = "2023-12",
    val_end: str = "2024-12",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by date. Train: →train_end, Val: →val_end, Test: rest."""
    train_cutoff = pd.Timestamp(train_end)
    val_cutoff = pd.Timestamp(val_end)
    train = df[df["date"] <= train_cutoff].copy()
    val = df[(df["date"] > train_cutoff) & (df["date"] <= val_cutoff)].copy()
    test = df[df["date"] > val_cutoff].copy()
    return train, val, test
