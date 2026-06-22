"""Tests for data loading and processing."""

import pandas as pd
import pytest

from airport_forecast.data import load_pax
from airport_forecast.constants import VINCI_AIRPORTS


def test_load_pax_shape():
    df = load_pax(with_holidays=False)
    assert len(df) > 0
    assert "airport" in df.columns
    assert "date" in df.columns
    assert "pax" in df.columns


def test_load_pax_holidays():
    df = load_pax(with_holidays=True)
    assert "n_holidays" in df.columns
    assert "is_school_vacation" in df.columns


def test_load_pax_no_nulls_in_pax():
    df = load_pax(with_holidays=False)
    assert df["pax"].isna().sum() == 0


def test_load_pax_positive_values():
    df = load_pax(with_holidays=False)
    assert (df["pax"] >= 0).all()


def test_load_pax_airports_are_vinci():
    df = load_pax(with_holidays=False)
    for code in df["airport"].unique():
        assert code in VINCI_AIRPORTS


def test_load_pax_date_sorted():
    df = load_pax(with_holidays=False)
    for _, grp in df.groupby("airport"):
        dates = grp["date"].values
        assert (dates[1:] >= dates[:-1]).all()
