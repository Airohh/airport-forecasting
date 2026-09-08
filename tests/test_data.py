"""Tests for data loading and processing."""

import pytest

from airport_forecast.data import DATA_DIR, load_pax, load_enriched
from airport_forecast.constants import AIRPORTS, CORE_AIRPORTS

_MONTHLY = DATA_DIR / "processed" / "pax_monthly.parquet"
_skip_monthly = pytest.mark.skipif(
    not _MONTHLY.exists(),
    reason="pax_monthly.parquet is not shipped (gitignored)",
)


@_skip_monthly
def test_load_pax_shape():
    df = load_pax(with_holidays=False)
    assert len(df) > 0
    assert "airport" in df.columns
    assert "date" in df.columns
    assert "pax" in df.columns


@_skip_monthly
def test_load_pax_holidays():
    df = load_pax(with_holidays=True)
    assert "n_holidays" in df.columns
    assert "is_school_vacation" in df.columns


@_skip_monthly
def test_load_pax_no_nulls_in_pax():
    df = load_pax(with_holidays=False)
    assert df["pax"].isna().sum() == 0


@_skip_monthly
def test_load_pax_positive_values():
    df = load_pax(with_holidays=False)
    assert (df["pax"] >= 0).all()


@_skip_monthly
def test_load_pax_airports_are_known():
    df = load_pax(with_holidays=False)
    for code in df["airport"].unique():
        assert code in AIRPORTS


@_skip_monthly
def test_load_pax_date_sorted():
    df = load_pax(with_holidays=False)
    for _, grp in df.groupby("airport"):
        dates = grp["date"].values
        assert (dates[1:] >= dates[:-1]).all()


def test_load_enriched_schema():
    df = load_enriched()
    for col in ["airport", "date", "pax", "n_flights", "pax_per_flight"]:
        assert col in df.columns, f"Missing column: {col}"
    assert len(df) > 0


def test_load_enriched_core_airports():
    df = load_enriched()
    for code in CORE_AIRPORTS:
        assert code in df["airport"].values, f"Missing airport: {code}"


def test_load_enriched_sorted():
    df = load_enriched()
    for _, grp in df.groupby("airport"):
        dates = grp["date"].values
        assert (dates[1:] >= dates[:-1]).all()
