"""Data loading utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def load_pax(with_holidays: bool = True) -> pd.DataFrame:
    """Load processed monthly PAX data."""
    name = "pax_monthly_holidays.parquet" if with_holidays else "pax_monthly.parquet"
    path = DATA_DIR / "processed" / name
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["airport", "date"]).reset_index(drop=True)


def load_enriched() -> pd.DataFrame:
    """Load macro-enriched PAX data with flight movements."""
    df = pd.read_parquet(DATA_DIR / "processed" / "pax_enriched.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["airport", "date"]).reset_index(drop=True)
