"""Add holiday and school vacation features per airport country."""

from __future__ import annotations

import holidays as hd
import numpy as np
import pandas as pd

from airport_forecast.constants import AIRPORT_COUNTRIES

SCHOOL_VACATION_MONTHS_FR = {2, 4, 7, 8, 10, 12}
SCHOOL_VACATION_MONTHS_GB = {4, 7, 8, 12}
SCHOOL_VACATION_MONTHS_PT = {7, 8, 12}
SCHOOL_VACATION_MONTHS_HU = {7, 8, 12}

_SCHOOL_VACATIONS: dict[str, set[int]] = {
    "FR": SCHOOL_VACATION_MONTHS_FR,
    "GB": SCHOOL_VACATION_MONTHS_GB,
    "PT": SCHOOL_VACATION_MONTHS_PT,
    "HU": SCHOOL_VACATION_MONTHS_HU,
    "RS": {7, 8, 12},
}


def count_holidays_in_month(year: int, month: int, country_code: str) -> int:
    try:
        cal = hd.country_holidays(country_code, years=year)
    except NotImplementedError:
        return 0
    return sum(1 for d in cal.keys() if d.month == month)


def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add holiday count and school vacation flag per row.

    Expects columns: 'airport' (Eurostat code) and 'date' (datetime).
    """
    out = df.copy()
    out["country"] = out["airport"].map(AIRPORT_COUNTRIES)
    out["n_holidays"] = 0
    out["is_school_vacation"] = 0

    for _, row in out.iterrows():
        cc = row["country"]
        if pd.isna(cc):
            continue
        yr, mo = row["date"].year, row["date"].month
        out.at[_, "n_holidays"] = count_holidays_in_month(yr, mo, cc)
        vac_months = _SCHOOL_VACATIONS.get(cc, set())
        out.at[_, "is_school_vacation"] = int(mo in vac_months)

    out["n_holidays"] = out["n_holidays"].astype(int)
    out["is_school_vacation"] = out["is_school_vacation"].astype(int)
    return out


def add_holiday_features_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """Faster vectorized version — precomputes per (country, year, month)."""
    out = df.copy()
    out["country"] = out["airport"].map(AIRPORT_COUNTRIES)

    combos = out[["country", "date"]].dropna(subset=["country"]).copy()
    combos["year"] = combos["date"].dt.year
    combos["month"] = combos["date"].dt.month
    unique_combos = combos[["country", "year", "month"]].drop_duplicates()

    holiday_cache: dict[tuple[str, int, int], int] = {}
    for _, r in unique_combos.iterrows():
        key = (r["country"], r["year"], r["month"])
        holiday_cache[key] = count_holidays_in_month(r["year"], r["month"], r["country"])

    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["n_holidays"] = out.apply(
        lambda r: holiday_cache.get((r["country"], r["year"], r["month"]), 0), axis=1
    )
    out["is_school_vacation"] = out.apply(
        lambda r: int(r["month"] in _SCHOOL_VACATIONS.get(r["country"], set()))
        if pd.notna(r["country"]) else 0,
        axis=1,
    )
    out = out.drop(columns=["year", "month", "country"])
    return out
