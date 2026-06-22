"""Download macro-economic enrichment data v2 — fixed dedup + real oil prices."""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

COUNTRY_MAP = {
    "FR_LFLL": "FR", "FR_LFRS": "FR",
    "UK_EGKK": "UK", "UK_EGPH": "UK",
    "HU_LHBP": "HU", "PT_LPPT": "PT", "PT_LPPR": "PT",
    "RS_LYBE": "RS",
}
COUNTRIES = ["FR", "HU", "PT", "RS", "UK"]


def download_unemployment() -> pd.DataFrame:
    import eurostat
    print("[1/5] Downloading unemployment...")
    df = eurostat.get_data_df("ei_lmhr_m")
    geo_col = [c for c in df.columns if "geo" in c.lower()][0]
    monthly_cols = [c for c in df.columns if len(c) == 7 and c[4] == "-" and c[5] != "Q"]

    mask = df[geo_col].isin(COUNTRIES)
    s_adj_col = [c for c in df.columns if "s_adj" in c.lower()]
    if s_adj_col:
        mask = mask & (df[s_adj_col[0]] == "SA")

    filtered = df[mask]
    long = filtered.melt(id_vars=[geo_col], value_vars=monthly_cols,
                         var_name="period", value_name="unemployment_rate")
    long["unemployment_rate"] = pd.to_numeric(long["unemployment_rate"], errors="coerce")
    long = long.dropna(subset=["unemployment_rate"])
    long["date"] = pd.to_datetime(long["period"], format="%Y-%m")
    long = long.rename(columns={geo_col: "country"})

    # DEDUP: one value per country-date
    result = long.groupby(["country", "date"])["unemployment_rate"].mean().reset_index()
    result = result.sort_values(["country", "date"])
    print(f"  OK: {result.shape[0]} rows, countries: {result.country.unique().tolist()}")
    return result


def download_gdp() -> pd.DataFrame:
    import eurostat
    print("[2/5] Downloading GDP...")
    df = eurostat.get_data_df("namq_10_gdp")
    geo_col = [c for c in df.columns if "geo" in c.lower()][0]
    quarterly_cols = [c for c in df.columns if len(c) == 7 and "-Q" in c]

    na_item_col = [c for c in df.columns if "na_item" in c.lower()]
    unit_col = [c for c in df.columns if c.lower() == "unit"]
    s_adj_col = [c for c in df.columns if "s_adj" in c.lower()]

    mask = df[geo_col].isin(COUNTRIES)
    if na_item_col:
        mask = mask & (df[na_item_col[0]] == "B1GQ")
    if unit_col:
        cp_vals = [v for v in df[unit_col[0]].unique() if "CP_MEUR" in str(v)]
        if cp_vals:
            mask = mask & (df[unit_col[0]] == cp_vals[0])
    if s_adj_col:
        sa_vals = [v for v in df[s_adj_col[0]].unique() if v == "SCA" or v == "SA"]
        if sa_vals:
            mask = mask & (df[s_adj_col[0]] == sa_vals[0])

    filtered = df[mask]
    if filtered.empty:
        mask2 = df[geo_col].isin(COUNTRIES)
        if na_item_col:
            mask2 = mask2 & (df[na_item_col[0]] == "B1GQ")
        filtered = df[mask2]
        if not filtered.empty:
            if unit_col:
                filtered = filtered[filtered[unit_col[0]] == filtered[unit_col[0]].iloc[0]]
            if s_adj_col:
                filtered = filtered[filtered[s_adj_col[0]] == filtered[s_adj_col[0]].iloc[0]]

    long = filtered.melt(id_vars=[geo_col], value_vars=quarterly_cols,
                         var_name="period", value_name="gdp")
    long["gdp"] = pd.to_numeric(long["gdp"], errors="coerce")
    long = long.dropna(subset=["gdp"])
    long = long.rename(columns={geo_col: "country"})

    # DEDUP
    long = long.groupby(["country", "period"])["gdp"].mean().reset_index()

    def q2date(q):
        return pd.Timestamp(year=int(q[:4]), month=(int(q[-1]) - 1) * 3 + 1, day=1)

    long["date"] = long["period"].apply(q2date)

    # Interpolate quarterly -> monthly
    parts = []
    for country in long["country"].unique():
        sub = long[long["country"] == country].set_index("date")["gdp"].sort_index()
        idx = pd.date_range(sub.index.min(), sub.index.max(), freq="MS")
        monthly = sub.reindex(idx).interpolate("linear")
        parts.append(pd.DataFrame({"country": country, "date": monthly.index, "gdp": monthly.values}))

    result = pd.concat(parts, ignore_index=True)
    print(f"  OK: {result.shape[0]} rows (quarterly interpolated to monthly)")
    return result


def download_oil_prices() -> pd.DataFrame:
    print("[3/5] Downloading Brent oil prices (FRED)...")
    try:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=POILBREUSDM"
        oil = pd.read_csv(url)
        oil.columns = ["date", "oil_price_usd"]
        oil["date"] = pd.to_datetime(oil["date"])
        oil["oil_price_usd"] = pd.to_numeric(oil["oil_price_usd"], errors="coerce")
        oil = oil.dropna(subset=["oil_price_usd"])
        # Align to first of month
        oil["date"] = oil["date"].dt.to_period("M").dt.to_timestamp()
        oil = oil.groupby("date")["oil_price_usd"].mean().reset_index()
        print(f"  OK: {oil.shape[0]} months, {oil.date.min().strftime('%Y-%m')} to {oil.date.max().strftime('%Y-%m')}")
        return oil
    except Exception as e:
        print(f"  FRED failed: {e}, trying EIA...")
        try:
            url2 = "https://www.eia.gov/dnav/pet/hist_xls/RBRTEm.xls"
            oil2 = pd.read_excel(url2, sheet_name=1, skiprows=2)
            oil2.columns = ["date", "oil_price_usd"]
            oil2["date"] = pd.to_datetime(oil2["date"])
            oil2["date"] = oil2["date"].dt.to_period("M").dt.to_timestamp()
            oil2 = oil2.groupby("date")["oil_price_usd"].mean().reset_index()
            print(f"  OK (EIA): {oil2.shape[0]} months")
            return oil2
        except Exception as e2:
            print(f"  EIA also failed: {e2}")
            print("  SKIPPING oil prices")
            return pd.DataFrame(columns=["date", "oil_price_usd"])


def download_exchange_rates() -> pd.DataFrame:
    print("[4/5] Downloading exchange rates (ECB)...")
    rates = []
    currencies = {"HU": "HUF", "UK": "GBP"}

    for country, cur in currencies.items():
        try:
            url = f"https://data-api.ecb.europa.eu/service/data/EXR/M.{cur}.EUR.SP00.A?format=csvdata"
            ecb = pd.read_csv(url)
            ecb["date"] = pd.to_datetime(ecb["TIME_PERIOD"], format="%Y-%m")
            ecb["exchange_rate"] = pd.to_numeric(ecb["OBS_VALUE"], errors="coerce")
            ecb["country"] = country
            rates.append(ecb[["country", "date", "exchange_rate"]])
            print(f"  {cur}: {len(ecb)} months")
        except Exception as e:
            print(f"  {cur} failed: {e}")

    # EUR countries
    if rates:
        all_dates = pd.concat([r["date"] for r in rates]).drop_duplicates().sort_values()
    else:
        all_dates = pd.date_range("1999-01-01", "2026-06-01", freq="MS")

    for country in ["FR", "PT"]:
        rates.append(pd.DataFrame({"country": country, "date": all_dates, "exchange_rate": 1.0}))

    # Serbia: use approximate EUR/RSD (relatively stable ~117)
    rates.append(pd.DataFrame({"country": "RS", "date": all_dates, "exchange_rate": 117.0}))

    result = pd.concat(rates, ignore_index=True).sort_values(["country", "date"])
    result = result.groupby(["country", "date"])["exchange_rate"].mean().reset_index()
    print(f"  Total: {result.shape[0]} rows")
    return result


def build_events() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("[5/5] Building event flags...")

    # Airport-specific events
    rows = []
    airports = ["FR_LFLL", "FR_LFRS", "HU_LHBP", "PT_LPPT", "PT_LPPR", "RS_LYBE",
                "UK_EGKK", "UK_EGPH"]
    all_dates = pd.date_range("1998-01-01", "2026-06-01", freq="MS")

    for ap in airports:
        for d in all_dates:
            row = {"airport": ap, "date": d}

            # COVID
            row["event_covid"] = int(pd.Timestamp("2020-03-01") <= d <= pd.Timestamp("2022-06-01"))

            # Ukraine war (global but stronger impact on Eastern Europe)
            row["event_ukraine_war"] = int(d >= pd.Timestamp("2022-02-01") and ap in ["HU_LHBP", "RS_LYBE"])

            # Major sports
            row["event_major_sport"] = 0
            if ap == "FR_LFLL" and d in [pd.Timestamp("2024-07-01"), pd.Timestamp("2024-08-01")]:
                row["event_major_sport"] = 1  # JO Paris
            if ap == "FR_LFRS" and d in [pd.Timestamp("2024-07-01"), pd.Timestamp("2024-08-01")]:
                row["event_major_sport"] = 1  # JO Paris
            if ap == "PT_LPPT" and d == pd.Timestamp("2004-06-01"):
                row["event_major_sport"] = 1  # Euro 2004
            if ap == "PT_LPPT" and d == pd.Timestamp("2004-07-01"):
                row["event_major_sport"] = 1

            # Conferences
            row["event_conference"] = 0
            if ap == "PT_LPPT" and d.month == 11 and d.year >= 2016:
                row["event_conference"] = 1  # Web Summit

            rows.append(row)

    events_df = pd.DataFrame(rows)
    n_covid = events_df["event_covid"].sum()
    n_war = events_df["event_ukraine_war"].sum()
    n_sport = events_df["event_major_sport"].sum()
    n_conf = events_df["event_conference"].sum()
    print(f"  Events: covid={n_covid}, ukraine_war={n_war}, sport={n_sport}, conference={n_conf}")
    return events_df


def main():
    print("=" * 60)
    print("MACRO ENRICHMENT v2")
    print("=" * 60)

    unemployment = download_unemployment()
    gdp = download_gdp()
    oil = download_oil_prices()
    exchange = download_exchange_rates()
    events = build_events()

    # Save raw
    for name, data in [("unemployment", unemployment), ("gdp", gdp),
                       ("oil_prices", oil), ("exchange_rates", exchange),
                       ("events", events)]:
        data.to_parquet(RAW_DIR / f"{name}.parquet", index=False)

    # Load PAX
    print("\nMerging into PAX dataset...")
    pax = pd.read_parquet(PROCESSED_DIR / "pax_monthly_holidays.parquet")
    pax["date"] = pd.to_datetime(pax["date"])
    pax["country"] = pax["airport"].map(COUNTRY_MAP)
    original_len = len(pax)

    # Merge each source
    pax = pax.merge(unemployment, on=["country", "date"], how="left")
    assert len(pax) == original_len, f"Unemployment merge created duplicates: {len(pax)} vs {original_len}"
    print(f"  + unemployment: {pax['unemployment_rate'].notna().sum()}/{original_len}")

    pax = pax.merge(gdp, on=["country", "date"], how="left")
    assert len(pax) == original_len, f"GDP merge created duplicates: {len(pax)} vs {original_len}"
    print(f"  + GDP: {pax['gdp'].notna().sum()}/{original_len}")

    if not oil.empty:
        pax = pax.merge(oil, on="date", how="left")
        assert len(pax) == original_len, f"Oil merge created duplicates"
        print(f"  + oil: {pax['oil_price_usd'].notna().sum()}/{original_len}")

    pax = pax.merge(exchange, on=["country", "date"], how="left")
    assert len(pax) == original_len, f"Exchange merge created duplicates"
    print(f"  + exchange: {pax['exchange_rate'].notna().sum()}/{original_len}")

    pax = pax.merge(events, on=["airport", "date"], how="left")
    assert len(pax) == original_len, f"Events merge created duplicates: {len(pax)} vs {original_len}"
    for col in ["event_covid", "event_ukraine_war", "event_major_sport", "event_conference"]:
        pax[col] = pax[col].fillna(0).astype(int)

    pax = pax.drop(columns=["country"])

    print(f"\nFinal shape: {pax.shape} (original: {original_len})")
    print(f"Columns ({len(pax.columns)}): {pax.columns.tolist()}")
    print("\nMissing:")
    for c in pax.columns:
        n = pax[c].isna().sum()
        if n > 0:
            print(f"  {c}: {n} ({n/len(pax)*100:.1f}%)")

    pax.to_parquet(PROCESSED_DIR / "pax_enriched.parquet", index=False)
    pax.to_csv(PROCESSED_DIR / "pax_enriched.csv", index=False)
    print(f"\nSaved: pax_enriched.parquet ({len(pax)} rows, {len(pax.columns)} cols)")


if __name__ == "__main__":
    main()
