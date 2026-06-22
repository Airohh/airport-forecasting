"""Process raw Eurostat avia_paoa into clean monthly PAX per VINCI airport."""

import re
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

VINCI_AIRPORTS = {
    "FR_LFLL": "Lyon Saint-Exupéry",
    "FR_LFRS": "Nantes Atlantique",
    "UK_EGKK": "London Gatwick",
    "HU_LHBP": "Budapest",
    "PT_LPPT": "Lisbon",
    "PT_LPPR": "Porto",
    "RS_LYBE": "Belgrade",
    "UK_EGPH": "Edinburgh",
}

MONTHLY_PATTERN = re.compile(r"^\d{4}-\d{2}$")


def main() -> None:
    raw_path = RAW_DIR / "avia_paoa_full.parquet"
    print(f"Loading {raw_path}...")
    df = pd.read_parquet(raw_path)
    print(f"Raw shape: {df.shape}")

    # Filter: VINCI airports, monthly frequency, total passengers carried, total schedule
    vinci_codes = list(VINCI_AIRPORTS.keys())
    mask = (
        df["rep_airp"].isin(vinci_codes)
        & (df["freq"] == "M")
        & (df["tra_meas"] == "PAS_CRD")  # passengers carried (total)
        & (df["schedule"] == "TOT")       # all schedules
    )
    filtered = df[mask].copy()
    print(f"After VINCI + monthly + PAS_CRD + TOT filter: {filtered.shape}")

    if filtered.empty:
        print("\nNo data with PAS_CRD + TOT. Checking available values...")
        vinci_data = df[df["rep_airp"].isin(vinci_codes)]
        print(f"VINCI rows total: {len(vinci_data)}")
        print(f"tra_meas values: {vinci_data['tra_meas'].unique().tolist()}")
        print(f"schedule values: {vinci_data['schedule'].unique().tolist()}")
        print(f"freq values: {vinci_data['freq'].unique().tolist()}")
        tra_cov_col = [c for c in df.columns if "tra_cov" in c.lower()]
        if tra_cov_col:
            print(f"tra_cov values: {vinci_data[tra_cov_col[0]].unique().tolist()}")
        return

    # Identify monthly columns (YYYY-MM format)
    monthly_cols = [c for c in df.columns if MONTHLY_PATTERN.match(c)]
    print(f"Monthly columns found: {len(monthly_cols)} ({monthly_cols[0]} to {monthly_cols[-1]})")

    # Melt wide -> long
    id_cols = ["rep_airp"]
    tra_cov_col = [c for c in df.columns if "tra_cov" in c.lower()]
    if tra_cov_col:
        id_cols.append(tra_cov_col[0])

    long = filtered.melt(
        id_vars=id_cols,
        value_vars=monthly_cols,
        var_name="period",
        value_name="pax",
    )

    long["pax"] = pd.to_numeric(long["pax"], errors="coerce")
    long = long.dropna(subset=["pax"])
    long["pax"] = long["pax"].astype(int)
    long["date"] = pd.to_datetime(long["period"], format="%Y-%m")

    # If tra_cov exists, keep only TOTAL coverage
    if tra_cov_col:
        cov_col = tra_cov_col[0]
        total_vals = [v for v in long[cov_col].unique() if "TOTAL" in str(v).upper() or v == "TOTAL"]
        if total_vals:
            long = long[long[cov_col].isin(total_vals)]
            print(f"Filtered tra_cov to {total_vals}: {len(long)} rows")
        else:
            print(f"tra_cov values: {long[cov_col].unique().tolist()}")
            print("Keeping all tra_cov values (no TOTAL found)")

    long = long.rename(columns={"rep_airp": "airport"})
    long["airport_name"] = long["airport"].map(VINCI_AIRPORTS)

    result = long[["airport", "airport_name", "date", "pax"]].sort_values(
        ["airport", "date"]
    ).reset_index(drop=True)

    # Summary
    print(f"\nFinal dataset: {result.shape}")
    print(f"Date range: {result['date'].min()} to {result['date'].max()}")
    print("\nPer airport:")
    for code, name in VINCI_AIRPORTS.items():
        sub = result[result["airport"] == code]
        if len(sub) > 0:
            print(f"  {name} ({code}): {len(sub)} months, "
                  f"{sub['date'].min().strftime('%Y-%m')} to {sub['date'].max().strftime('%Y-%m')}, "
                  f"avg PAX: {sub['pax'].mean():,.0f}")
        else:
            print(f"  {name} ({code}): NO DATA")

    out_path = PROCESSED_DIR / "pax_monthly.parquet"
    result.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path}")

    csv_path = PROCESSED_DIR / "pax_monthly.csv"
    result.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
