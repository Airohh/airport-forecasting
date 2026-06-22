"""Download Eurostat avia_paoa dataset and filter for VINCI Airports."""

import time
from pathlib import Path

import eurostat
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

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


def main() -> None:
    print("Downloading avia_paoa from Eurostat (this may take 1-2 min)...")
    t0 = time.time()
    df = eurostat.get_data_df("avia_paoa")
    print(f"Downloaded in {time.time() - t0:.0f}s — shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()[:10]}...")

    raw_path = RAW_DIR / "avia_paoa_full.parquet"
    df.to_parquet(raw_path, index=False)
    print(f"Full dataset saved: {raw_path} ({raw_path.stat().st_size / 1e6:.1f} MB)")

    airport_col = None
    for col in df.columns:
        if "airp" in col.lower() or "geo" in col.lower():
            sample = df[col].dropna().astype(str).head(100)
            if sample.str.contains("FR_LFLL|UK_EGKK|LFLL|EGKK", regex=True).any():
                airport_col = col
                break

    if airport_col is None:
        print("\nCould not auto-detect airport column. Columns:")
        for c in df.columns:
            print(f"  {c}: {df[c].dropna().astype(str).head(3).tolist()}")
        return

    print(f"\nAirport column detected: '{airport_col}'")
    print(f"Sample values: {df[airport_col].dropna().unique()[:10]}")

    vinci_codes = list(VINCI_AIRPORTS.keys())
    mask = df[airport_col].isin(vinci_codes)
    df_vinci = df[mask].copy()
    print(f"\nFiltered VINCI airports: {df_vinci.shape[0]} rows")
    print(f"Airports found: {df_vinci[airport_col].unique().tolist()}")

    vinci_path = RAW_DIR / "avia_paoa_vinci.parquet"
    df_vinci.to_parquet(vinci_path, index=False)
    print(f"VINCI subset saved: {vinci_path}")


if __name__ == "__main__":
    main()
