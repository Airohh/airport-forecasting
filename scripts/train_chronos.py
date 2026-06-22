"""Train Chronos foundation model and add results to comparison."""

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from airport_forecast.models import evaluate_chronos, results_to_dataframe

REPORTS = Path(__file__).resolve().parent.parent / "reports"

SHORT = {
    "FR_LFLL": "Lyon", "FR_LFRS": "Nantes", "HU_LHBP": "Budapest",
    "PT_LPPT": "Lisbon", "PT_LPPR": "Porto", "RS_LYBE": "Belgrade",
}
CORE = list(SHORT.keys())

# Load data
enriched = pd.read_parquet(
    Path(__file__).resolve().parent.parent / "data" / "processed" / "pax_enriched.parquet"
)
enriched["date"] = pd.to_datetime(enriched["date"])
raw = enriched[enriched["airport"].isin(CORE)].copy()

print("=== Chronos (zero-shot foundation model) ===")
print("Loading model (first run downloads ~150MB)...")

all_results = []
for ap in CORE:
    t0 = time.time()
    try:
        results = evaluate_chronos(raw, ap)
        all_results.extend(results)
        for r in results:
            print(f"  {SHORT[ap]}: MAPE={r.mape:.1f}%, MAE={r.mae:,.0f}, h={r.horizon}")
    except Exception as e:
        print(f"  {SHORT[ap]}: FAILED - {e}")
    print(f"    ({time.time()-t0:.1f}s)")

if not all_results:
    print("No results. Exiting.")
    sys.exit(1)

# Merge with existing results
df_new = results_to_dataframe(all_results)
existing = pd.read_csv(REPORTS / "model_results.csv")
combined = pd.concat([existing, df_new], ignore_index=True)
combined.to_csv(REPORTS / "model_results.csv", index=False)

print("\n--- CHRONOS RESULTS ---")
df_new["airport_name"] = df_new["airport"].map(SHORT)
print(df_new[["airport_name", "model", "mape", "mae"]].to_string(index=False))

print(f"\nAvg MAPE: {df_new['mape'].mean():.1f}%")
print("Results appended to model_results.csv")
