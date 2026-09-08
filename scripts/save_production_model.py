"""Train LightGBM up to VAL_END and write models/lightgbm_global.pkl.

Same cutoff as evaluate_lightgbm_recursive / the Validation tab.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from airport_forecast.constants import CORE_AIRPORTS, VAL_END
from airport_forecast.data import load_enriched
from airport_forecast.models import evaluate_lightgbm_recursive

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "lightgbm_global.pkl"


def main() -> None:
    raw = load_enriched()
    model, fcols, results = evaluate_lightgbm_recursive(
        raw, val_end=VAL_END, core_airports=CORE_AIRPORTS
    )
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "feature_cols": fcols}, f)
    print(f"Wrote {MODEL_PATH.relative_to(ROOT)} (train <= {VAL_END}, {len(fcols)} features)")
    for r in results:
        print(f"  {r.airport}  horizon={r.horizon}  MAPE={r.mape:.2f}")


if __name__ == "__main__":
    main()
