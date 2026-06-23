"""Train the growth-ratio LightGBM (Budapest / high-growth fix) and backtest it
against the level model in a genuine FORWARD setting.

Why this exists
---------------
A level tree cannot predict above the traffic it saw in training, so on a fast-
growing airport (Budapest ~+20%/yr) the forward forecast plateaus and under-shoots.
The growth model predicts the YoY ratio (stationary, in-range) and rebuilds the
level via pax_lag_12 — so it can extrapolate the trend.

The in-sample backtest (forecasting 2025, which mostly stays within 2024 range)
HIDES the problem. The honest test is the forward window past end-of-data, where
the level model collapses on Budapest. This script reports the backtest MAPE AND
the forward divergence so the fix is justified by numbers, not assertion.

Run:  python scripts/train_growth.py
Writes: models/lightgbm_growth.pkl, reports/growth_vs_level.csv
"""

from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from airport_forecast.constants import CORE_AIRPORTS, SHORT_NAMES as SHORT, VAL_END
from airport_forecast.data import load_enriched
from airport_forecast.features import build_features, temporal_train_val_test_split
from airport_forecast.models import (
    forecast_future_global,
    recursive_forecast_global,
    train_lightgbm_global,
    train_lightgbm_growth,
)

warnings.filterwarnings("ignore")
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)


def _mape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    m = y_true > 0
    return float(np.mean(np.abs((y_true[m] - y_pred[m]) / y_true[m])) * 100)


def main() -> None:
    enriched = load_enriched()
    enriched = enriched[enriched["airport"].isin(CORE_AIRPORTS)].copy()
    feat = build_features(enriched)

    # ── 1. Honest forward backtest: train ≤ VAL_END, recurse over the 2025 test ──
    train, val, test = temporal_train_val_test_split(feat, VAL_END, VAL_END)
    lag_cols = [c for c in feat.columns if "lag" in c or "rolling" in c]
    trainval = pd.concat([train, val]).dropna(subset=lag_cols)

    lvl_model, lvl_cols = train_lightgbm_global(trainval, None)
    grw_model, grw_cols = train_lightgbm_growth(trainval, None)

    fc_lvl = recursive_forecast_global(
        lvl_model, lvl_cols, enriched, VAL_END, CORE_AIRPORTS, target_mode="level"
    )
    fc_grw = recursive_forecast_global(
        grw_model, grw_cols, enriched, VAL_END, CORE_AIRPORTS, target_mode="growth"
    )

    rows = []
    for ap in CORE_AIRPORTS:
        a = fc_lvl[fc_lvl["airport"] == ap].dropna(subset=["pax_actual", "pax_pred"])
        b = fc_grw[fc_grw["airport"] == ap].dropna(subset=["pax_actual", "pax_pred"])
        if a.empty or b.empty:
            continue
        rows.append({
            "airport": SHORT[ap],
            "backtest_MAPE_level": round(_mape(a["pax_actual"], a["pax_pred"]), 2),
            "backtest_MAPE_growth": round(_mape(b["pax_actual"], b["pax_pred"]), 2),
        })
    bt = pd.DataFrame(rows)

    # ── 2. Forward divergence past end-of-data (where the level model collapses) ──
    # Use PRODUCTION models trained on ALL available data, not the backtest fold.
    all_clean = feat.dropna(subset=lag_cols)
    lvl_full, lvl_full_cols = train_lightgbm_global(all_clean, None)
    grw_full, grw_full_cols = train_lightgbm_growth(all_clean, None)
    fwd_lvl = forecast_future_global(lvl_full, lvl_full_cols, enriched, CORE_AIRPORTS, 12,
                                     target_mode="level")
    fwd_grw = forecast_future_global(grw_full, grw_full_cols, enriched, CORE_AIRPORTS, 12,
                                     target_mode="growth")
    for ap in CORE_AIRPORTS:
        lvl = fwd_lvl[fwd_lvl["airport"] == ap].sort_values("date")["pax_pred"]
        g = fwd_grw[fwd_grw["airport"] == ap].sort_values("date")["pax_pred"]
        if lvl.empty or g.empty:
            continue
        idx = bt.index[bt["airport"] == SHORT[ap]]
        if len(idx):
            bt.loc[idx, "fwd_M12_level_M"] = round(lvl.iloc[-1] / 1e6, 2)
            bt.loc[idx, "fwd_M12_growth_M"] = round(g.iloc[-1] / 1e6, 2)
            bt.loc[idx, "fwd_uplift_%"] = round((g.iloc[-1] - lvl.iloc[-1]) / lvl.iloc[-1] * 100, 0)

    bt.to_csv(REPORTS / "growth_vs_level.csv", index=False)
    print(bt.to_string(index=False))

    # ── 3. Save the production growth model (trained on ALL data in step 2) ──
    with open(MODELS / "lightgbm_growth.pkl", "wb") as f:
        pickle.dump({"model": grw_full, "feature_cols": grw_full_cols}, f)
    print(f"\nSaved {MODELS / 'lightgbm_growth.pkl'}")

    # ── 4. Per-airport champion: pick level vs growth by backtested forward MAPE.
    # Honest model selection — growth is NOT a blanket win (it overshoots Budapest).
    champ = bt.assign(
        champion=np.where(bt["backtest_MAPE_growth"] < bt["backtest_MAPE_level"],
                          "growth", "level")
    )[["airport", "backtest_MAPE_level", "backtest_MAPE_growth", "champion"]]
    champ.to_csv(REPORTS / "champion_by_airport.csv", index=False)
    print("\nPer-airport champion (by backtest forward MAPE):")
    print(champ.to_string(index=False))


if __name__ == "__main__":
    main()
