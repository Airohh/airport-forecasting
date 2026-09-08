"""80% intervals on the recursive path (split conformal).

Calibrate on 2024 (train <= 2023-12), evaluate coverage on 2025+ (train <= 2024-12).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from airport_forecast.conformal import apply_interval, coverage_and_width, split_conformal_q
from airport_forecast.constants import CORE_AIRPORTS, SHORT_NAMES
from airport_forecast.data import load_enriched
from airport_forecast.features import build_features, temporal_train_val_test_split
from airport_forecast.models import recursive_forecast_global, train_lightgbm_global

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
CAL_END = "2023-12"
TEST_END = "2024-12"


def _train_until(enriched: pd.DataFrame, until: str):
    feat = build_features(enriched)
    feat_core = feat[feat["airport"].isin(CORE_AIRPORTS)].copy()
    train, _, _ = temporal_train_val_test_split(feat_core, until, until)
    lag_cols = [c for c in train.columns if "lag" in c or "rolling" in c]
    train_clean = train.dropna(subset=lag_cols)
    return train_lightgbm_global(train_clean, None)


def main() -> None:
    enriched = load_enriched()

    print(f"Calibrate: train <= {CAL_END}, residuals on 2024")
    model_cal, fcols_cal = _train_until(enriched, CAL_END)
    fc_cal = recursive_forecast_global(
        model_cal, fcols_cal, enriched, origin_date=CAL_END, airports=CORE_AIRPORTS
    )
    cal = fc_cal.dropna(subset=["pax_actual", "pax_pred"])
    residuals = (cal["pax_actual"] - cal["pax_pred"]).to_numpy()
    q = split_conformal_q(residuals, coverage=0.80)
    print(f"  n_cal={len(residuals)}  q={q:,.0f} PAX")

    print(f"Test: train <= {TEST_END}, coverage on 2025+")
    model_test, fcols_test = _train_until(enriched, TEST_END)
    fc_test = recursive_forecast_global(
        model_test, fcols_test, enriched, origin_date=TEST_END, airports=CORE_AIRPORTS
    )
    test = fc_test.dropna(subset=["pax_actual", "pax_pred"]).copy()
    lower, upper = apply_interval(test["pax_pred"].to_numpy(), q)
    test["lower"] = lower
    test["upper"] = upper
    test["in_interval"] = (test["pax_actual"] >= test["lower"]) & (
        test["pax_actual"] <= test["upper"]
    )
    cov, width = coverage_and_width(
        test["pax_actual"].to_numpy(), test["lower"].to_numpy(), test["upper"].to_numpy()
    )
    print(f"  coverage={cov:.1f}% (cible 80%)  largeur moyenne={width:,.0f} PAX")
    for ap in CORE_AIRPORTS:
        sub = test[test["airport"] == ap]
        if sub.empty:
            continue
        c, w = coverage_and_width(
            sub["pax_actual"].to_numpy(), sub["lower"].to_numpy(), sub["upper"].to_numpy()
        )
        print(f"  {SHORT_NAMES[ap]:>10s}: {c:.0f}%  width={w:,.0f}")

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = test[["airport", "date", "pax_actual", "pax_pred", "lower", "upper", "in_interval"]]
    out.to_csv(REPORTS / "conformal_intervals.csv", index=False)
    summary = {
        "method": "split-conformal on recursive residuals",
        "coverage_target": 0.80,
        "q_pax": round(q, 1),
        "n_cal": int(len(residuals)),
        "cal_origin": CAL_END,
        "test_origin": TEST_END,
        "test_coverage_pct": round(cov, 1),
        "test_mean_width_pax": round(width, 0),
        "n_test": int(len(test)),
    }
    (REPORTS / "conformal_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("Wrote reports/conformal_intervals.csv and conformal_summary.json")


if __name__ == "__main__":
    main()
