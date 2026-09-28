"""80% intervals on the recursive path (split conformal).

Calibrate on 2024 (train <= 2023-12), evaluate coverage on 2025+ (train <= 2024-12).

The calibration forecast runs past 2024 (the data goes to 2025/2026), so its
residuals are cut at TEST_END: anything later overlaps the test window.

Primary score is relative (|y - ŷ| / ŷ). The absolute score (± PAX) is kept in
the summary as a baseline: one PAX width over-covers small airports and
under-covers Lisbon.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from airport_forecast.conformal import (
    apply_interval,
    apply_relative_interval,
    coverage_and_width,
    relative_residuals,
    split_conformal_q,
)
from airport_forecast.constants import CORE_AIRPORTS, SHORT_NAMES
from airport_forecast.data import load_enriched
from airport_forecast.features import build_features, temporal_train_val_test_split
from airport_forecast.models import recursive_forecast_global, train_lightgbm_global

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
CAL_END = "2023-12"
TEST_END = "2024-12"
COVERAGE = 0.80


def _train_until(enriched: pd.DataFrame, until: str):
    feat = build_features(enriched)
    feat_core = feat[feat["airport"].isin(CORE_AIRPORTS)].copy()
    train, _, _ = temporal_train_val_test_split(feat_core, until, until)
    lag_cols = [c for c in train.columns if "lag" in c or "rolling" in c]
    train_clean = train.dropna(subset=lag_cols)
    return train_lightgbm_global(train_clean, None)


def _score(test: pd.DataFrame, lower: np.ndarray, upper: np.ndarray) -> dict:
    y = test["pax_actual"].to_numpy()
    cov, width = coverage_and_width(y, lower, upper)
    per_airport = {}
    for ap in CORE_AIRPORTS:
        m = (test["airport"] == ap).to_numpy()
        if not m.any():
            continue
        c, w = coverage_and_width(y[m], lower[m], upper[m])
        per_airport[ap] = {"coverage_pct": round(c, 1), "mean_width_pax": round(w, 0)}
    return {
        "test_coverage_pct": round(cov, 1),
        "test_mean_width_pax": round(width, 0),
        "min_airport_coverage_pct": min(v["coverage_pct"] for v in per_airport.values()),
        "per_airport": per_airport,
    }


def main() -> None:
    enriched = load_enriched()

    print(f"Calibrate: train <= {CAL_END}, residuals on ({CAL_END}, {TEST_END}]")
    model_cal, fcols_cal = _train_until(enriched, CAL_END)
    fc_cal = recursive_forecast_global(
        model_cal, fcols_cal, enriched, origin_date=CAL_END, airports=CORE_AIRPORTS
    )
    cal = fc_cal.dropna(subset=["pax_actual", "pax_pred"])
    cal = cal[cal["date"] <= pd.Timestamp(TEST_END)]
    y_cal = cal["pax_actual"].to_numpy()
    p_cal = cal["pax_pred"].to_numpy()
    q_abs = split_conformal_q(y_cal - p_cal, coverage=COVERAGE)
    q_rel = split_conformal_q(relative_residuals(y_cal, p_cal), coverage=COVERAGE)
    print(f"  n_cal={len(cal)}  q_rel={q_rel:.3f}  q_abs={q_abs:,.0f} PAX")

    print(f"Test: train <= {TEST_END}, coverage on 2025+")
    model_test, fcols_test = _train_until(enriched, TEST_END)
    fc_test = recursive_forecast_global(
        model_test, fcols_test, enriched, origin_date=TEST_END, airports=CORE_AIRPORTS
    )
    test = fc_test.dropna(subset=["pax_actual", "pax_pred"]).copy()
    pred = test["pax_pred"].to_numpy()

    lower, upper = apply_relative_interval(pred, q_rel)
    rel = _score(test, lower, upper)
    abs_lower, abs_upper = apply_interval(pred, q_abs)
    absolute = _score(test, abs_lower, abs_upper)

    print(f"{'':>10s}  {'relatif':>18s}  {'absolu':>18s}")
    print(f"{'total':>10s}  {rel['test_coverage_pct']:>5.1f}%  {rel['test_mean_width_pax']:>10,.0f}"
          f"  {absolute['test_coverage_pct']:>5.1f}%  {absolute['test_mean_width_pax']:>10,.0f}")
    for ap in rel["per_airport"]:
        r, a = rel["per_airport"][ap], absolute["per_airport"][ap]
        print(f"{SHORT_NAMES[ap]:>10s}  {r['coverage_pct']:>5.0f}%  {r['mean_width_pax']:>10,.0f}"
              f"  {a['coverage_pct']:>5.0f}%  {a['mean_width_pax']:>10,.0f}")

    test["lower"] = lower
    test["upper"] = upper
    test["in_interval"] = (test["pax_actual"] >= test["lower"]) & (
        test["pax_actual"] <= test["upper"]
    )

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = test[["airport", "date", "pax_actual", "pax_pred", "lower", "upper", "in_interval"]]
    out.to_csv(REPORTS / "conformal_intervals.csv", index=False)
    summary = {
        "method": "split-conformal on recursive residuals, relative score |y-pred|/pred",
        "coverage_target": COVERAGE,
        "q_rel": round(q_rel, 4),
        "n_cal": int(len(cal)),
        "cal_origin": CAL_END,
        "cal_end": TEST_END,
        "test_origin": TEST_END,
        "n_test": int(len(test)),
        **rel,
        "absolute_baseline": {"q_pax": round(q_abs, 1), **absolute},
    }
    (REPORTS / "conformal_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("Wrote reports/conformal_intervals.csv and conformal_summary.json")


if __name__ == "__main__":
    main()
