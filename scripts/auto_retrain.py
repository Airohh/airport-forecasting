"""Auto-retrain orchestrator: PSI drift check -> conditional model retrain.

Wires the "PSI Drift Detection -> Retrain trigger" arrow from the architecture
diagram. Run on a schedule (cron / Airflow / Kubeflow). Compares the recent
production window against the training reference; if features have drifted past
the trigger rule (see monitoring.should_retrain), retrains the global LightGBM
on all available data and atomically swaps models/lightgbm_global.pkl.

Usage:
    python scripts/auto_retrain.py                  # check + retrain if drifted
    python scripts/auto_retrain.py --check-only     # report drift, never retrain
    python scripts/auto_retrain.py --force          # retrain regardless of drift
    python scripts/auto_retrain.py --prod-months 6  # production window size

Exit code 0 = no retrain needed, 1 = retrained, 2 = drift check failed.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from airport_forecast.constants import CORE_AIRPORTS
from airport_forecast.data import load_enriched
from airport_forecast.features import build_features
from airport_forecast.models import train_lightgbm_global
from airport_forecast.monitoring import monitor_drift, should_retrain

# PSI is only meaningful on roughly stationary features. Raw level features
# (pax lags, rolling means, gdp, network/country totals) trend upward as the
# network recovers and grows post-COVID, so PSI flags them every month — but a
# rising level is expected and already handled by the lag-based model, not the
# kind of drift that breaks it. The drift that actually degrades the model lives
# in ratios and relationships: load factor, market share, YoY growth, seasonality
# shape, and macro regime (oil, FX, unemployment rates). We monitor those.
MONITOR_FEATURES = [
    "pax_yoy_growth",          # demand growth regime
    "pax_per_flight_lag1",     # load factor (airline supply relationship)
    "country_market_share",    # network position
    "network_rank",
    "month_sin", "month_cos", "is_summer",  # seasonality shape
    "n_holidays", "is_school_vacation",      # calendar intensity
    "unemployment_rate", "oil_price_usd", "exchange_rate",  # macro regime (levels that signal real shifts)
]

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "lightgbm_global.pkl"
REPORTS_DIR = ROOT / "reports"
DRIFT_REPORT = REPORTS_DIR / "drift_report.csv"
RETRAIN_LOG = REPORTS_DIR / "retrain_log.jsonl"


def _split_reference_production(feat: pd.DataFrame, prod_months: int, ref_months: int):
    """Production = last `prod_months`; reference = the `ref_months` window just before it.

    Reference is bounded (not the full 1998+ history): comparing a recent prod
    window against decades of aviation growth would flag drift trivially. We ask
    "did the distribution shift versus the recent past the model learned from?"
    """
    prod_cutoff = feat["date"].max() - pd.DateOffset(months=prod_months)
    ref_cutoff = prod_cutoff - pd.DateOffset(months=ref_months)
    reference = feat[(feat["date"] > ref_cutoff) & (feat["date"] <= prod_cutoff)]
    production = feat[feat["date"] > prod_cutoff]
    return reference, production


def _retrain_and_swap(feat_core: pd.DataFrame) -> dict:
    """Retrain global LightGBM on ALL available data, atomic-swap the served model.

    Old model is backed up to .pkl.bak so a bad retrain can be rolled back.
    """
    lag_cols = [c for c in feat_core.columns if "lag" in c or "rolling" in c]
    train_clean = feat_core.dropna(subset=lag_cols)
    model, fcols = train_lightgbm_global(train_clean)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        shutil.copy2(MODEL_PATH, MODEL_PATH.with_suffix(".pkl.bak"))
    tmp = MODEL_PATH.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump({"model": model, "feature_cols": fcols}, f)
    tmp.replace(MODEL_PATH)
    return {"n_train_rows": int(len(train_clean)), "n_features": len(fcols)}


def main() -> int:
    parser = argparse.ArgumentParser(description="PSI drift check + conditional retrain")
    parser.add_argument("--check-only", action="store_true", help="report drift, never retrain")
    parser.add_argument("--force", action="store_true", help="retrain regardless of drift")
    parser.add_argument("--prod-months", type=int, default=12, help="production window (months)")
    parser.add_argument("--ref-months", type=int, default=36, help="reference window before prod (months)")
    parser.add_argument("--n-warning", type=int, default=3, help="WARNING count that triggers retrain")
    args = parser.parse_args()

    feat = build_features(load_enriched())
    feat_core = feat[feat["airport"].isin(CORE_AIRPORTS)].copy()
    feature_cols = [c for c in MONITOR_FEATURES if c in feat_core.columns]

    reference, production = _split_reference_production(
        feat_core, args.prod_months, args.ref_months
    )
    report = monitor_drift(reference, production, feature_cols)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(DRIFT_REPORT, index=False)

    decision = should_retrain(report, n_warning_to_retrain=args.n_warning)
    ref_end = reference["date"].max()
    prod_end = production["date"].max()

    print(f"Drift check  ref<= {ref_end:%Y-%m}  prod {args.prod_months}mo -> {prod_end:%Y-%m}")
    print(f"  features compared: {len(report)}  "
          f"critical: {decision['n_critical']}  warning: {decision['n_warning']}")
    if decision["drivers"]:
        print(f"  drivers: {', '.join(decision['drivers'][:8])}")
    print(f"  decision: {decision['reason']}")
    print(f"  drift report -> {DRIFT_REPORT.relative_to(ROOT)}")

    do_retrain = args.force or (decision["retrain"] and not args.check_only)

    log = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "ref_end": f"{ref_end:%Y-%m}",
        "prod_end": f"{prod_end:%Y-%m}",
        "prod_months": args.prod_months,
        "n_critical": decision["n_critical"],
        "n_warning": decision["n_warning"],
        "decision_reason": decision["reason"],
        "drivers": decision["drivers"],
        "forced": args.force,
        "check_only": args.check_only,
        "retrained": do_retrain,
    }

    if do_retrain:
        info = _retrain_and_swap(feat_core)
        log.update(info)
        print(f"  RETRAINED on {info['n_train_rows']} rows, "
              f"{info['n_features']} features -> {MODEL_PATH.relative_to(ROOT)} "
              f"(backup: lightgbm_global.pkl.bak)")
    elif decision["retrain"] and args.check_only:
        print("  retrain WOULD trigger, but --check-only set; model unchanged")
    else:
        print("  no retrain")

    with open(RETRAIN_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(log) + "\n")

    return 1 if do_retrain else 0


if __name__ == "__main__":
    raise SystemExit(main())
