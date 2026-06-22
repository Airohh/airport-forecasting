"""Drift monitoring using Population Stability Index (PSI)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def psi(reference: np.ndarray, production: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two distributions.

    PSI < 0.1  : no drift
    0.1 <= PSI < 0.25 : moderate drift (warning)
    PSI >= 0.25 : significant drift (retrain)

    Bin edges are reference quantiles (standard PSI). Production never defines
    the edges — outliers fall into the open end bins (+/-inf), so a new extreme
    production value cannot silently reshape the bins and mask drift.
    """
    eps = 1e-6
    ref = np.asarray(reference, dtype=float)
    prod = np.asarray(production, dtype=float)

    # Quantile (equal-frequency) breakpoints from the reference only.
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        # Near-constant reference: fall back to equal-width on reference range.
        edges = np.linspace(ref.min(), ref.max(), bins + 1)
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts = np.histogram(ref, bins=edges)[0] / len(ref) + eps
    prod_counts = np.histogram(prod, bins=edges)[0] / len(prod) + eps

    return float(np.sum((prod_counts - ref_counts) * np.log(prod_counts / ref_counts)))


def monitor_drift(
    train_df: pd.DataFrame,
    prod_df: pd.DataFrame,
    feature_cols: list[str],
    psi_threshold_warning: float = 0.1,
    psi_threshold_critical: float = 0.25,
) -> pd.DataFrame:
    """Compute PSI for each feature between train and production data."""
    results = []
    for col in feature_cols:
        ref = train_df[col].dropna().values.astype(float)
        prod = prod_df[col].dropna().values.astype(float)

        if len(ref) < 10 or len(prod) < 5:
            continue

        score = psi(ref, prod)
        if score >= psi_threshold_critical:
            status = "CRITICAL"
        elif score >= psi_threshold_warning:
            status = "WARNING"
        else:
            status = "OK"

        results.append({
            "feature": col,
            "psi": round(score, 4),
            "status": status,
            "ref_mean": round(float(ref.mean()), 2),
            "prod_mean": round(float(prod.mean()), 2),
            "ref_std": round(float(ref.std()), 2),
            "prod_std": round(float(prod.std()), 2),
        })

    return pd.DataFrame(results).sort_values("psi", ascending=False).reset_index(drop=True)


def should_retrain(
    drift_report: pd.DataFrame,
    n_warning_to_retrain: int = 3,
) -> dict:
    """Decide whether drift warrants a retrain from a `monitor_drift` report.

    Trigger rule (conservative, auditable):
      - ANY feature CRITICAL (PSI >= 0.25)  -> retrain
      - OR >= `n_warning_to_retrain` features WARNING (0.1 <= PSI < 0.25) -> retrain

    A single moderate warning is noise (seasonality, one new event); a cluster of
    warnings or one hard break is a real distribution shift.
    """
    if drift_report.empty:
        return {"retrain": False, "reason": "no comparable features", "n_critical": 0, "n_warning": 0, "drivers": []}

    n_critical = int((drift_report["status"] == "CRITICAL").sum())
    n_warning = int((drift_report["status"] == "WARNING").sum())
    critical = drift_report[drift_report["status"] == "CRITICAL"]["feature"].tolist()
    warning = drift_report[drift_report["status"] == "WARNING"]["feature"].tolist()

    if n_critical > 0:
        return {
            "retrain": True,
            "reason": f"{n_critical} feature(s) CRITICAL (PSI>=0.25)",
            "n_critical": n_critical, "n_warning": n_warning, "drivers": critical,
        }
    if n_warning >= n_warning_to_retrain:
        return {
            "retrain": True,
            "reason": f"{n_warning} feature(s) WARNING (>= {n_warning_to_retrain} threshold)",
            "n_critical": 0, "n_warning": n_warning, "drivers": warning,
        }
    return {
        "retrain": False,
        "reason": f"{n_warning} warning(s), 0 critical — below trigger",
        "n_critical": 0, "n_warning": n_warning, "drivers": warning,
    }


def check_prediction_drift(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mae_threshold: float | None = None,
) -> dict:
    """Check if model predictions have degraded."""
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100)
    bias = float(np.mean(y_pred - y_true))

    status = "OK"
    if mae_threshold is not None and mae > mae_threshold:
        status = "DEGRADED"

    return {
        "mae": round(mae, 0),
        "mape": round(mape, 2),
        "bias": round(bias, 0),
        "status": status,
    }
