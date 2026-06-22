"""Drift monitoring using Population Stability Index (PSI)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def psi(reference: np.ndarray, production: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two distributions.

    PSI < 0.1  : no drift
    0.1 <= PSI < 0.25 : moderate drift (warning)
    PSI >= 0.25 : significant drift (retrain)
    """
    eps = 1e-6
    breakpoints = np.linspace(
        min(reference.min(), production.min()) - eps,
        max(reference.max(), production.max()) + eps,
        bins + 1,
    )
    ref_counts = np.histogram(reference, bins=breakpoints)[0] / len(reference) + eps
    prod_counts = np.histogram(production, bins=breakpoints)[0] / len(production) + eps

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
