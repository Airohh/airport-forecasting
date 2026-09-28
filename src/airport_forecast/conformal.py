"""Split-conformal intervals on the recursive forecast (not one-step quantiles).

Two score functions:
- absolute: |y - ŷ| in PAX → interval ŷ ± q. One width for every airport.
- relative: |y - ŷ| / ŷ → interval ŷ·(1 ± q). Width scales with traffic, so
  Lisbon (~3M PAX/month) and Nantes (~0.6M) get comparable coverage.
"""

from __future__ import annotations

import numpy as np


def split_conformal_q(residuals: np.ndarray, coverage: float = 0.80) -> float:
    """Absolute residual quantile with the finite-sample conformal correction.

    q = sorted(|r|)[ceil((n+1)*coverage) - 1]
    """
    scores = np.abs(np.asarray(residuals, dtype=float))
    scores = scores[np.isfinite(scores)]
    n = len(scores)
    if n == 0:
        raise ValueError("no residuals")
    k = int(np.ceil((n + 1) * coverage))
    k = min(max(k, 1), n)
    return float(np.sort(scores)[k - 1])


def relative_residuals(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """(y - ŷ) / ŷ, NaN where ŷ <= 0 (split_conformal_q drops those)."""
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    out = np.full_like(pred, np.nan)
    ok = pred > 0
    out[ok] = (y[ok] - pred[ok]) / pred[ok]
    return out


def apply_interval(pred: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred, dtype=float)
    lower = np.maximum(pred - q, 0.0)
    upper = pred + q
    return lower, upper


def apply_relative_interval(pred: np.ndarray, q_rel: float) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred, dtype=float)
    lower = np.maximum(pred * (1.0 - q_rel), 0.0)
    upper = pred * (1.0 + q_rel)
    return lower, upper


def coverage_and_width(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y, dtype=float)
    inside = (y >= lower) & (y <= upper)
    width = float(np.mean(upper - lower))
    return float(inside.mean() * 100), width
