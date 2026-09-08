"""Split-conformal intervals on the recursive forecast (not one-step quantiles)."""

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


def apply_interval(pred: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred, dtype=float)
    lower = np.maximum(pred - q, 0.0)
    upper = pred + q
    return lower, upper


def coverage_and_width(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y, dtype=float)
    inside = (y >= lower) & (y <= upper)
    width = float(np.mean(upper - lower))
    return float(inside.mean() * 100), width
