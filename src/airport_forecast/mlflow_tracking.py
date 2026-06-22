"""MLflow tracking for model training runs."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


def log_model_run(
    model_name: str,
    results: list,
    feature_importance: pd.DataFrame | None = None,
    extra_params: dict | None = None,
):
    try:
        import mlflow
    except ImportError:
        print("mlflow not installed. pip install mlflow")
        return

    runs_dir = Path(__file__).resolve().parent.parent.parent / "mlruns"
    uri = os.environ.get("MLFLOW_TRACKING_URI", runs_dir.as_uri())
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("airport-forecasting")

    with mlflow.start_run(run_name=model_name):
        params = {"model": model_name}
        if extra_params:
            params.update(extra_params)
        mlflow.log_params(params)

        mapes = [r.mape for r in results]
        maes = [r.mae for r in results]
        mlflow.log_metrics({
            "avg_mape": float(np.mean(mapes)),
            "avg_mae": float(np.mean(maes)),
            "min_mape": float(np.min(mapes)),
            "max_mape": float(np.max(mapes)),
            "n_airports": len(set(r.airport for r in results)),
        })

        if feature_importance is not None:
            mlflow.log_dict(
                dict(zip(feature_importance["feature"], feature_importance["importance"])),
                "feature_importance.json",
            )

        per_airport = {}
        for r in results:
            per_airport[f"{r.airport}_mape"] = r.mape
            per_airport[f"{r.airport}_mae"] = r.mae
        mlflow.log_metrics(per_airport)

    print(f"  MLflow: logged run '{model_name}' (uri={uri})")
