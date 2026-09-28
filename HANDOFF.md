# Notes

Projet perso. 6 aéroports, Eurostat. Claim = éval récursive (`horizon_results.csv`).

Intervalles : `python scripts/evaluate_conformal.py` → `reports/conformal_summary.json` (score relatif, calibration coupée à 2024-12).
Pickle backtest : `python scripts/save_production_model.py`.
Drift : `python scripts/auto_retrain.py` (cutoff VAL_END par défaut).

