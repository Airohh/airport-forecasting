# HANDOFF — Airport PAX Forecasting

État du projet au 2026-06-22. À lire pour reprendre le travail.

## But

Projet portfolio pour candidature **Data Scientist Junior — VINCI Airports** (réf 2026-127857, Nanterre, CDI). Prévision du trafic passagers mensuel sur 6 aéroports du réseau VINCI. Démontre forecasting + MLOps pour matcher l'offre (axe "ML & Forecasting" du Smart Data Hub).

Localisation : `C:\Users\sampo\OneDrive\Desktop\TRANSFERT_TURQUIE\TRAVAIL\Contexte\projets\airport-forecasting`
Dossier candidature : `TRAVAIL\candidatures\VINCI Airports` (lettre + CV + offre)

## Données

- **Source PAX** : Eurostat `avia_paoa`, mensuel 1998-2026
- **6 aéroports core** : Lyon (FR_LFLL), Nantes (FR_LFRS), Budapest (HU_LHBP), Lisbon (PT_LPPT), Porto (PT_LPPR), Belgrade (RS_LYBE)
- **2 exclus** (data stop 2020, Brexit) : Gatwick, Edinburgh — gardés en EDA seulement
- **Enrichissement macro** : chômage (Eurostat), PIB (Eurostat), Brent oil (FRED), taux change EUR/HUF EUR/GBP (ECB), holidays (package), événements manuels (COVID, guerre Ukraine, JO 2024, Web Summit)
- Dataset final : `data/processed/pax_enriched.parquet` — 1988 rows, 14 cols
- **33 features** après `build_features` : lags(1,2,3,6,12), rolling mean/std(3,6,12), YoY (lagged), calendrier cyclique, macro, événements, network (market share, rank), airline supply (n_flights, pax_per_flight — lagged)

## Modèles (5)

SARIMA, LightGBM Global, LightGBM Local, Prophet, Chronos (Amazon zero-shot), + Ensemble.

## Résultats honnêtes (post-fix leakage + recursive)

### Par horizon (LightGBM Recursive vs SARIMA)

| Horizon | LGB Recursive | SARIMA | Usage |
|---------|--------------|--------|-------|
| M+1 | **2.9%** | 6.0% | staffing, gates |
| M+3 | **4.2%** | 5.2% | capacity planning |
| M+6 | **3.7%** | 6.0% | route planning |
| M+12 | 6.3% | **5.2%** | budget, contracts |

Point clé : LightGBM domine M+1 à M+6, SARIMA gagne M+12 (error accumulation récursive).

### Tous modèles (test 2025+, one-step)

| Model | Avg MAPE |
|-------|----------|
| LightGBM Global | 4.4% |
| SARIMA | 5.5% |
| LightGBM Local | 7.4% |
| Chronos | 11.0% |
| Prophet | 17.9% |

Optuna : 100 trials, best params dans `reports/best_params.json`, val MAPE tuné = 4.19%.

### Bugs corrigés

1. **Target leakage** : `pax_yoy_growth` utilisait `pax` brut (la cible). Corrigé avec `shift(1) - shift(13)`.
2. **API backcasting** : `/predict` retournait des valeurs historiques. Corrigé avec `recursive_forecast_global`.
3. **Test mort** : `test_no_leakage_in_rolling` avait `or True`. Corrigé.
4. **Dead code** : `holidays_features.py`, `download_macro.py` (v1), `eda.py`, `eda_advanced.py` supprimés.

## Structure

```
src/airport_forecast/  : api.py, constants.py, dashboard.py, data.py, features.py,
                         logging_config.py, mlflow_tracking.py,
                         models.py, monitoring.py (PSI drift)
scripts/               : download_eurostat, process_eurostat, download_macro_v2,
                         eda_full, train_all_models, train_chronos,
                         tune_lightgbm, compare_recursive, evaluate_horizons
tests/                 : 30 tests (data, features, models, API, monitoring) — tous pass
reports/figures/       : 25+ plots EDA
.meta/graphify/        : knowledge graphs (src, scripts, tests, full)
Dockerfile, docker-compose.yml, .github/workflows/ci.yml, README.md (Mermaid)
```

## Git

15 commits conventional. Repo local, PAS encore push GitHub.

## TODO restant

1. ~~Commit le forecasting récursif~~ ✅
2. ~~Éval par horizon (M+1, M+3, M+6, M+12)~~ ✅
3. ~~README avec chiffres honnêtes~~ ✅
4. Push GitHub (`git remote add origin` + push)
5. Optionnel : log-transform + growth_acceleration pour améliorer Porto M+12
6. Optionnel : Kubeflow (gap technique de l'offre)
7. Lire rapport annuel VINCI Airports avant entretien
8. Mettre à jour lettre de motivation avec ce projet

## Env technique

Windows, Python 3.14, packages installés : eurostat, pandas, lightgbm, statsmodels,
prophet, chronos-forecasting, torch (CPU), fastapi, streamlit, optuna, mlflow, pytest, ruff.
