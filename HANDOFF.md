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
- **37 features** après `build_features` : lags(1,2,3,6,12), rolling mean/std(3,6,12), YoY, calendrier cyclique, macro, événements, network (market share, rank)

## Modèles (5)

SARIMA, LightGBM Global, LightGBM Local, Prophet, Chronos (Amazon zero-shot), + Ensemble.

## ⚠️ FINDING CRITIQUE — biais d'évaluation (RÉSOLU)

Le LightGBM original faisait du **one-step-ahead avec lags réels** (utilise le vrai mois précédent), PAS de la vraie prévision multi-step comme SARIMA/Prophet/Chronos. Comparaison injuste.

**Correction implémentée** : `recursive_forecast_global()` + `evaluate_lightgbm_recursive()` dans `models.py`. Prédit mois par mois, réinjecte chaque prédiction comme lag.

### Résultats HONNÊTES (test 2025+, par horizon)

| Horizon | Méthode | MAPE | Usage |
|---------|---------|------|-------|
| Court terme M+1 | LightGBM one-step (lags réels = légitime à h=1) | ~2.6% (tuné) | staffing, gates |
| Long terme M+12 | LightGBM récursif | 6.5% | budget |
| Long terme M+12 | SARIMA | 5.5% | budget |

Point clé : à **M+1 le 2.6% est honnête** (on connaît vraiment le mois dernier). À M+12 l'erreur s'accumule. **SARIMA bat LightGBM récursif en long terme** (5.5% vs 6.5%). Budapest récursif = 15.2% car croissance +20%/an fait diverger.

Optuna : 100 trials, best params dans `reports/best_params.json`, M+1 tuné = 2.6%.

## Structure

```
src/airport_forecast/  : api.py, constants.py, dashboard.py, data.py, features.py,
                         holidays_features.py, logging_config.py, mlflow_tracking.py,
                         models.py, monitoring.py (PSI drift)
scripts/               : download_eurostat, process_eurostat, download_macro_v2,
                         eda_full, eda_advanced, train_all_models, train_chronos,
                         tune_lightgbm, compare_recursive
tests/                 : 30 tests (data, features, models, API, monitoring) — tous pass
reports/figures/       : 33 plots EDA
Dockerfile, docker-compose.yml, .github/workflows/ci.yml, README.md (Mermaid)
```

## Git

11 commits conventional (init/feat/test/ci/data). Repo local, PAS encore push GitHub.
`recursive_forecast` + `compare_recursive.py` + ce HANDOFF.md = changements NON commités.

## TODO restant

1. Commit le forecasting récursif (changement non commité)
2. Éval propre PAR HORIZON (M+1, M+3, M+6, M+12 séparés) — script à faire
3. Mettre à jour README avec les chiffres honnêtes (le README montre encore 2.6% sans nuance horizon)
4. Push GitHub (`git remote add origin` + push)
5. Optionnel : log-transform + growth_acceleration pour améliorer Budapest
6. Optionnel : Kubeflow (gap technique de l'offre)
7. Lire rapport annuel VINCI Airports avant entretien
8. Mettre à jour lettre de motivation avec ce projet

## Env technique

Windows, Python 3.14, packages installés : eurostat, pandas, lightgbm, statsmodels,
prophet, chronos-forecasting, torch (CPU), fastapi, streamlit, optuna, mlflow, pytest, ruff.
