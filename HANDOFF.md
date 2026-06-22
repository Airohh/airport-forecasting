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

## Résultats honnêtes (post-fix leakage + recursive + exog honnête)

**Régime exog honnête** : au moment du forecast, les exogènes futurs NE sont PAS pris dans les actuals. `assume_future_exog` (models.py) remplace `n_flights`/`pax_per_flight`/calendrier par seasonal-naive (même mois N-1), et carry-forward pour macro + flags événements. Même régime en eval, tuning ET API servie. **Params Optuna (best_params.json) chargés par `train_lightgbm_global`** → eval/serve/tune partagent UNE config.

### Par horizon (LightGBM Recursive vs SARIMA) — primary fold, head(h), tuned

| Horizon | LGB Recursive | SARIMA | Usage |
|---------|--------------|--------|-------|
| M+1 | **3.5%** | 6.0% | staffing, gates |
| M+3 | **4.1%** | 5.2% | capacity planning |
| M+6 | **3.8%** | 6.0% | route planning |
| M+12 | **3.9%** | 5.2% | budget, contracts |

Point clé : **LightGBM bat SARIMA à TOUS les horizons, M+12 inclus (3.9 vs 5.2)**. Le tuning sur l'objectif recursive honnête (397 arbres, depth 7, lr 0.017, régularisé) tue l'accumulation d'erreur long horizon.

### Full horizon honnête (compare_recursive, fenêtre test complète par aéroport)

| Model | Avg MAPE |
|-------|----------|
| LGB recursive (tuned) | **4.4%** |
| SARIMA | 5.5% |
| LGB 1-step (optimiste) | 4.7% |

LGB recursive honnête+tuned **bat SARIMA** (4.4 vs 5.5), aucun effondrement (Budapest 6.2%, Porto 3.2%, Lisbon 3.0%). ⚠️ Avec les params DEFAULT (tuné one-step), Budapest/Porto explosaient (14.7%/13.3%) — **le tuning sur objectif recursive a corrigé ça, pas un changement de features**. Leçon entretien : tune dans le régime que tu sers.

### Tous modèles (test 2025+, one-step) — inchangé (non exog-dépendant)

| Model | Avg MAPE |
|-------|----------|
| LightGBM Global | 4.4% |
| SARIMA | 5.5% |
| LightGBM Local | 7.4% |
| Chronos | 11.0% |
| Prophet | 17.9% |

Optuna : 100 trials sur **MAPE recursive honnête**. `best_params.json` : val 5.61%, test recursive 4.50%. Chargé automatiquement par `train_lightgbm_global` (relancer `tune_lightgbm.py` regénère).

### Bugs corrigés

1. **Target leakage** : `pax_yoy_growth` utilisait `pax` brut (la cible). Corrigé avec `shift(1) - shift(13)`.
2. **API backcasting** : `/predict` retournait des valeurs historiques. Premier "fix" (recursive avec origin = last-horizon) re-prédisait quand même l'historique. **Vrai fix** : `forecast_future_global` + `make_future_enriched` ajoutent des lignes futures réelles → forecast post end-of-data.
3. **Test mort** : `test_no_leakage_in_rolling` avait `or True`. Corrigé.
4. **Dead code** : `holidays_features.py`, `download_macro.py` (v1), `eda.py`, `eda_advanced.py` supprimés.
5. **Exog leakage recursive** : eval/tune utilisaient les vrais n_flights/macro futurs → chiffres gonflés. Corrigé via `assume_future_exog` (seasonal-naive + carry), `recursive_forecast_global(honest_exog=True)`. Régime unique eval/tune/serve.
6. **PSI non-standard** : bins équi-largeur + production dans les bords → drift masqué. Corrigé en bins quantiles de la référence, bords ±inf (`monitoring.psi`).
7. **Tune mauvais régime** : `tune_lightgbm.py` optimisait one-step ; maintenant recursive honnête.
8. **Params non câblés** : seul tune utilisait Optuna, eval/serve restaient en defaults. `train_lightgbm_global` charge maintenant `best_params.json` (`_load_best_params`). `model.pkl` réentraîné (397 arbres). Une seule config partout.

## Auto-retrain (PSI → trigger) — câblé

`scripts/auto_retrain.py` ferme la boucle monitoring (avant : design only, flèche "manual").

- Compare fenêtre **production** récente (`--prod-months 12`) vs **référence** récente bornée (`--ref-months 36`) — PAS tout l'historique 1998+ (sinon drift trivial sur croissance trafic).
- Règle déclenchement `monitoring.should_retrain` : 1 feature CRITICAL (PSI≥0.25) OU ≥3 WARNING. Conservateur : 1 warning isolé = bruit.
- Si trigger → réentraîne LightGBM global sur TOUTES les données (charge `best_params.json`), **swap atomique** du `models/lightgbm_global.pkl` (ancien → `.pkl.bak` pour rollback). Log append `reports/retrain_log.jsonl`, rapport `reports/drift_report.csv`.
- **PSI seulement sur features stationnaires** (`MONITOR_FEATURES`) : ratios (market share, load factor), croissance (yoy), forme saisonnière (sin/cos), régime macro (oil, FX, chômage). EXCLUT niveaux monotones (lags, rolling means, totals, gdp) — montent avec le trafic, flag chaque mois sans signal de casse modèle. Argument méthodo entretien : PSI a du sens sur stationnaire, pas sur série tendancielle.
- Flags : `--check-only` (rapport, jamais retrain), `--force` (retrain forcé). Exit 0 = pas de retrain, 1 = retrained.
- Drift réel actuel détecté : oil 85→67, yoy growth 0.74→0.06 (fin reprise COVID), load factor monté → 4 CRITICAL.
- Unité qu'un scheduler (cron / Airflow / Kubeflow, TODO #9) appellerait.

## Structure

```
src/airport_forecast/  : api.py, constants.py, dashboard.py, data.py, features.py,
                         logging_config.py, mlflow_tracking.py,
                         models.py, monitoring.py (PSI drift + should_retrain)
scripts/               : download_eurostat, process_eurostat, download_macro_v2,
                         eda_full, train_all_models, train_chronos,
                         tune_lightgbm, compare_recursive, evaluate_horizons,
                         auto_retrain (PSI → retrain)
tests/                 : 38 tests (data, features, models, API, monitoring) — tous pass
reports/figures/       : 25+ plots EDA
.meta/graphify/        : knowledge graphs (src, scripts, tests, full)
Dockerfile, docker-compose.yml, .github/workflows/ci.yml, README.md (Mermaid)
```

## Git

Commits conventional. Pushé sur `github.com/Airohh/airport-forecasting` (master). Dernier : `feat: wire PSI drift detection to auto-retrain trigger`.

## TODO restant

1. ~~Commit le forecasting récursif~~ ✅
2. ~~Éval par horizon (M+1, M+3, M+6, M+12)~~ ✅
3. ~~README avec chiffres honnêtes~~ ✅
4. ~~Relancer tune (recursive honnête) → best_params.json~~ ✅ val 5.61% / test 4.50%
5. ~~Relancer evaluate_horizons + compare_recursive, figer CSV~~ ✅ (tuned)
6. ~~Câbler auto-retrain (PSI → trigger)~~ ✅ `scripts/auto_retrain.py` (voir section dédiée)
7. Optionnel : proxy n_flights meilleur (forward schedule OAG) ou two-stage flight-count forecast pour pousser encore
8. ~~Push GitHub~~ ✅ `github.com/Airohh/airport-forecasting` (remote existait déjà)
9. Optionnel : Kubeflow (gap technique de l'offre)
10. Lire rapport annuel VINCI Airports avant entretien
11. MAJ lettre de motivation avec ce projet (insister sur l'honnêteté méthodo : c'est le différenciateur)

## Env technique

Windows, Python 3.14, packages installés : eurostat, pandas, lightgbm, statsmodels,
prophet, chronos-forecasting, torch (CPU), fastapi, streamlit, optuna, mlflow, pytest, ruff.
