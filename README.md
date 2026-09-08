# Airport forecasting

J’ai pris 6 aéroports européens (Eurostat) pour voir si un seul LightGBM tient partout, sans lire le futur au moment du forecast.

Le chiffre qui compte : l’éval **récursive** (`reports/horizon_results.csv`). Pas le one-step de `model_results.csv`.

[Chiffres](https://airohh.github.io/airport-forecasting/) — dashboard : `streamlit run app.py` (ou Streamlit Cloud sur `app.py`).

## Résultat

MAPE moyen.

| Horizon | LightGBM | SARIMA | Naive (même mois N-1) |
|---------|----------|--------|------------------------|
| M+1 | 3.5% | 6.0% | 6.2% |
| M+3 | 4.1% | 5.2% | 5.7% |
| M+6 | 3.8% | 6.0% | 6.3% |
| M+12 | 3.9% | 5.2% | 6.5% |

Fenêtre test complète : LightGBM 4.4%, SARIMA 5.5%.

À M+1 / M+3, LightGBM gagne en MAPE mais pas en MAE contre la naïve (MASE 1.13 / 1.22). À M+6 / M+12, MASE < 1.

Intervalles : split-conformal sur les résidus **récursifs** (calibré 2024, testé 2025). Cible 80%, couverture observée **90%**, largeur ±140k PAX. Fichier : `reports/conformal_summary.json`.

## Données

Eurostat `avia_paoa` (PAX + mouvements), chômage / PIB, pétrole FRED, change BCE, jours fériés.

## Lancer

```bash
git clone https://github.com/Airohh/airport-forecasting.git
cd airport-forecasting
pip install -e ".[all]"

python scripts/download_eurostat.py
python scripts/process_eurostat.py
python scripts/download_macro_v2.py

python scripts/evaluate_horizons.py
python scripts/evaluate_conformal.py
python scripts/save_production_model.py
uvicorn airport_forecast.api:app --reload
streamlit run app.py
```

Les CSV et le pickle sont déjà dans `reports/` et `models/` si tu veux juste regarder les scores.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"airport\": \"FR_LFLL\", \"horizon\": 6, \"model\": \"lightgbm\"}"
```

`pytest tests/ -q`

## Drift

`python scripts/auto_retrain.py` : PSI, retrain jusqu’à 2024-12. `--until all` pour tout l’historique. Pas de cron.

## Limites

- Vols futurs = même mois N-1.
- SARIMA M+12 : 4 aéroports dans le CSV (Lyon / Nantes absents).
- Intervalles volontiers larges (conformal conservateur).
- Chronos / Prophet : pas servis.
- `train_all_models.py` écrit encore le CSV one-step.

Licence [MIT](LICENSE).
