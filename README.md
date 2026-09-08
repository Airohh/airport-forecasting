# Airport forecasting

J’ai pris 6 aéroports européens (Eurostat) pour voir si un seul LightGBM tient partout, sans lire le futur au moment du forecast.

Le chiffre qui compte : l’éval **récursive** (`reports/horizon_results.csv`). Pas le one-step de `model_results.csv` (lags du test, trop gentil).

## Résultat

MAPE moyen, 6 aéroports (Lyon, Nantes, Budapest, Lisbonne, Porto, Belgrade).

| Horizon | LightGBM | SARIMA |
|---------|----------|--------|
| M+1 | 3.5% | 6.0% |
| M+3 | 4.1% | 5.2% |
| M+6 | 3.8% | 6.0% |
| M+12 | 3.9% | 5.2% |

Fenêtre test complète : LightGBM 4.4%, SARIMA 5.5%.

Les intervalles nominaux 80% ne couvrent que ~52% — et ils sont calculés en one-step, pas en récursif. Le point forecast est le claim ; l’incertitude pas encore.

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
python scripts/save_production_model.py
uvicorn airport_forecast.api:app --reload
streamlit run src/airport_forecast/dashboard.py
```

Les CSV et le pickle sont déjà dans `reports/` et `models/` si tu veux juste regarder les scores.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"airport\": \"FR_LFLL\", \"horizon\": 6, \"model\": \"lightgbm\"}"
```

`/models/{airport}/metrics` lit `horizon_results.csv` (récursif).

`pytest tests/ -q` — dont `test_recursive_honesty.py` (les vols / PAX futurs ne doivent pas fuiter).

## Drift

`python scripts/auto_retrain.py` : PSI, puis retrain **jusqu’à 2024-12** (même cutoff que le backtest). `--until all` pour tout l’historique. Pas de cron.

## Limites

- Vols futurs = même mois N-1.
- SARIMA M+12 : 4 aéroports dans le CSV (Lyon / Nantes absents).
- Chronos / Prophet : scripts de comparaison, pas servis.
- `train_all_models.py` écrit encore le CSV one-step. Ne pas s’en servir pour le pitch.

Licence [MIT](LICENSE).
