# Airport forecasting

J’ai pris 6 aéroports européens (Eurostat) pour voir si un seul LightGBM tient partout, sans lire le futur au moment du forecast. Comparaison avec SARIMA, Prophet et Chronos.

Les chiffres sont dans `reports/horizon_results.csv`.

## Résultat

MAPE, moyenne sur les 6 aéroports (Lyon, Nantes, Budapest, Lisbonne, Porto, Belgrade).

| Horizon | LightGBM | SARIMA |
|---------|----------|--------|
| M+1 | 3.5% | 6.0% |
| M+3 | 4.1% | 5.2% |
| M+6 | 3.8% | 6.0% |
| M+12 | 3.9% | 5.2% |

Sur la fenêtre test complète : LightGBM 4.4%, SARIMA 5.5%.

Les intervalles nominaux 80% ne couvrent que ~52% des points. Le point forecast tient ; l’incertitude pas encore.

## Données

Eurostat `avia_paoa` (PAX + mouvements), chômage / PIB Eurostat, pétrole FRED, change BCE, jours fériés. Features : lags, rolling, saison, macro, flags (COVID, etc.).

## Lancer

```bash
git clone https://github.com/Airohh/airport-forecasting.git
cd airport-forecasting
pip install -e ".[all]"

python scripts/download_eurostat.py
python scripts/process_eurostat.py
python scripts/download_macro_v2.py

python scripts/train_all_models.py
uvicorn airport_forecast.api:app --reload
streamlit run src/airport_forecast/dashboard.py
```

Les CSV et le pickle LightGBM sont déjà dans `reports/` et `models/` si tu veux juste regarder les scores.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"airport\": \"FR_LFLL\", \"horizon\": 6, \"model\": \"lightgbm\"}"
```

`pytest tests/ -q` — data, features, modèles, PSI.

## Drift

`python scripts/auto_retrain.py` compare une fenêtre récente au passé (PSI sur des ratios / saison / macro, pas les niveaux de trafic). Si ça passe le seuil : réentraîne, remplace `models/lightgbm_global.pkl`, garde un `.bak`. Pas de cron.

## Limites

- Les vols futurs sont remplacés par le même mois N-1. En vrai on aurait les programmes compagnies.
- Pas d’Airflow / Kubeflow.
- Chronos varie beaucoup d’un aéroport à l’autre. Ce n’est pas le modèle servi.

Licence [MIT](LICENSE).
