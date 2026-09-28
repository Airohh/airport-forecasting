# Airport forecasting

J’ai pris 6 aéroports européens (Eurostat) pour voir si un seul LightGBM tient partout, sans lire le futur au moment du forecast.

Le chiffre qui compte : l’éval **récursive** (`reports/horizon_results.csv`). Pas le one-step de `model_results.csv`.

[Chiffres](https://airohh.github.io/airport-forecasting/) — dashboard : `streamlit run app.py` (ou Streamlit Cloud sur `app.py`).

## Résultat

MAPE moyen sur les h premiers mois après l’origine (entraînement jusqu’à 2024-12).

| Horizon | LightGBM | SARIMA | Naive (même mois N-1) |
|---------|----------|--------|------------------------|
| M+1 | 3.5% | 5.7% | 6.2% |
| M+3 | 4.1% | 4.8% | 5.7% |
| M+6 | 3.8% | 5.6% | 6.3% |
| M+12 | 3.3% | 4.6% | 6.5% |

Chaque ligne compare les trois modèles sur les mêmes aéroports. À M+12 il n’en reste que 4 : les données de Lyon et Nantes s’arrêtent en 2025-11 (11 mois de test).

À M+1 / M+3, LightGBM gagne en MAPE mais pas en MAE contre la naïve (MASE 1.13 / 1.22). À M+6 / M+12, MASE < 1 (0.66 / 0.61).

Intervalles 80% : split-conformal sur les résidus **récursifs**, score relatif |y − ŷ| / ŷ. Calibré sur 2024 uniquement (modèle entraîné jusqu’à 2023-12), testé sur 2025+ (modèle entraîné jusqu’à 2024-12). Bande ŷ × (1 ± 7.8%) : couverture **88%**, au moins 80% sur chaque aéroport, largeur moyenne 219k PAX. L’API renvoie ces bornes (`pax_lower`, `pax_upper`). Fichier : `reports/conformal_summary.json`.

Même calibrage avec un score absolu (± 127k PAX pour tous) : 90% au total, mais 67% à Lisbonne (~3M PAX/mois) et 100% à Nantes (~0.6M). Détail par aéroport dans le JSON (`absolute_baseline`).

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
- M+12 : 4 aéroports seulement (Lyon / Nantes absents, données jusqu’à 2025-11).
- Intervalles : 72 points de calibration, 73 de test. La couverture par aéroport repose sur 11 à 15 mois, donc elle est bruitée.
- CV 3 plis (`cv_results.csv`) : les hyperparamètres ont été tunés sur 2024, qui est aussi le test du pli 2. Ce pli est optimiste ; le chiffre propre est le pli 3 (tableau ci-dessus).
- Chronos / Prophet : pas servis.
- `train_all_models.py` écrit encore le CSV one-step.

Licence [MIT](LICENSE).
