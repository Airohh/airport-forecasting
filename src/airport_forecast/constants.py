"""Airport PAX forecasting — shared constants."""

EUROSTAT_DATASET = "avia_paoa"

VINCI_AIRPORTS: dict[str, str] = {
    "FR_LFLL": "Lyon Saint-Exupéry",
    "FR_LFRS": "Nantes Atlantique",
    "UK_EGKK": "London Gatwick",
    "HU_LHBP": "Budapest",
    "PT_LPPT": "Lisbon",
    "PT_LPPR": "Porto",
    "RS_LYBE": "Belgrade",
    "UK_EGPH": "Edinburgh",
}

AIRPORT_COUNTRIES: dict[str, str] = {
    "FR_LFLL": "FR",
    "FR_LFRS": "FR",
    "UK_EGKK": "GB",
    "HU_LHBP": "HU",
    "PT_LPPT": "PT",
    "PT_LPPR": "PT",
    "RS_LYBE": "RS",
    "UK_EGPH": "GB",
}

SHORT_NAMES: dict[str, str] = {
    "FR_LFLL": "Lyon",
    "FR_LFRS": "Nantes",
    "HU_LHBP": "Budapest",
    "PT_LPPT": "Lisbon",
    "PT_LPPR": "Porto",
    "RS_LYBE": "Belgrade",
}

CORE_AIRPORTS: list[str] = list(SHORT_NAMES)

HORIZONS = [1, 3, 6, 12]

TRAIN_END = "2023-12"
VAL_END = "2024-12"

RANDOM_STATE = 42
