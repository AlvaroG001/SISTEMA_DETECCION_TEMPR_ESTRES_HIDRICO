from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PLOTS_DIR = OUTPUTS_DIR / "plots"

RAW_DATA_FILE = RAW_DIR / "sentinel_stress_by_parcel_20160101_to_20260510.csv"
RAW_COORDS_FILE = RAW_DIR / "Jerovia - Parcelas - Coordenadas (1).xlsx"
PROCESSED_DATA_FILE = PROCESSED_DIR / "dataset_modeling.csv"
PARCELS_GEOJSON_FILE = PROCESSED_DIR / "parcels.geojson"
GEOJSON_WARNING_FILE = PROCESSED_DIR / "parcels_geojson_warning.txt"

TARGET = "target_stress_7d"
TARGET_DATE = "target_date"
HORIZON_DAYS = 7
TARGET_TOLERANCE_DAYS = 14
RANDOM_STATE = 42

MODEL_SCRIPTS = [
    "train_random_forest.py",
    "train_xgboost.py",
    "train_lstm.py",
    "train_gru.py",
    "train_cnn_lstm.py",
    "train_convlstm.py",
    "train_chronos_bolt.py",
    "train_tft.py",
    "train_tcn.py",
    "train_prophet.py",
    "train_sarimax.py",
]


def ensure_directories() -> None:
    for path in [
        RAW_DIR,
        PROCESSED_DIR,
        MODELS_DIR,
        PREDICTIONS_DIR,
        METRICS_DIR,
        PLOTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
