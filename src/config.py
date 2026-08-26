"""Rutas y constantes compartidas para el proyecto de EDA de ASL Fingerspelling."""
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"


def resolve_data_dir() -> Path:
    """Prioridad: ASL_DATA_DIR (env) > data/raw/ local."""
    env_path = os.environ.get("ASL_DATA_DIR")
    return Path(env_path) if env_path else DATA_RAW_DIR


# sequence_id es el indice del parquet, no una columna.
def train_csv_path() -> Path:
    return resolve_data_dir() / "train.csv"


def supplemental_csv_path() -> Path:
    return resolve_data_dir() / "supplemental_metadata.csv"


def char_to_pred_json_path() -> Path:
    return resolve_data_dir() / "character_to_prediction_index.json"


def train_landmarks_dir() -> Path:
    return resolve_data_dir() / "train_landmarks"


LANDMARK_TYPES = ["face", "left_hand", "pose", "right_hand"]
COORDS = ["x", "y", "z"]

# Muestra fija que todo el equipo descarga (ver notebooks/00_setup_datos.ipynb).
SAMPLE_LANDMARK_PATHS = [
    "train_landmarks/1019715464.parquet",
    "train_landmarks/1021040628.parquet",
]
