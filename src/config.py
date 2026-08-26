"""Rutas y constantes compartidas para el proyecto de EDA de ASL Fingerspelling."""
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"

# Ruta que guarda notebooks/00_setup_datos.ipynb tras descargar con kagglehub.
_KAGGLEHUB_PATH_FILE = DATA_RAW_DIR / ".kagglehub_path.txt"


def resolve_data_dir() -> Path:
    """Prioridad: ASL_DATA_DIR (env) > cache de kagglehub > data/raw/ local."""
    env_path = os.environ.get("ASL_DATA_DIR")
    if env_path:
        return Path(env_path)
    if _KAGGLEHUB_PATH_FILE.exists():
        return Path(_KAGGLEHUB_PATH_FILE.read_text(encoding="utf-8").strip())
    return DATA_RAW_DIR


# TODO(equipo): confirmar estos nombres contra los datos reales una vez descargados.
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
