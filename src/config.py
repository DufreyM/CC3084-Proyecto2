"""Rutas y constantes compartidas para el proyecto de EDA de ASL Fingerspelling."""
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"

# Escrito por notebooks/00_setup_datos.ipynb cuando los datos se descargan con
# kagglehub (que los cachea fuera del repo, no en data/raw/).
_KAGGLEHUB_PATH_FILE = DATA_RAW_DIR / ".kagglehub_path.txt"


def resolve_data_dir() -> Path:
    """Devuelve el directorio donde estan los datos crudos de la competencia.

    Prioridad: variable de entorno ASL_DATA_DIR > ruta guardada por
    notebooks/00_setup_datos.ipynb tras descargar con kagglehub > data/raw/ local.
    """
    env_path = os.environ.get("ASL_DATA_DIR")
    if env_path:
        return Path(env_path)
    if _KAGGLEHUB_PATH_FILE.exists():
        return Path(_KAGGLEHUB_PATH_FILE.read_text(encoding="utf-8").strip())
    return DATA_RAW_DIR


# Archivos tal como los entrega Kaggle (competencia asl-fingerspelling).
# TODO(equipo): confirmar estos nombres una vez descargados los datos reales,
# Kaggle a veces ajusta el esquema entre versiones de la competencia.
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
