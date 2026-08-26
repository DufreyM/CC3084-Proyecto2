"""Rutas y constantes compartidas para el proyecto de EDA de ASL Fingerspelling."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"

# Archivos tal como los entrega Kaggle (competencia asl-fingerspelling).
# TODO(equipo): confirmar estos nombres una vez descargados los datos reales,
# Kaggle a veces ajusta el esquema entre versiones de la competencia.
TRAIN_CSV = DATA_RAW_DIR / "train.csv"
SUPPLEMENTAL_CSV = DATA_RAW_DIR / "supplemental_metadata.csv"
CHAR_TO_PRED_JSON = DATA_RAW_DIR / "character_to_prediction_index.json"
TRAIN_LANDMARKS_DIR = DATA_RAW_DIR / "train_landmarks"

LANDMARK_TYPES = ["face", "left_hand", "pose", "right_hand"]
COORDS = ["x", "y", "z"]
