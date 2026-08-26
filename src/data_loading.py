"""
Utilidades compartidas para cargar los datos de la competencia ASL Fingerspelling.

Estructura esperada en data/raw/ (tal como la entrega Kaggle):
    train.csv                          -> una fila por secuencia (path, file_id,
                                           sequence_id, participant_id, phrase)
    supplemental_metadata.csv          -> secuencias adicionales sin frase verificada
    character_to_prediction_index.json -> mapeo caracter -> indice para la frase
    train_landmarks/*.parquet          -> un frame por fila, columnas x_/y_/z_
                                           por cada landmark (face, pose,
                                           left_hand, right_hand)

TODO(equipo): confirmar estos nombres/columnas contra los archivos reales una
vez descargados -- Kaggle a veces ajusta el esquema entre versiones.

Todas las funciones aquí son compartidas por los notebooks. Si necesitas algo
nuevo para tu parte del EDA, agrégalo aquí en vez de duplicarlo en el notebook.
"""
from functools import lru_cache
import json

import pandas as pd

from . import config


def load_train_index() -> pd.DataFrame:
    """Carga train.csv: metadata de cada secuencia (una fila por secuencia)."""
    return pd.read_csv(config.train_csv_path())


def load_supplemental_index() -> pd.DataFrame:
    """Carga supplemental_metadata.csv (secuencias sin frase verificada por humano)."""
    return pd.read_csv(config.supplemental_csv_path())


@lru_cache(maxsize=1)
def load_char_to_prediction_index() -> dict:
    """Carga el mapeo caracter -> indice usado para codificar las frases."""
    with open(config.char_to_pred_json_path(), "r", encoding="utf-8") as f:
        return json.load(f)


def load_landmarks(parquet_path, sequence_id: int | None = None) -> pd.DataFrame:
    """Carga un archivo parquet de landmarks.

    Si se especifica sequence_id, filtra solo esa secuencia (un parquet suele
    contener varias secuencias de un mismo participante/sesión).
    """
    df = pd.read_parquet(parquet_path)
    if sequence_id is not None:
        if "sequence_id" in df.columns:
            df = df[df["sequence_id"] == sequence_id]
        elif "sequence_id" in (df.index.names or []):
            df = df[df.index.get_level_values("sequence_id") == sequence_id]
    return df


def get_landmark_columns(df: pd.DataFrame, landmark_type: str, coord: str) -> list[str]:
    """Devuelve las columnas de un tipo de landmark y coordenada dados.

    landmark_type: uno de config.LANDMARK_TYPES (face, pose, left_hand, right_hand)
    coord: uno de config.COORDS (x, y, z)
    """
    prefix = f"{coord}_{landmark_type}_"
    return [c for c in df.columns if c.startswith(prefix)]


def missing_landmark_rate(df: pd.DataFrame, landmark_type: str) -> float:
    """Proporción de valores faltantes (NaN) en las columnas x de un tipo de landmark.

    Útil para detectar frames donde, por ejemplo, la mano no aparece en cámara.
    """
    cols = get_landmark_columns(df, landmark_type, "x")
    if not cols:
        return float("nan")
    return df[cols].isna().mean().mean()
