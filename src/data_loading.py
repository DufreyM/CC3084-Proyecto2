"""Utilidades compartidas para cargar los datos de ASL Fingerspelling."""
from functools import lru_cache
import json

import pandas as pd

from . import config


def load_train_index() -> pd.DataFrame:
    """Carga train.csv (una fila por secuencia)."""
    return pd.read_csv(config.train_csv_path())


def load_supplemental_index() -> pd.DataFrame:
    """Carga supplemental_metadata.csv (secuencias sin frase verificada)."""
    return pd.read_csv(config.supplemental_csv_path())


@lru_cache(maxsize=1)
def load_char_to_prediction_index() -> dict:
    """Carga el mapeo caracter -> indice de character_to_prediction_index.json."""
    with open(config.char_to_pred_json_path(), "r", encoding="utf-8") as f:
        return json.load(f)


def landmark_path(row_path: str):
    """Ruta absoluta a un parquet de landmarks a partir de train_df['path']."""
    return config.resolve_data_dir() / row_path


def load_landmarks(parquet_path, sequence_id: int | None = None) -> pd.DataFrame:
    """Carga un parquet de landmarks; si se da sequence_id, filtra esa secuencia."""
    df = pd.read_parquet(parquet_path)
    if sequence_id is not None:
        if "sequence_id" in df.columns:
            df = df[df["sequence_id"] == sequence_id]
        elif "sequence_id" in (df.index.names or []):
            df = df[df.index.get_level_values("sequence_id") == sequence_id]
    return df


def get_landmark_columns(df: pd.DataFrame, landmark_type: str, coord: str) -> list[str]:
    """Columnas x/y/z de un tipo de landmark (face, pose, left_hand, right_hand)."""
    prefix = f"{coord}_{landmark_type}_"
    return [c for c in df.columns if c.startswith(prefix)]


def missing_landmark_rate(df: pd.DataFrame, landmark_type: str) -> float:
    """Proporcion de NaN en las columnas x de un tipo de landmark."""
    cols = get_landmark_columns(df, landmark_type, "x")
    if not cols:
        return float("nan")
    return df[cols].isna().mean().mean()
