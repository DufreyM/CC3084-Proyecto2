"""Convierte los landmarks de una secuencia en la matriz (T, F) que reciben los modelos."""
import numpy as np
import pandas as pd

from . import config

PUNTOS = {"left_hand": 21, "right_hand": 21, "pose": 33}


def columnas(tipo: str) -> list[str]:
    """Columnas de un tipo ordenadas por punto y luego x, y, z."""
    return [f"{c}_{tipo}_{i}" for i in range(PUNTOS[tipo]) for c in config.COORDS]


COLUMNAS_MODELO = ["frame"] + [c for tipo in PUNTOS for c in columnas(tipo)]


def leer_parquet(path) -> pd.DataFrame:
    """Lee solo manos y pose de un parquet de landmarks (indice = sequence_id)."""
    return pd.read_parquet(path, columns=COLUMNAS_MODELO)


def a_puntos(seq_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Una secuencia -> {tipo: array (T, puntos, 3)} ordenado por frame."""
    seq_df = seq_df.sort_values("frame")
    return {
        tipo: seq_df[columnas(tipo)].to_numpy(np.float32).reshape(len(seq_df), n, 3)
        for tipo, n in PUNTOS.items()
    }
