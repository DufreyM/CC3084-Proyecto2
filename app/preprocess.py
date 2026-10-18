"""Preprocesa un parquet de landmarks subido en la app con el mismo codigo que el entrenamiento."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # para importar src desde Streamlit

from src import features  # noqa: E402
from src.dataset import collate_ctc  # noqa: E402

MAX_FRAMES = 384  # igual que el entrenamiento


def leer_landmarks(origen) -> pd.DataFrame:
    """Lee un parquet (ruta o archivo subido) y deja sequence_id como indice."""
    df = pd.read_parquet(origen)
    if "sequence_id" in df.columns:
        df = df.set_index("sequence_id")
    faltan = [c for c in features.COLUMNAS_MODELO if c not in df.columns]
    if faltan:
        raise ValueError(f"El archivo no tiene las columnas de landmarks esperadas (faltan {len(faltan)}, p. ej. {faltan[0]}).")
    return df[features.COLUMNAS_MODELO]


def preprocesar(origen) -> dict[int, dict]:
    """Parquet -> {sequence_id: {"x": (T, F), "cobertura_mano": float}}.

    No descarta secuencias con poca mano (a diferencia del entrenamiento): la app siempre predice
    y deja la cobertura para avisar que la prediccion puede ser poco confiable.
    """
    df = leer_landmarks(origen)
    salida = {}
    for seq_id, seq_df in df.groupby(level=0):
        x = features.preprocesar_secuencia(seq_df)
        salida[int(seq_id)] = {"x": x, "cobertura_mano": features.cobertura_mano(x[:, -2:])}
    return salida


def a_lote(secuencias: list[np.ndarray], max_frames: int = MAX_FRAMES):
    """Lista de (T, F) -> (x, x_len) listos para model(x, x_len), con el mismo submuestreo del entrenamiento."""
    batch = [(torch.from_numpy(np.ascontiguousarray(features.submuestrear(x, max_frames))), torch.zeros(1, dtype=torch.long))
             for x in secuencias]
    x, x_len, _, _ = collate_ctc(batch)
    return x, x_len


def preprocesar_parquet(archivo) -> dict[int, np.ndarray]:
    """{sequence_id: (T, F)} de un parquet subido (lo que devuelve st.file_uploader)."""
    return {sid: d["x"] for sid, d in preprocesar(archivo).items()}
