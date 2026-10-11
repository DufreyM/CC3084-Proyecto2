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


MUNECA, MCP_MEDIO = 0, 9
HOMBRO_IZQ, HOMBRO_DER = 11, 12
EPS = 1e-6


def _escala(dist: np.ndarray) -> float:
    """Mediana de la secuencia para que un cuadro ruidoso no infle la escala."""
    med = np.nanmedian(dist) if np.isfinite(dist).any() else 1.0
    return float(med) if med > EPS else 1.0


def normalizar_mano(mano: np.ndarray) -> np.ndarray:
    """Centra en la muneca y escala por la distancia muneca-MCP medio."""
    centrada = mano - mano[:, MUNECA : MUNECA + 1]
    return centrada / _escala(np.linalg.norm(centrada[:, MCP_MEDIO, :2], axis=-1))


def normalizar_pose(pose: np.ndarray) -> np.ndarray:
    """Centra en el punto medio de los hombros y escala por el ancho de hombros."""
    hombros = pose[:, [HOMBRO_IZQ, HOMBRO_DER]]
    centrada = pose - hombros.mean(axis=1, keepdims=True)
    return centrada / _escala(np.linalg.norm(hombros[:, 0, :2] - hombros[:, 1, :2], axis=-1))


UMBRAL_MANO = 0.10  # mismo umbral que 02_limpieza_preprocesamiento
NUM_FEATURES = sum(n * 3 for n in PUNTOS.values()) + 2


def mascara_deteccion(puntos: dict[str, np.ndarray]) -> np.ndarray:
    """(T, 2): 1 si la mano izquierda / derecha fue detectada en el cuadro."""
    manos = [puntos["left_hand"], puntos["right_hand"]]
    return np.stack([~np.isnan(m).all(axis=(1, 2)) for m in manos], axis=1).astype(np.float32)


def cobertura_mano(mascara: np.ndarray) -> float:
    """Proporcion de cuadros con al menos una mano detectada."""
    return float(mascara.any(axis=1).mean()) if len(mascara) else 0.0


def forward_fill(x: np.ndarray) -> np.ndarray:
    """Rellena NaN con el ultimo cuadro valido; lo que queda al inicio va a 0."""
    df = pd.DataFrame(x).ffill().fillna(0.0)
    return df.to_numpy(np.float32)


def preprocesar_secuencia(seq_df: pd.DataFrame) -> np.ndarray:
    """Una secuencia cruda -> (T, NUM_FEATURES) normalizada, sin NaN, con mascara al final."""
    puntos = a_puntos(seq_df)
    mascara = mascara_deteccion(puntos)
    partes = [
        normalizar_mano(puntos["left_hand"]),
        normalizar_mano(puntos["right_hand"]),
        normalizar_pose(puntos["pose"]),
    ]
    coords = np.concatenate([p.reshape(len(mascara), -1) for p in partes], axis=1)
    return np.concatenate([forward_fill(coords), mascara], axis=1)


def submuestrear(x: np.ndarray, max_frames: int) -> np.ndarray:
    """Si la secuencia es mas larga que max_frames, toma cuadros equiespaciados."""
    if len(x) <= max_frames:
        return x
    idx = np.linspace(0, len(x) - 1, max_frames).round().astype(int)
    return x[idx]
