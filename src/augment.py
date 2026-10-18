"""Aumentacion de secuencias (T, F) ya normalizadas; se pasa como transform a LandmarkDataset."""
import numpy as np

from . import features

N_COORDS = features.NUM_FEATURES - 2  # lo ultimo son las dos columnas de la mascara de deteccion
N_PUNTOS = N_COORDS // 3


def _coords(x: np.ndarray) -> np.ndarray:
    """Vista (T, puntos, 3) de las coordenadas; las columnas son punto a punto y luego x, y, z."""
    return x[:, :N_COORDS].reshape(len(x), N_PUNTOS, 3)


def descartar_cuadros(x: np.ndarray, rng: np.random.Generator, prob: float = 0.1) -> np.ndarray:
    """Quita cuadros al azar (nunca deja menos de 2)."""
    conservar = rng.random(len(x)) >= prob
    if conservar.sum() < 2:
        return x
    return x[conservar]


def descartar_puntos(x: np.ndarray, rng: np.random.Generator, prob: float = 0.05) -> np.ndarray:
    """Pone en 0 puntos sueltos de algunos cuadros, como una no deteccion de MediaPipe."""
    x = x.copy()
    caidos = rng.random((len(x), N_PUNTOS)) < prob
    _coords(x)[caidos] = 0.0
    return x


def agregar_ruido(x: np.ndarray, rng: np.random.Generator, sigma: float = 0.01) -> np.ndarray:
    """Ruido gaussiano en las coordenadas (la mascara no se toca)."""
    x = x.copy()
    x[:, :N_COORDS] += rng.normal(0.0, sigma, size=(len(x), N_COORDS)).astype(x.dtype)
    return x


def escalar_y_rotar(x: np.ndarray, rng: np.random.Generator, escala: float = 0.1, grados: float = 10.0) -> np.ndarray:
    """Escala toda la secuencia y la rota en el plano x-y.

    Las coordenadas ya estan centradas (muneca / hombros), asi que el origen es el centro natural.
    """
    x = x.copy()
    s = rng.uniform(1 - escala, 1 + escala)
    a = np.deg2rad(rng.uniform(-grados, grados))
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], dtype=x.dtype)
    c = _coords(x)
    c[..., :2] = c[..., :2] @ rot.T
    c *= s
    return x


def cambiar_velocidad(x: np.ndarray, rng: np.random.Generator, rango: tuple[float, float] = (0.8, 1.25)) -> np.ndarray:
    """Remuestrea en el tiempo por interpolacion lineal; factor > 1 acelera (menos cuadros)."""
    factor = rng.uniform(*rango)
    nuevo_largo = max(2, int(round(len(x) / factor)))
    if nuevo_largo == len(x):
        return x
    t = np.linspace(0, len(x) - 1, nuevo_largo)
    i0 = np.floor(t).astype(int)
    i1 = np.minimum(i0 + 1, len(x) - 1)
    w = (t - i0)[:, None].astype(x.dtype)
    y = x[i0] * (1 - w) + x[i1] * w
    y[:, N_COORDS:] = (y[:, N_COORDS:] >= 0.5).astype(x.dtype)  # la mascara vuelve a ser 0/1
    return y


class Aumentacion:
    """Aplica cada transformacion con su propia probabilidad. Uso: LandmarkDataset(..., transform=Aumentacion())."""

    def __init__(self, p_cuadros: float = 0.5, p_puntos: float = 0.5, p_ruido: float = 0.5,
                 p_geometria: float = 0.5, p_velocidad: float = 0.5, semilla: int | None = None):
        self.p = {
            "cuadros": p_cuadros, "puntos": p_puntos, "ruido": p_ruido,
            "geometria": p_geometria, "velocidad": p_velocidad,
        }
        self.rng = np.random.default_rng(semilla)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        rng, p = self.rng, self.p
        if rng.random() < p["velocidad"]:
            x = cambiar_velocidad(x, rng)
        if rng.random() < p["cuadros"]:
            x = descartar_cuadros(x, rng)
        if rng.random() < p["geometria"]:
            x = escalar_y_rotar(x, rng)
        if rng.random() < p["puntos"]:
            x = descartar_puntos(x, rng)
        if rng.random() < p["ruido"]:
            x = agregar_ruido(x, rng)
        return x
