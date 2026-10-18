"""Carga de metricas y modelos, paleta y helpers compartidos por las paginas de la app."""
import html
import json
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import dataset, features, metrics, train  # noqa: E402

MAX_FRAMES = 384  # mismo recorte que en el entrenamiento

# --- paleta -----------------------------------------------------------------
# Parte de la presentacion (navy / teal / naranja). Los dos modelos centrales usan colores
# complementarios (teal vs naranja) para que contrasten; el resto se reparte en el circulo
# cromatico con luminosidad parecida, de modo que ninguno destaque sin razon y se distingan
# tambien con daltonismo (no se usa el par rojo-verde).
NAVY = "#0F1E33"
TEAL = "#2CA6A4"
NARANJA = "#F28C28"
GRIS = "#5B6B7C"
FONDO = "#F6F8FA"

NOMBRES = {
    "cnn_bigru": "CNN-BiGRU",
    "transformer": "Transformer",
    "lstm": "LSTM",
    "gru": "GRU",
    "bilstm": "BiLSTM",
}
COLORES = {
    "cnn_bigru": TEAL,
    "transformer": NARANJA,
    "lstm": "#3B6EA8",
    "gru": "#8E6BBF",
    "bilstm": "#C8507A",
}
ROLES = {
    "cnn_bigru": "Principal",
    "transformer": "Mayor techo",
    "lstm": "Línea base",
    "gru": "Ablación de la línea base",
    "bilstm": "Variante",
}
# Escala secuencial de un solo tono para tasas de error (claro = poco error, oscuro = mucho)
ERROR = "#B23A48"  # errores en la frase predicha; distinto de los colores de modelo
ESCALA_ERROR = [[0.0, "#FFF5EB"], [0.5, "#F28C28"], [1.0, "#7F2704"]]


def nombre(modelo: str) -> str:
    return NOMBRES.get(modelo, modelo)


def color(modelo: str) -> str:
    return COLORES.get(modelo, GRIS)


def mapa_colores(modelos) -> dict:
    """{nombre legible: color} para color_discrete_map de Plotly."""
    return {nombre(m): color(m) for m in modelos}


def configurar_pagina(titulo: str, icono: str = "🤟"):
    st.set_page_config(page_title=f"{titulo} · ASL Fingerspelling", page_icon=icono, layout="wide")


# --- datos ------------------------------------------------------------------
@st.cache_data
def cargar_reportes() -> dict:
    """{modelo: metricas} de reports/metrics/*.json (sin los archivos de ajuste)."""
    reportes = {}
    for path in sorted(train.METRICS_DIR.glob("*.json")):
        if path.stem.endswith("_tuning"):
            continue
        reportes[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    orden = list(NOMBRES)
    return dict(sorted(reportes.items(), key=lambda kv: orden.index(kv[0]) if kv[0] in orden else len(orden)))


def modelos_disponibles() -> list[str]:
    """Modelos con metricas y pesos guardados."""
    return [m for m in cargar_reportes() if (train.MODELS_DIR / f"{m}.pt").exists()]


@st.cache_resource
def cargar_modelo(modelo: str) -> torch.nn.Module:
    return train.cargar_modelo(modelo)


@st.cache_resource
def cargar_cache():
    """Cache de secuencias de modelado, o None si no se ha generado (python -m src.dataset)."""
    if not dataset.CACHE_PATH.exists():
        return None
    return dataset.cargar_cache()


def secuencias_test(cache) -> list[dict]:
    """Secuencias del split de test: id, frase real y posicion en la cache."""
    return [
        {"sequence_id": int(cache["sequence_id"][j]), "frase": str(cache["phrase"][j]), "j": int(j)}
        for j in dataset.indices_de(cache, "test")
    ]


def secuencia_de_cache(cache, j: int) -> np.ndarray:
    return cache["X"][cache["offsets"][j]:cache["offsets"][j + 1]].astype(np.float32)


# --- prediccion -------------------------------------------------------------
@torch.no_grad()
def predecir(modelo: str, x: np.ndarray) -> tuple[str, float]:
    """Secuencia preprocesada (T, F) -> (frase predicha, ms de inferencia)."""
    red = cargar_modelo(modelo)
    x = features.submuestrear(x, MAX_FRAMES)
    xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None]
    inicio = time.perf_counter()
    log_probs, out_len = red(xt, torch.tensor([len(x)]))
    ms = 1000 * (time.perf_counter() - inicio)
    return metrics.greedy_decode(log_probs, out_len)[0], ms


def cer_frase(real: str, predicha: str) -> float:
    return metrics.cer([real], [predicha])


def resaltar(real: str, predicha: str) -> str:
    """HTML de la frase predicha: verde si acierta, rojo si sustituye, tachado si borra, subrayado si inserta."""
    partes = []
    for c_ref, c_hyp in metrics.alinear(real, predicha):
        if c_ref == c_hyp:
            partes.append(f"<span style='color:{TEAL}'>{html.escape(c_hyp)}</span>")
        elif not c_hyp:
            partes.append(f"<span style='color:{GRIS};text-decoration:line-through;opacity:.6'>{html.escape(c_ref)}</span>")
        elif not c_ref:
            partes.append(f"<span style='color:{ERROR};text-decoration:underline'>{html.escape(c_hyp)}</span>")
        else:
            partes.append(f"<span style='color:{ERROR};font-weight:700'>{html.escape(c_hyp)}</span>")
    return "<span style='font-family:monospace;font-size:1.15rem;white-space:pre-wrap'>" + "".join(partes) + "</span>"


LEYENDA_RESALTADO = (
    f"<small><span style='color:{TEAL}'>■</span> correcto &nbsp; "
    f"<span style='color:{ERROR};font-weight:700'>■</span> sustitución &nbsp; "
    f"<span style='color:{ERROR};text-decoration:underline'>a</span> inserción &nbsp; "
    f"<span style='color:{GRIS};text-decoration:line-through'>a</span> letra omitida</small>"
)
