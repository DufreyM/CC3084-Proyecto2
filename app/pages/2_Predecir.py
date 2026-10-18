"""Prediccion: frase predicha vs real para una secuencia de test o un parquet subido."""
import pandas as pd
import streamlit as st

import utils
from src import data_loading, features

utils.configurar_pagina("Predecir")
st.title("Predecir una frase")

disponibles = utils.modelos_disponibles()
if not disponibles:
    st.warning("No hay modelos con pesos en `models/`. Entrena al menos uno para usar esta página.")
    st.stop()


def preprocesar_subido(archivo) -> dict:
    """{sequence_id: (T, F)} de un parquet de landmarks subido."""
    from preprocess import preprocesar_parquet
    return preprocesar_parquet(archivo)


# --- 1. Elegir la secuencia -------------------------------------------------
st.subheader("1. Secuencia")
fuente = st.radio("Fuente", ["Secuencia de test", "Subir un parquet de landmarks"], horizontal=True)

x, real, etiqueta = None, None, ""
if fuente == "Secuencia de test":
    cache = utils.cargar_cache()
    if cache is None:
        st.warning("No existe la caché de secuencias. Genérala con `python -m src.dataset` o sube un parquet.")
        st.stop()
    test = utils.secuencias_test(cache)
    st.caption(f"{len(test)} secuencias de 14 participantes que ningún modelo vio al entrenar.")
    elegida = st.selectbox("Secuencia", test, format_func=lambda s: f"{s['sequence_id']} — {s['frase']}")
    x, real, etiqueta = utils.secuencia_de_cache(cache, elegida["j"]), elegida["frase"], str(elegida["sequence_id"])
else:
    archivo = st.file_uploader("Parquet con el formato de `train_landmarks/` de la competencia", type="parquet",
                               help="Los archivos completos de la competencia pesan ~1.5 GB; sube uno con pocas secuencias.")
    if archivo is None:
        st.info("Sube un archivo para continuar.")
        st.stop()
    with st.spinner("Preprocesando..."):
        secuencias = preprocesar_subido(archivo)
    if not secuencias:
        st.error("El archivo no tiene secuencias con el formato esperado.")
        st.stop()
    frases = data_loading.load_train_index().set_index("sequence_id")["phrase"]
    sid = st.selectbox("Secuencia", list(secuencias), format_func=lambda s: f"{s} — {frases.get(s, 'frase desconocida')}")
    x, etiqueta = secuencias[sid], str(sid)
    real = frases.get(sid)

cobertura = features.cobertura_mano(x[:, -2:])
c1, c2, c3 = st.columns(3)
c1.metric("Cuadros", len(x))
c2.metric("Cuadros con mano detectada", f"{100 * cobertura:.0f} %")
c3.metric("Caracteres reales", len(real) if real else "—")
if cobertura < features.UMBRAL_MANO:
    st.warning("Menos del 10 % de los cuadros tienen una mano detectada: en el entrenamiento estas secuencias se descartaron.")
if len(x) > utils.MAX_FRAMES:
    st.caption(f"La secuencia se submuestrea a {utils.MAX_FRAMES} cuadros, igual que en el entrenamiento.")

# --- 2. Elegir modelos ------------------------------------------------------
st.subheader("2. Modelos")
modo = st.radio("Modelos a usar", ["Todos", "Uno"], horizontal=True)
if modo == "Todos":
    modelos = disponibles
else:
    modelos = [st.selectbox("Modelo", disponibles, format_func=utils.nombre)]

# --- 3. Resultado -----------------------------------------------------------
st.subheader("3. Resultado")
if real:
    st.markdown(f"**Frase real:** <span style='font-family:monospace;font-size:1.15rem'>{real}</span>",
                unsafe_allow_html=True)

filas = []
for modelo in modelos:
    predicha, ms = utils.predecir(modelo, x)
    filas.append({"modelo": modelo, "predicha": predicha, "ms": ms,
                  "cer": utils.cer_frase(real, predicha) if real else None})

for f in sorted(filas, key=lambda f: f["cer"] if f["cer"] is not None else 0):
    with st.container(border=True):
        cab, met = st.columns([3, 1])
        cab.markdown(f"<span style='color:{utils.color(f['modelo'])};font-size:1.2rem'>●</span> "
                     f"**{utils.nombre(f['modelo'])}**", unsafe_allow_html=True)
        if f["cer"] is not None:
            met.markdown(f"CER **{f['cer']:.3f}** · {f['ms']:.0f} ms")
            st.markdown(utils.resaltar(real, f["predicha"]) if f["predicha"] else "*(predicción vacía)*",
                        unsafe_allow_html=True)
        else:
            met.markdown(f"{f['ms']:.0f} ms")
            st.code(f["predicha"] or "(predicción vacía)", language=None)

if real:
    st.markdown(utils.LEYENDA_RESALTADO, unsafe_allow_html=True)
    if len(filas) > 1:
        tabla = pd.DataFrame(filas).assign(modelo=lambda d: d["modelo"].map(utils.nombre))
        st.dataframe(tabla.rename(columns={"predicha": "frase predicha", "ms": "ms de inferencia", "cer": "CER"}),
                     hide_index=True, column_config={"CER": st.column_config.NumberColumn(format="%.3f"),
                                                     "ms de inferencia": st.column_config.NumberColumn(format="%.1f")})
st.caption(f"Secuencia {etiqueta}. CER = ediciones (inserciones, borrados y sustituciones) / caracteres de la frase real.")
