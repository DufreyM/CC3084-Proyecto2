"""Inicio de la app: streamlit run app/app.py"""
from pathlib import Path

import streamlit as st

import utils


def inicio():
    utils.configurar_pagina("Inicio")
    st.title("🤟 Reconocimiento de deletreo manual en ASL")
    st.markdown(
        """
    Esta app acompaña el **Proyecto 2 de CC3084 – Data Science (UVG)**. A partir de los *landmarks* de MediaPipe
    (manos y pose, cuadro por cuadro) de la competencia **Google – ASL Fingerspelling Recognition**, varios modelos
    secuenciales con pérdida **CTC** transcriben la frase deletreada con el alfabeto manual de la lengua de señas
    americana: direcciones, teléfonos y URLs, carácter por carácter.
    """
    )

    reportes = utils.cargar_reportes()
    disponibles = utils.modelos_disponibles()

    if reportes:
        mejor = min(reportes, key=lambda m: reportes[m]["cer_test"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Modelos entrenados", len(reportes))
        c2.metric("Mejor CER en test", f"{reportes[mejor]['cer_test']:.3f}", help="Character error rate: ediciones / caracteres reales. Menor es mejor.")
        c3.metric("Mejor modelo", utils.nombre(mejor))

    st.subheader("Modelos")
    if not reportes:
        st.info("Aún no hay métricas en `reports/metrics/`. Entrena un modelo con su notebook para verlo aquí.")
    for modelo, r in reportes.items():
        estado = "" if modelo in disponibles else " · *(sin pesos en `models/`, solo métricas)*"
        st.markdown(
            f"<span style='color:{utils.color(modelo)};font-size:1.3rem'>●</span> "
            f"**{utils.nombre(modelo)}** — {utils.ROLES.get(modelo, '')} · CER test **{r['cer_test']:.3f}** · "
            f"{r['n_parametros'] / 1e6:.2f} M parámetros{estado}",
            unsafe_allow_html=True,
        )

    st.subheader("Cómo usar la app")
    st.markdown(
        """
    | Página | Qué hace |
    |--------|----------|
    | **Explorar datos** | Recorre el dataset limpio y la historia del análisis exploratorio. |
    | **Predecir** | Elige una secuencia de test (o sube un parquet de *landmarks*), elige uno o todos los modelos y compara la frase predicha con la real. |
    | **Rendimiento** | Gráficas interactivas del desempeño en test: CER, curvas de entrenamiento, costo vs error y error por carácter. Cada gráfica se puede ocultar. |

    Todos los modelos usan **los mismos datos, el mismo split por participante** (las personas de test no se vieron
    al entrenar) y el mismo loop de entrenamiento, así que sus métricas son comparables.
    """
    )

    st.caption("María José Girón · Leonardo Mejía · Cindy Gualim · Daniela Ramírez · José Donado")


# Cada archivo de app/pages/ es una pagina; el titulo sale del nombre (1_Explorar_datos.py -> "Explorar datos")
paginas = [st.Page(inicio, title="Inicio", icon="🏠", default=True)] + [
    st.Page(str(p), title=p.stem.split("_", 1)[-1].replace("_", " "), url_path=p.stem.split("_", 1)[-1])
    for p in sorted((Path(__file__).parent / "pages").glob("*.py"))
]
st.navigation(paginas).run()
