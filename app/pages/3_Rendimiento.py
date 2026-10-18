"""Rendimiento en test: graficas interactivas de los modelos, cada una se puede ocultar."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import utils

utils.configurar_pagina("Rendimiento")
st.title("Rendimiento de los modelos")

reportes = utils.cargar_reportes()
if not reportes:
    st.warning("Aún no hay métricas en `reports/metrics/`.")
    st.stop()

with st.sidebar:
    st.header("Opciones")
    elegidos = st.multiselect("Modelos", list(reportes), default=list(reportes), format_func=utils.nombre)
    st.caption("Desactiva una gráfica con su interruptor para ocultarla.")
if not elegidos:
    st.info("Elige al menos un modelo en la barra lateral.")
    st.stop()

reportes = {m: reportes[m] for m in elegidos}
colores = utils.mapa_colores(reportes)
PLANTILLA = dict(template="plotly_white", font=dict(family="sans-serif", size=13),
                 margin=dict(l=10, r=10, t=50, b=10), legend_title_text="Modelo")


def seccion(titulo: str, clave: str, explicacion: str) -> bool:
    """Encabezado con interruptor; devuelve si la grafica se muestra."""
    cab, interruptor = st.columns([5, 1])
    cab.subheader(titulo)
    visible = interruptor.toggle("Mostrar", value=True, key=clave)
    if visible:
        st.caption(explicacion)
    return visible


# --- resumen ----------------------------------------------------------------
resumen = pd.DataFrame([{
    "Modelo": utils.nombre(m),
    "CER test": r["cer_test"],
    "CER validación": r["cer_val"],
    "Parámetros (M)": r["n_parametros"] / 1e6,
    "Entrenamiento (min)": r["tiempo_entrenamiento_s"] / 60,
    "ms por secuencia": r["ms_por_secuencia"],
    "Épocas": r.get("epocas", len(r["historial"]["val_cer"])),
} for m, r in reportes.items()]).sort_values("CER test")
mejor = resumen.iloc[0]
c1, c2, c3 = st.columns(3)
c1.metric("Mejor modelo en test", mejor["Modelo"])
c2.metric("CER test", f"{mejor['CER test']:.3f}")
c3.metric("Caracteres correctos (≈ 1 − CER)", f"{100 * (1 - mejor['CER test']):.0f} %")
st.dataframe(resumen, hide_index=True, width="stretch",
             column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ["CER test", "CER validación"]}
             | {c: st.column_config.NumberColumn(format="%.2f") for c in ["Parámetros (M)", "Entrenamiento (min)", "ms por secuencia"]})

# --- 1. CER por modelo ------------------------------------------------------
if seccion("CER por modelo", "g_cer", "Character error rate en validación y en test (menor es mejor). "
           "Validación se usó para elegir hiperparámetros; test solo se usó al final."):
    largo = resumen.melt(id_vars="Modelo", value_vars=["CER validación", "CER test"], var_name="Partición", value_name="CER")
    fig = px.bar(largo, x="Modelo", y="CER", color="Modelo", pattern_shape="Partición", barmode="group",
                 color_discrete_map=colores, pattern_shape_map={"CER validación": "/", "CER test": ""},
                 text_auto=".3f", hover_data={"CER": ":.3f"})
    fig.update_layout(**(PLANTILLA | {"legend_title_text": ""}), yaxis_range=[0, 1])
    st.plotly_chart(fig, width="stretch")

# --- 2. Curvas de entrenamiento ---------------------------------------------
if seccion("Curvas de entrenamiento", "g_curvas", "Evolución por época de la métrica elegida. Las primeras épocas "
           "tienen CER ≈ 1 porque CTC empieza prediciendo solo *blanks*."):
    metrica = st.radio("Métrica", {"val_cer": "CER validación", "val_loss": "Pérdida validación",
                                   "train_loss": "Pérdida entrenamiento"}.items(),
                       format_func=lambda kv: kv[1], horizontal=True, key="metrica_curva")
    filas = [{"Modelo": utils.nombre(m), "Época": i + 1, metrica[1]: v}
             for m, r in reportes.items() for i, v in enumerate(r["historial"][metrica[0]])]
    fig = px.line(pd.DataFrame(filas), x="Época", y=metrica[1], color="Modelo", color_discrete_map=colores)
    fig.update_traces(hovertemplate="época %{x}<br>%{y:.3f}")
    fig.update_layout(**PLANTILLA, hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

# --- 3. Costo vs error ------------------------------------------------------
if seccion("Costo vs error", "g_costo", "Cada burbuja es un modelo: más a la izquierda es más rápido al predecir, "
           "más abajo comete menos errores; el tamaño es el número de parámetros."):
    fig = px.scatter(resumen, x="ms por secuencia", y="CER test", size="Parámetros (M)", color="Modelo",
                     color_discrete_map=colores, size_max=55, text="Modelo",
                     hover_data={"Entrenamiento (min)": ":.1f", "Parámetros (M)": ":.2f", "CER test": ":.3f",
                                 "ms por secuencia": ":.1f", "Modelo": False})
    fig.update_traces(textposition="top center", marker=dict(line=dict(width=1, color="white")))
    # Ejes desde 0 para no exagerar diferencias pequenas entre modelos
    fig.update_layout(**PLANTILLA, showlegend=False, yaxis_range=[0, 1],
                      xaxis_range=[0, resumen["ms por secuencia"].max() * 1.3])
    st.plotly_chart(fig, width="stretch")

# --- 4. Error por caracter --------------------------------------------------
if seccion("Error por carácter", "g_caracter", "Tasa de error en test de cada carácter (solo los que aparecen al "
           "menos 5 veces), ordenados por frecuencia. Los caracteres raros se fallan más."):
    filas = []
    for m, r in reportes.items():
        for c, s in r["cer_por_caracter"]["por_caracter"].items():
            filas.append({"Modelo": utils.nombre(m), "Carácter": "␣" if c == " " else c, "Tasa": s["tasa"], "Total": s["total"]})
    por_caracter = pd.DataFrame(filas)
    por_caracter = por_caracter[por_caracter["Total"] >= 5]
    orden = por_caracter.groupby("Carácter")["Total"].max().sort_values(ascending=False).index
    matriz = por_caracter.pivot(index="Modelo", columns="Carácter", values="Tasa").reindex(columns=orden)
    fig = go.Figure(go.Heatmap(z=matriz.values, x=list(matriz.columns), y=list(matriz.index), zmin=0, zmax=1,
                               colorscale=utils.ESCALA_ERROR, colorbar_title="Tasa",
                               hovertemplate="%{y} · '%{x}'<br>tasa de error %{z:.2f}<extra></extra>"))
    fig.update_layout(**PLANTILLA, height=150 + 45 * len(matriz), xaxis=dict(type="category", tickfont_size=14))
    st.plotly_chart(fig, width="stretch")

    modelo_conf = st.selectbox("Confusiones más frecuentes de", list(reportes), format_func=utils.nombre)
    conf = pd.Series(reportes[modelo_conf]["cer_por_caracter"]["confusiones"]).head(12)
    conf = conf.rename(lambda k: k.replace(" ", "␣")).rename("veces").rename_axis("real → predicho").reset_index()
    fig = px.bar(conf, x="veces", y="real → predicho", orientation="h",
                 color_discrete_sequence=[utils.color(modelo_conf)])
    fig.update_layout(**PLANTILLA, yaxis=dict(autorange="reversed"), height=420)
    st.plotly_chart(fig, width="stretch")

# --- 5. Predicciones de test ------------------------------------------------
if seccion("Predicciones en test", "g_tabla", "Frase real y predicha de cada secuencia de test. Filtra por texto u "
           "ordena por CER para ver los mejores y peores casos."):
    modelo_tabla = st.selectbox("Modelo", list(reportes), format_func=utils.nombre, key="modelo_tabla")
    pred = pd.DataFrame(reportes[modelo_tabla]["predicciones_test"])
    pred["CER"] = [utils.cer_frase(r, p) for r, p in zip(pred["real"], pred["predicha"])]
    texto = st.text_input("Buscar en la frase real")
    if texto:
        pred = pred[pred["real"].str.contains(texto, case=False, regex=False)]
    st.dataframe(pred.sort_values("CER"), hide_index=True, width="stretch",
                 column_config={"CER": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1),
                                "sequence_id": st.column_config.NumberColumn(format="%d")})
