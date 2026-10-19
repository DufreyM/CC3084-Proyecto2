# Plan de trabajo – Proyecto 2: Resultados

**Entrega:** lunes 19 de octubre de 2026, en clase (PDF + scripts + link de Drive + link del repo + presentación).

Igual que en el EDA, cada quien trabaja en **sus propios archivos** para no generar conflictos.
La nota es individual según los commits, así que commiteen seguido y desde su propia cuenta.

## Modelos que vamos a entrenar (de la revisión bibliográfica)

| Rol | Modelo | Responsable |
|-----|--------|-------------|
| Principal | CNN 1D + BiGRU + CTC | Leonardo |
| Mayor techo | Transformer pequeño (encoder) + CTC | María José |
| Línea base | LSTM unidireccional + CTC (+ GRU unidireccional como ablación) | Daniela |
| Variante | BiLSTM + CTC | José |

Todos usan **el mismo split, el mismo preprocesamiento y el mismo loop de entrenamiento** (`src/`),
para que la comparación sea justa. Métrica principal: **CER** (error por carácter) en validación/test
separados por participante.

## Qué reutilizamos de la fase 1 (EDA)

No se empieza de cero: el informe y la app de esta fase **amplían** lo del Avance 1.

| De la fase 1 | Se usa en |
|--------------|-----------|
| `src/config.py`, `src/data_loading.py` y la muestra fija de 2 parquet | Toda la parte de modelado (`src/features.py`, `src/dataset.py`) |
| Decisiones de `02_limpieza_preprocesamiento` (quitar rostro, umbral de 10 % de mano, *forward-fill*) | `src/features.py` las aplica igual, ahora por secuencia |
| Longitudes de `03`/`04` (mediana 147, cola ~750) | `max_frames = 384` y el stride de los modelos |
| Desbalance de caracteres de `03`/`05` | Error por carácter en `src/metrics.py` y en la comparación |
| Desbalance de participantes de `01`/`04` | Split por participante |
| `reports/informe_final.md` (investigación, situación problemática, objetivos, datos, limpieza, EDA) | Base de las secciones del informe final: se actualizan, no se reescriben |
| `reports/figures/*.png` | Sección de EDA del informe y página de exploración de la app |
| `reports/build_informe.py` y `build_presentacion.py` | Generar el PDF y la presentación finales (se extienden con las secciones nuevas) |

## Dependencias: ¿quién puede trabajar ya?

La única dependencia fuerte era la **parte inicial** (datos para modelado + loop de entrenamiento).
**Ya está lista** (Leonardo, 10 oct): todos pueden trabajar en paralelo. Para empezar:
`git pull`, `pip install -r requirements.txt`, correr `notebooks/06_preparacion_modelado.ipynb`
(o `python -m src.dataset`) y seguir la plantilla de la sección 3 de ese notebook.

| Tarea | ¿Depende de algo? |
|-------|-------------------|
| Arquitectura de cada modelo (`nn.Module`) | No |
| Entrenar y tunear modelos | ~~`src/dataset.py` + `src/train.py`~~ **listo**: ya se puede entrenar |
| Aumentación (Cindy) | No: funciones sobre arrays `(T, F)` |
| Página de exploración de datos (José) | No: usa `data/processed` del EDA y `train.csv` |
| Páginas de predicción y rendimiento (María José) | Solo del formato del JSON de métricas (abajo); se puede maquetar con un JSON falso |
| Comparación y figuras (Daniela) | Del formato del JSON; datos reales a partir del jueves 15 |
| Secciones del informe | No |

## Contrato entre partes (leer antes de empezar)

- **Entrada al modelo** (`src/dataset.py`, Leonardo): tensor `(T, F)` por secuencia con landmarks de manos +
  pose normalizados respecto a la muñeca, más máscara de detección; etiqueta = índices de
  `character_to_prediction_index.json` (+1, el 0 es el blank de CTC).
- **Cada modelo vive en `src/models/<nombre>.py`** con su clase y una función `construir(**hiperparametros)`,
  para que la app lo pueda reconstruir (no dejen la clase solo dentro del notebook).
- **Interfaz de los modelos:** `forward(x, x_len) -> (log_probs (B, T', C), out_len)` con
  `log_softmax` aplicado y `C = vocab.NUM_CLASSES`.
- **Entrenamiento** (`src/train.py`, Leonardo): `fit(model, train_dl, val_dl, config)` con `nn.CTCLoss`,
  decodificación greedy y CER en `src/metrics.py`.
- **Cada modelo guarda:**
  - pesos en `models/<nombre>.pt`
  - métricas en `reports/metrics/<nombre>.json` con: `cer_val`, `cer_test`, `cer_por_caracter`,
    `historial` (loss y CER por época), `tiempo_entrenamiento_s`, `ms_por_secuencia`, `n_parametros`,
    `hiperparametros`, `predicciones_test` (frase real vs predicha).
- La app y el notebook de comparación **solo leen** esos archivos.

## Notas para entrenar

- En CPU, un modelo pequeño tarda ~20-25 s por época; para tunear conviene Colab o Kaggle con GPU (`train.py` usa CUDA si existe).
- Las primeras épocas dan CER ≈ 1.0 porque CTC empieza prediciendo solo blanks; es normal, hay que esperar.
- Solo hay 1,315 secuencias de entrenamiento: usen dropout y la aumentación de Cindy, y no esperen CER cercanos a los de Kaggle.
- Si el modelo reduce el tiempo con stride, verifiquen que `out_len` >= largo de la frase, o CTC no puede alinear.

## TODOs por integrante

### Leonardo – Parte inicial (bloquea a todos) + modelo principal + integración del informe
- [x] `src/vocab.py`, `src/features.py`, `src/dataset.py`: normalizar respecto a la muñeca, máscara de detección, forward-fill, submuestreo de secuencias largas, caché en `data/processed`.
- [x] Split **por participante** en `data/split.json` (versionado): 64 / 14 / 14 participantes → 1,315 / 295 / 268 secuencias.
- [x] `src/metrics.py` y `src/train.py`: CTC loss, decodificación greedy, CER, error por carácter, early stopping, semilla fija, guardado del JSON de métricas.
- [x] `notebooks/06_preparacion_modelado.ipynb`: genera la caché y el split, y muestra cómo usarlos.
- [x] Framework: PyTorch, en `requirements.txt`.
- [x] `src/models/cnn_bigru.py` + `notebooks/07_modelo_cnn_bigru.ipynb`: CNN 1D con stride + BiGRU + CTC; tuning de 4 variantes. **CER test 0.490**, guardado en `models/cnn_bigru.pt` y `reports/metrics/cnn_bigru.json` (sirve de ejemplo del formato para la app y la comparación).
- [x] `docs/05_modelo_cnn_bigru.md`: sección del modelo principal para el informe (justificación, ajuste y resultados).
- [ ] Informe: integración final (`reports/build_informe.py` → PDF) y carpeta de Google Drive.
- > ⚠️ **Pendiente para María José — borrar esta nota al implementarlo:** integrar en `build_informe.py`
  > `docs/06_modelo_transformer.md` (modelo Transformer) y `docs/08_app.md` (descripción de la app, con capturas
  > `reports/figures/app_*.png`).

### Cindy – Aumentación + preprocesamiento de la app + marco teórico
- [x] `src/augment.py`: descarte aleatorio de cuadros y de puntos, ruido, escala/rotación leve, cambio de velocidad. Se pasa como `transform` a `LandmarkDataset`.
- [x] `app/preprocess.py`: preprocesar un parquet/secuencia nueva subida en la app con `src.features` (el usuario no debe notar nada).
- [x] Informe (`docs/06_marco_teorico_metodologia.md`): **Marco teórico** (adaptar la revisión bibliográfica del Avance 2 + MediaPipe/landmarks), **Metodología** (pasos, split, herramientas y cómputo) y **Referencias APA**.

### Daniela – Línea base + comparación de modelos
- [x] `src/models/lstm.py` (+ `gru.py`) + `notebooks/08_modelo_lstm.ipynb`: LSTM unidireccional + CTC (línea base) y GRU unidireccional (ablación); tuning de 4 variantes por celda. CER test LSTM 0.566, GRU 0.537 (la GRU le gana a la LSTM con 23 % menos parámetros, Wilcoxon p = 6e-6), guardados en `models/{lstm,gru}.pt` y `reports/metrics/{lstm,gru}.json`.
- [x] `notebooks/11_comparacion_modelos.ipynb`: compara CER (IC 95 % por bootstrap y Wilcoxon por pares), costo, curvas, tipos de error, error por carácter y CER por cobertura de mano.
- [x] 5 visualizaciones estáticas en `reports/figures/comparacion_*.png` (CER, costo, curvas, error por carácter, cobertura de mano), más `08_lstm_tuning.png`, `08_gru_tuning.png` y `08_lstm_gru_entrenamiento.png`.
- [x] Informe: **Resultados y análisis** – ajuste de parámetros y comparación de algoritmos en `docs/09_resultados_analisis.md`.

### José – BiLSTM + exploración de datos en la app
- [ ] `src/models/bilstm.py` + `notebooks/09_modelo_bilstm.ipynb`: BiLSTM + CTC; tuning comparable al de la BiGRU.
- [ ] `app/pages/1_Explorar_datos.py`: explorar las variables del dataset limpio (longitudes, faltantes por mano, frecuencia de caracteres, animación de landmarks de una secuencia). Debe contar la historia del EDA.
- > ⚠️ **Pendiente para María José — borrar esta nota al implementarlo:**
  > - Guardar el modelo con `train.guardar_resultados("bilstm", ...)` (ese nombre exacto) para que la app y la presentación lo detecten.
  > - En `1_Explorar_datos.py` empezar con `import utils` y `utils.configurar_pagina("Explorar datos")`, y usar
  >   `utils.TEAL`, `utils.NARANJA`, `utils.NAVY`, `utils.GRIS` y Plotly para que la página tenga la misma paleta que las demás.
  >   `utils` ya agrega la raíz del repo al `sys.path`, así que después se puede importar `from src import ...`.
  >   La app se corre con `streamlit run app/app.py` desde la raíz del repo.
- [ ] Informe: **Introducción** y **Objetivos** (general y específicos, actualizados al modelado) + descripción del dataset original y EDA (reutilizar el informe del Avance 1).

### María José – Transformer + resultados en la app + presentación
- [x] `src/models/transformer.py` + `notebooks/10_modelo_transformer.ipynb`: encoder Transformer pequeño + CTC (submuestreo temporal antes de la atención para secuencias de ~750 cuadros); tuning de 6 variantes. **CER test 0.499** (empata con el CNN-BiGRU, Wilcoxon p = 0.21, con 0.63 M parámetros y 1.8 ms/secuencia), guardado en `models/transformer.pt` y `reports/metrics/transformer.json`.
- [x] `app/app.py` (inicio) y `app/pages/2_Predecir.py`: subir/elegir una secuencia, elegir un modelo o todos, mostrar frase predicha vs real.
- [x] `app/pages/3_Rendimiento.py`: gráficas **interactivas** de rendimiento en test con opción de **ocultarlas**; paleta coherente (teoría del color).
- [x] Presentación final (`reports/build_presentacion.py` → `reports/presentacion_final.pptx`). Lee `reports/metrics/*.json`: volver a correrlo cuando entren la LSTM, la GRU y la BiLSTM, y actualizar `CONCLUSIONES` en el script.
- [x] Informe: **descripción de la app** con capturas y justificación de tecnologías en `docs/08_app.md`; sección del Transformer en `docs/06_modelo_transformer.md`.

### Todo el equipo
- [ ] **Conclusiones** del informe: cómo se cumplió cada objetivo y qué tan efectivo fue cada modelo.
- [ ] Revisión cruzada del informe y ensayo de la presentación.

## Calendario

| Fecha | Meta |
|-------|------|
| Sáb 10 – Dom 11 oct | Leonardo sube `src/dataset.py`, split y `src/train.py`. El resto escribe la arquitectura de su modelo, la aumentación y el esqueleto de la app con datos falsos. |
| Lun 12 – Mié 14 oct | Entrenar y tunear los 4 modelos. App leyendo JSON de prueba. |
| Jue 15 oct | **Congelar** `models/*.pt` y `reports/metrics/*.json`. |
| Vie 16 oct | Notebook de comparación + figuras estáticas. App conectada a modelos reales. |
| Sáb 17 oct | Borrador completo del informe en Drive. |
| Dom 18 oct | PDF final, presentación, ensayo. |
| Lun 19 oct | Entrega en clase. |

## Estructura nueva del repo

```
app/                      # Streamlit
  app.py
  preprocess.py
  pages/1_Explorar_datos.py, 2_Predecir.py, 3_Rendimiento.py
models/                   # pesos entrenados (*.pt)
notebooks/06..11_*.ipynb
reports/metrics/*.json    # métricas por modelo
src/vocab.py, features.py, dataset.py, metrics.py, train.py, augment.py
```
