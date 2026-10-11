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

## Contrato entre partes (leer antes de empezar)

- **Entrada al modelo** (`src/dataset.py`, Cindy): tensor `(T, F)` por secuencia con landmarks de manos +
  pose normalizados respecto a la muñeca, más máscara de detección; etiqueta = índices de
  `character_to_prediction_index.json`.
- **Entrenamiento** (`src/train.py`, Leonardo): `fit(model, train_dl, val_dl, config)` con `nn.CTCLoss`,
  decodificación greedy y CER en `src/metrics.py`.
- **Cada modelo guarda:**
  - pesos en `models/<nombre>.pt`
  - métricas en `reports/metrics/<nombre>.json` con: `cer_val`, `cer_test`, `cer_por_caracter`,
    `historial` (loss y CER por época), `tiempo_entrenamiento_s`, `ms_por_secuencia`, `n_parametros`,
    `hiperparametros`, `predicciones_test` (frase real vs predicha).
- La app y el notebook de comparación **solo leen** esos archivos.

## TODOs por integrante

### Cindy – Datos para modelado + preprocesamiento de la app
- [ ] `notebooks/06_preparacion_modelado.ipynb` + `src/dataset.py`: normalizar respecto a la muñeca, máscara de detección, forward-fill, recorte/padding de longitud.
- [ ] Split **por participante** train / val / test; guardar ids en `data/processed/split.json`. Revisar cuántos participantes hay en la muestra de 2 parquet y, si son muy pocos, proponer bajar más archivos.
- [ ] Aumentación simple (descarte aleatorio de cuadros/puntos, ruido) como opción en el `Dataset`.
- [ ] `app/preprocess.py`: misma función de preprocesamiento para un parquet/secuencia nueva subida en la app (el usuario no debe notar nada).
- [ ] Informe: **Metodología** – selección de conjuntos de entrenamiento y prueba, preprocesamiento.
- [ ] **Prioridad: tener `src/dataset.py` y el split listos el domingo 11**, el resto depende de esto.

### Leonardo – Loop de entrenamiento + modelo principal + integración del informe
- [ ] `src/train.py` y `src/metrics.py`: CTC loss, decodificación greedy, CER, guardado del JSON de métricas, early stopping, semilla fija.
- [ ] `notebooks/07_modelo_cnn_bigru.ipynb`: CNN 1D con stride + BiGRU + CTC; tuning (stride, capas, hidden, dropout, lr).
- [ ] Definir framework (propuesta: PyTorch) y actualizar `requirements.txt`.
- [ ] Informe: **Marco teórico** (adaptar la revisión bibliográfica del Avance 2), **Referencias APA** e integración final (`reports/build_informe.py` → PDF).
- [ ] Crear la carpeta de Google Drive del informe y compartir el link.

### Daniela – Línea base + comparación de modelos
- [ ] `notebooks/08_modelo_lstm.ipynb`: LSTM unidireccional + CTC (línea base) y GRU unidireccional (ablación); tuning básico.
- [ ] `notebooks/11_comparacion_modelos.ipynb`: leer `reports/metrics/*.json` y comparar CER, tiempo de entrenamiento, ms por secuencia y n.º de parámetros.
- [ ] Al menos **3 visualizaciones estáticas** en `reports/figures/` (p. ej. CER por modelo, curvas de entrenamiento, error por carácter / matriz de confusión de letras M-N-T).
- [ ] Informe: **Resultados y análisis** – ajuste de parámetros y comparación de algoritmos.

### José – BiLSTM + exploración de datos en la app
- [ ] `notebooks/09_modelo_bilstm.ipynb`: BiLSTM + CTC; tuning comparable al de la BiGRU.
- [ ] `app/pages/1_Explorar_datos.py`: explorar las variables del dataset limpio (longitudes, faltantes por mano, frecuencia de caracteres, animación de landmarks de una secuencia). Debe contar la historia del EDA.
- [ ] Informe: **Introducción** y **Objetivos** (general y específicos, actualizados al modelado) + descripción del dataset original y EDA (reutilizar el informe del Avance 1).

### María José – Transformer + resultados en la app + presentación
- [ ] `notebooks/10_modelo_transformer.ipynb`: encoder Transformer pequeño + CTC (submuestreo temporal antes de la atención para secuencias de ~750 cuadros).
- [ ] `app/app.py` (inicio) y `app/pages/2_Predecir.py`: subir/elegir una secuencia, elegir un modelo o todos, mostrar frase predicha vs real.
- [ ] `app/pages/3_Rendimiento.py`: gráficas **interactivas** de rendimiento en test con opción de **ocultarlas**; paleta coherente (teoría del color).
- [ ] Presentación final (`reports/build_presentacion.py` o diapositivas nuevas).
- [ ] Informe: **descripción de la app** con capturas y justificación de tecnologías (propuesta: Streamlit + Plotly).

### Todo el equipo
- [ ] **Conclusiones** del informe: cómo se cumplió cada objetivo y qué tan efectivo fue cada modelo.
- [ ] Revisión cruzada del informe y ensayo de la presentación.

## Calendario

| Fecha | Meta |
|-------|------|
| Sáb 10 – Dom 11 oct | `src/dataset.py`, split y `src/train.py` listos. El resto escribe la arquitectura de su modelo y el esqueleto de la app con datos falsos. |
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
src/dataset.py, train.py, metrics.py
```
