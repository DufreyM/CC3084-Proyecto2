# Efecto de la aumentación de datos

**Responsable:** Cindy Mishelle Gualim Perez

> Sección del informe final (Resultados y análisis). Código en [src/augment.py](../src/augment.py) y
> [src/ablacion_aumentacion.py](../src/ablacion_aumentacion.py); resultados en
> `reports/metrics/ablacion_aumentacion.json`.

## Pregunta y diseño

Con solo 1,315 secuencias de entrenamiento de 64 participantes, la hipótesis era que la aumentación reduciría el
sobreajuste y bajaría el CER en personas nuevas. Para comprobarlo se entrenó la misma CNN 1D + BiGRU dos veces,
cambiando únicamente la aumentación:

- **Arquitectura:** variante `stride_4` del modelo principal (192 canales, 128 unidades GRU, *stride* 4,
  *dropout* 0.3; 888,828 parámetros). Se eligió por ser la más rápida de las variantes competitivas.
- **Entrenamiento:** idéntico en ambas corridas (semilla 42, AdamW con tasa 0.003 en ciclo *one-cycle*, 60 épocas,
  *early stopping* con paciencia 10, mismo split por participante).
- **Aumentación** (solo en entrenamiento, cada transformación con probabilidad 0.5): cambio de velocidad
  (×0.8–1.25), descarte de 10 % de cuadros, escala ±10 % y rotación ±10°, descarte de 5 % de los puntos y ruido
  gaussiano ($\sigma$ = 0.01). Validación y prueba no se aumentan.

## Resultados

| Corrida | CER validación | CER prueba | Pérdida train (última) | Pérdida val (última) |
|---------|---------------:|-----------:|-----------------------:|---------------------:|
| Sin aumentación | **0.559** | **0.493** | 1.16 | 2.14 |
| Con aumentación | 0.571 | 0.506 | 1.35 | 2.07 |

Ninguna corrida se detuvo antes de la época 60 y la mejor época fue la 57 (sin) y la 59 (con): las dos seguían
mejorando al terminar.

## Interpretación

- **La aumentación no mejoró el CER**: quedó 1.2 puntos peor en validación y 1.3 en prueba. Esa diferencia es del
  orden del ruido entre corridas: la corrida sin aumentación del ajuste de parámetros (notebook 07) obtuvo 0.554 en
  validación con la misma configuración, contra 0.559 al repetirla aquí. Con una sola semilla por condición, lo
  prudente es concluir que **no hay evidencia de que ayude** en este presupuesto de entrenamiento, no que perjudique.
- **Sí hizo lo esperado sobre el sobreajuste**: la brecha entre la pérdida de entrenamiento y la de validación pasó
  de 0.98 a 0.72, y la pérdida de validación bajó de 2.14 a 2.07. El modelo memoriza menos, pero no se tradujo en
  menor CER.
- **Hipótesis para el resultado:** la aumentación vuelve más difícil el entrenamiento (la pérdida de
  entrenamiento es mayor) y con 60 épocas el modelo aún no había convergido. La curva de CER en la época 30 lo
  sugiere (0.704 con aumentación contra 0.675 sin ella). Es probable que con más épocas la diferencia se achique o
  se invierta, pero no se probó.
- **Por carácter**, entre los caracteres con al menos 20 apariciones en prueba, la aumentación ayudó en *w*
  (−0.13 de tasa de error), *m* (−0.08) y *d* (−0.05), y empeoró *y* (+0.16), *v* (+0.12), *g* y *u* (+0.11). Con
  tan pocas apariciones por carácter, esto no distingue un efecto real de la variación entre corridas.

## Limitaciones

Una sola semilla por condición, una sola arquitectura y 60 épocas; no se probó cada transformación por separado ni
probabilidades distintas. El tiempo de entrenamiento (26.8 y 20.9 min) no es comparable entre corridas porque se
ejecutaron en momentos distintos con la CPU compartida.

## Trabajo futuro

Entrenar más épocas con aumentación, probar solo las transformaciones menos agresivas (ruido y escala) y repetir
con varias semillas para separar el efecto real del ruido.
