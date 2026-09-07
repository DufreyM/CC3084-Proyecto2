# Hallazgos y conclusiones (Actividad 5 de la guía)

**Responsable principal:** Todo el equipo (integrar al final, una vez cerrados 01-05)

> Archivo independiente. Se llena al final, cuando los notebooks de EDA
> (01-05) ya tengan resultados. Corresponde al rubro **Hallazgos y
> conclusiones (20 pts)**.

## Síntesis de la investigación técnica

El deletreo manual en ASL implica el uso de 5 a 6 letras por segundo, similar a la rapidez encontrada en el idioma español (verbal), que según Deutsche Welle (2025) utiliza 7.82 sílabas por segundo, es decir, un poco más de 12 letras por segundo. El deletreo del ASL se utiliza para nombres propios, siglas o términos técnicos.

El presente proyecto emplea MediaPipe para extraer 21 puntos tridimensionales (*landmarks*) por mano en lugar de procesar video crudo. Esto protege la privacidad, reduce significativamente el tamaño de los datos y elimina interferencias visuales como la iluminación o el fondo.

Por otro lado, se debe tomar en cuenta que la pérdida de *landmarks* (valores nulos) no es un error imprevisto, sino que puede deberse a la ausencia de detección por movimiento rápido, oclusión o porque la mano sale del encuadre. En los datos de muestra, la mano derecha está ausente en más del 54.5 % de los fotogramas.

Para traducir estos *landmarks* a texto, las arquitecturas de aprendizaje automático ideales incluyen LSTM, CNN 1D, Transformers y CTC (*Connectionist Temporal Classification*), siendo esta última fundamental para secuencias continuas sin un límite difuso entre letras.

## Resumen de hallazgos

* **[01_estructura_datos](../notebooks/01_estructura_datos.ipynb):** `train.csv` trae 67,208 secuencias de 94 participantes, pero están muy desbalanceadas entre sí: van de 1 a 1,535 secuencias por persona (mediana de 794), así que cualquier comparación por participante debe tomarse con cuidado. Las frases van de 1 a 31 caracteres (mediana de 17). Cada Parquet de *landmarks* trae aproximadamente 1,000 secuencias, indexadas por `sequence_id` (no es columna), con 1,630 columnas por cuadro: 468 de rostro, 33 de pose y 21 por mano. Al graficar un cuadro se observa que la mano queda agrupada cerca de la cara, algo esperable en *fingerspelling*.

* **[02_limpieza_preprocesamiento](../notebooks/02_limpieza_preprocesamiento.ipynb):** los datos ya contienen coordenadas normalizadas por MediaPipe, por lo que el principal problema no es la suciedad, sino las detecciones faltantes. Estas afectan especialmente a la mano derecha, con aproximadamente 54.5 % de valores `NaN`, frente a aproximadamente 0.7 % en el rostro. Los faltantes se trataron mediante *forward-fill* dentro de cada secuencia para evitar que la interpolación generara posiciones artificiales entre letras. Además, se redujeron las variables de 1,630 a 226 columnas eliminando los 468 puntos del rostro, que representan aproximadamente el 86 % de las columnas y aportan poca información al deletreo. Finalmente, se excluyeron las secuencias con detección de mano en menos del 10 % de sus cuadros y la muestra resultante se almacenó en `data/processed/`.

* **[03_analisis_univariado](../notebooks/03_analisis_univariado.ipynb):** `train.csv` no trae variables numéricas propias (solo identificadores), así que el análisis se apoya en tres derivadas: longitud de frase, longitud de secuencia y porcentaje de frames sin mano detectada. La frase está acotada y es casi simétrica (1 a 31 caracteres, media 17.8), mientras que la secuencia es asimétrica a la derecha (mediana ~147 frames, cola por encima de 700): dos personas deletreando la misma cantidad de caracteres pueden generar secuencias de duración muy distinta. En las categóricas se repite el mismo patrón de desbalance ya visto en 01: participantes entre 1 y 1,535 secuencias, y entre caracteres, unas pocas letras (e, a, o, r, n, t) concentran la mayoría de las apariciones mientras que j, q, z y varios símbolos casi no tienen muestras.

* **[04_analisis_bivariado](../notebooks/04_analisis_bivariado.ipynb):** se comprobó una correlación positiva, \(r \approx 0.5979\), entre la longitud de la frase y el número de frames utilizados. Frases con mayor cantidad de caracteres generan secuencias más largas. Asimismo, se observó una relación estadísticamente negativa de \(r \approx -0.1883\) entre la proporción de *landmarks* faltantes y la longitud de la secuencia. Esto indica que, si la cámara pierde los puntos de referencia, la secuencia registrada tiende a ser más corta.

  La matriz de correlación (*heatmap*) reveló que la relación cruzada entre la longitud de la frase y el promedio de información perdida (`NaN`) es lo suficientemente baja como para ser descartada del enfoque principal.

  Mediante el método del rango intercuartílico (IQR), se detectaron 28 secuencias atípicas sobre un total de 998, aproximadamente el 3 % de datos atípicos. Los límites estadísticos normales se establecieron entre 0 y 359.4 frames, y entre 0 y 37 caracteres, interpretando los límites inferiores negativos como un tope natural de cero.

* **[05_visualizaciones](../notebooks/05_visualizaciones.ipynb)**: las visualizaciones confirmaron que la duración de las secuencias presenta un sesgo hacia la derecha: la mayoría contiene entre 50 y 250 frames, con una mediana global de 147 y algunos valores extremos cercanos a 750. 

También hubo diferencias importantes entre participantes, cuyas medianas varían aproximadamente entre 75.5 y 241 frames. 

La longitud de las frases se concentra entre 10 y 30 caracteres, con una moda marcada de 12, y mantiene una correlación positiva moderada con la cantidad de frames ((r \approx 0.602)). 

Además, la frecuencia de caracteres está desbalanceada: predominan letras como e y a, mientras que j, z y varios símbolos cuentan con pocas observaciones. 

En conjunto, estos resultados indican que un futuro modelo deberá aceptar secuencias de longitud variable, considerar las diferencias de velocidad entre participantes y prestar especial atención a los caracteres menos frecuentes.


## Problemas de calidad de datos encontrados

- **Landmarks faltantes concentrados en las manos:** el rostro casi no falta (~0.7 %), pero la mano derecha falta en ~54.5 % de los frames. No es ruido ni corrupción, es ausencia de detección (mano fuera de cuadro, oclusión, movimiento rápido) y afecta justo a la región más informativa para el deletreo.
- **Desbalance entre participantes:** de 1 a 1,535 secuencias por persona (mediana 794); los 10 participantes con más datos solo concentran el 14.5 % del total, pero los que tienen pocas secuencias quedan casi sin representación.
- **Desbalance entre caracteres:** el alfabeto efectivo tiene 59 símbolos, pero unas pocas letras (e, a, o, r, n, t) concentran la mayoría de las apariciones; j, q, z y varios símbolos aparecen en una fracción mínima.
- **Longitud de secuencia muy variable y sesgada a la derecha:** mediana ~147 frames pero con cola hasta 750+, mientras que la frase objetivo es corta y casi simétrica (1-31 caracteres). El mismo contenido puede tomar duraciones muy distintas.
- **Outliers:** con el criterio de 1.5·IQR se detectaron 28 secuencias atípicas de 998 (~3 %) en la muestra, con límites normales de 0-359 frames y 0-37 caracteres.
- **Relación cruzada débil:** la correlación entre longitud de frase y proporción de landmarks faltantes resultó baja, así que no se puede usar una variable para explicar la otra.

## Conclusiones sobre los siguientes pasos

- **Tratar los landmarks faltantes sin inventar movimiento:** se usó forward-fill dentro de cada secuencia (no interpolación) para no generar posiciones artificiales entre letras, y se descartaron secuencias con mano detectada en menos del 10 % de sus frames.
- **Reducir columnas irrelevantes:** los 468 puntos de rostro (86 % de las columnas) aportan poco al deletreo; ya se redujo la muestra de 1,630 a 226 columnas en `02_limpieza_preprocesamiento.ipynb`.
- **Separar entrenamiento/validación por participante**, no de forma aleatoria, para no medir si un futuro modelo memorizó el estilo de deletreo de unos pocos participantes muy representados.
- **Aceptar longitud de secuencia variable** (recorte o relleno) en vez de asumir una duración fija, dado el sesgo y la cola larga encontrados en 01, 03 y 05.
- **Reportar desempeño por carácter y no solo en promedio**, ya que el desbalance de caracteres haría que el promedio general esconda un mal desempeño en letras poco frecuentes (j, q, z, símbolos).
- Todo esto es evidencia para la limpieza/preprocesamiento; el entrenamiento y evaluación de un modelo de transcripción quedan fuera del alcance de este análisis exploratorio.
