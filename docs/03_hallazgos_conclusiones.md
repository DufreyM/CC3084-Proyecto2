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

* TODO (de [03_analisis_univariado](../notebooks/03_analisis_univariado.ipynb))

* **[04_analisis_bivariado](../notebooks/04_analisis_bivariado.ipynb):** se comprobó una correlación positiva, \(r \approx 0.5979\), entre la longitud de la frase y el número de frames utilizados. Frases con mayor cantidad de caracteres generan secuencias más largas. Asimismo, se observó una relación estadísticamente negativa de \(r \approx -0.1883\) entre la proporción de *landmarks* faltantes y la longitud de la secuencia. Esto indica que, si la cámara pierde los puntos de referencia, la secuencia registrada tiende a ser más corta.

  La matriz de correlación (*heatmap*) reveló que la relación cruzada entre la longitud de la frase y el promedio de información perdida (`NaN`) es lo suficientemente baja como para ser descartada del enfoque principal.

  Mediante el método del rango intercuartílico (IQR), se detectaron 28 secuencias atípicas sobre un total de 998, aproximadamente el 3 % de datos atípicos. Los límites estadísticos normales se establecieron entre 0 y 359.4 frames, y entre 0 y 37 caracteres, interpretando los límites inferiores negativos como un tope natural de cero.

* **[05_visualizaciones](../notebooks/05_visualizaciones.ipynb)**: las visualizaciones confirmaron que la duración de las secuencias presenta un sesgo hacia la derecha: la mayoría contiene entre 50 y 250 frames, con una mediana global de 147 y algunos valores extremos cercanos a 750. 

También hubieron diferencias importantes entre participantes, cuyas medianas varían aproximadamente entre 75.5 y 241 frames. 

La longitud de las frases se concentra entre 10 y 30 caracteres, con una moda marcada de 12, y mantiene una correlación positiva moderada con la cantidad de frames ((r \approx 0.602)). 

Además, la frecuencia de caracteres está desbalanceada: predominan letras como e y a, mientras que j, z y varios símbolos cuentan con pocas observaciones. 

En conjunto, estos resultados indican que un futuro modelo deberá aceptar secuencias de longitud variable, considerar las diferencias de velocidad entre participantes y prestar especial atención a los caracteres menos frecuentes.


## Problemas de calidad de datos encontrados

<!-- TODO(equipo): Valores faltantes, outliers, secuencias corruptas o
inconsistentes, desbalance entre participantes o longitudes de frase, etc. -->

TODO

## Conclusiones sobre los siguientes pasos

<!-- TODO(equipo): A partir de lo encontrado, ¿qué implicaría esto para un
futuro modelo? (p. ej. normalizar por participante, descartar/rellenar
frames con landmarks faltantes, limitar longitud de secuencia, balancear por
longitud de frase, etc.) -->

TODO
