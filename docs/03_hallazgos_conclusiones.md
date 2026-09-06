# Hallazgos y conclusiones (Actividad 5 de la guía)

**Responsable principal:** Todo el equipo (integrar al final, una vez cerrados 01-05)

> Archivo independiente. Se llena al final, cuando los notebooks de EDA
> (01-05) ya tengan resultados. Corresponde al rubro **Hallazgos y
> conclusiones (20 pts)**.

## Síntesis de la investigación técnica
El deletreo manual en ASL implica el uso de 5 a 6 letras por segundo, similar a la rapidez encontrada con el idioma español (verbal), que según Deutsche Welle (2025) utiliza 7.82 sílabas por segundo, es decir: un poco más de 12 letras por segundo. El deletreo del ASL se utiliza para nombres propios, siglas o términos técnicos.

El proyecto presente emplea MediaPipe para extraer 21 puntos tridimensionales (o landmarks) por mano en lugar de procesar video crudo. Esto protege la privacidad, reduce significativamente el tamaño de los datos y elimina interferencias visuales como la iluminación o el fondo.  

Por otro lado, se ha de tomar en cuenta que la pérdida de landmarks (valores nulos) no es un error imprevisto, sino la ausencia de detección por movimiento rápido, oclusión o porque la mano sale del encuadre. En los datos de muestra, la mano derecha está ausente en más del 54.5% de los fotogramas.  

Para traducir estos landmarks a texto, las arquitecturas de aprendizaje automático ideales incluyen LSTM, CNN 1D, Transformers y CTC (Connectionist Temporal Classification), siendo esta última fundamental para secuencias continuas sin límite difuso entre letras.  

## Resumen de hallazgos

<!-- TODO(equipo): Resumir en viñetas los hallazgos más importantes de cada
notebook de EDA: estructura de los datos (01), decisiones de limpieza y su
justificación (02), estadística descriptiva y tablas de frecuencia (03),
relaciones entre variables y outliers (04), y lo que muestran los gráficos

- __[01_estructura_datos](../notebooks/01_estructura_datos.ipynb):__ train.csv trae 67,208
   secuencias de 94 participantes, pero muy desbalanceados entre si -- van de 1 a 1535
   secuencias por persona (mediana 794), asi que cualquier comparacion por participante
   hay que tomarla con cuidado. Las frases van de 1 a 31 caracteres (mediana 17). Cada
   parquet de landmarks trae unas 1000 secuencias, indexadas por sequence_id (no es
   columna), con 1630 columnas por cuadro: 468 de rostro, 33 de pose y 21 por mano. Al
   graficar un cuadro se ve que la mano queda agrupada cerca de la cara, algo esperable
   en fingerspelling.
- TODO (de [02_limpieza_preprocesamiento](../notebooks/02_limpieza_preprocesamiento.ipynb))
- TODO (de [03_analisis_univariado](../notebooks/03_analisis_univariado.ipynb))


- [04_analisis_bivariado](../notebooks/04_analisis_bivariado.ipynb)

Se comprobó una correlación positiva, $r \approx 0.5979$,  entre la longitud de la frase y el número de frames utilizados (obviamente). Frases con mayor cantidad de caracteres generan secuencias más largas.  Asimismo, se observó una relación estadísticamente negativa de $r \approx -0.1883$ entre la proporción de landmarks faltantes y la longitud de la secuencia. Esto indica que si la cámara pierde los puntos de referencia, la secuencia registrada tiende a ser más corta.  

La matriz de correlación (heatmap) reveló que la relación cruzada entre la longitud de la frase y el promedio de información perdida (NaNs) es lo suficientemente baja como para ser descartada del enfoque principal.  Mediante el método del rango de intercuartiles (IQR), se detectaron 28 secuencias atípicas sobre un total de 998 (como el 3% de datos atípicos). Los límites estadísticos normales se establecieron entre 0 y 359.4 frames, y entre 0 y 37 caracteres (interpretando los límites inferiores negativos como un tope natural de cero).  

- TODO (de [05_visualizaciones](../notebooks/05_visualizaciones.ipynb))

## Problemas de calidad de datos encontrados

<!-- TODO(equipo): Valores faltantes, outliers, secuencias corruptas o

TODO

## Conclusiones sobre los siguientes pasos

<!-- TODO(equipo): A partir de lo encontrado, ¿qué implicaría esto para un
futuro modelo? (p. ej. normalizar por participante, descartar/rellenar
frames con landmarks faltantes, limitar longitud de secuencia, balancear por

TODO
