# Hallazgos y conclusiones (Actividad 5 de la guía)

**Responsable principal:** Todo el equipo (integrar al final, una vez cerrados 01-05)

> Archivo independiente. Se llena al final, cuando los notebooks de EDA
> (01-05) ya tengan resultados. Corresponde al rubro **Hallazgos y
> conclusiones (20 pts)**.

## Resumen de hallazgos

<!-- TODO(equipo): Resumir en viñetas los hallazgos más importantes de cada
notebook de EDA: estructura de los datos (01), decisiones de limpieza y su
justificación (02), estadística descriptiva y tablas de frecuencia (03),
relaciones entre variables y outliers (04), y lo que muestran los gráficos
(05). No repetir todo el detalle, solo lo accionable/relevante. -->

- **[01_estructura_datos](../notebooks/01_estructura_datos.ipynb):** train.csv trae 67,208
  secuencias de 94 participantes, pero muy desbalanceados entre si -- van de 1 a 1535
  secuencias por persona (mediana 794), asi que cualquier comparacion por participante
  hay que tomarla con cuidado. Las frases van de 1 a 31 caracteres (mediana 17). Cada
  parquet de landmarks trae unas 1000 secuencias, indexadas por sequence_id (no es
  columna), con 1630 columnas por cuadro: 468 de rostro, 33 de pose y 21 por mano. Al
  graficar un cuadro se ve que la mano queda agrupada cerca de la cara, algo esperable
  en fingerspelling.
  
- * **[02_limpieza_preprocesamiento](../notebooks/02_limpieza_preprocesamiento.ipynb):** los datos ya contienen coordenadas normalizadas por MediaPipe, por lo que el principal problema no es la suciedad, sino las detecciones faltantes, estas afectan especialmente a la mano derecha, con ~54.5 % de valores `NaN`, frente a ~0.7 % en el rostro, los faltantes se trataron mediante *forward-fill* dentro de cada secuencia para evitar que la interpolación generara posiciones artificiales entre letras, además, se redujeron las variables de 1630 a 226 columnas eliminando los 468 puntos del rostro, que representan ~86 % de las columnas y aportan poca información al deletreo, finalmente, se excluyeron las secuencias con detección de mano en menos del 10 % de sus cuadros y la muestra resultante se almacenó en `data/processed/`


- TODO (de [03_analisis_univariado](../notebooks/03_analisis_univariado.ipynb))
- TODO (de [04_analisis_bivariado](../notebooks/04_analisis_bivariado.ipynb))
- TODO (de [05_visualizaciones](../notebooks/05_visualizaciones.ipynb))

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
