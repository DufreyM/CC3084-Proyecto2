# Planteamiento del problema

## Situación problemática

El deletreo manual (fingerspelling) es un componente importante del lenguaje de señas estadounidense (ASL), ya que permite representar letra por letra palabras que no cuentan con una seña propia, como nombres, direcciones, marcas, siglas y términos técnicos. Se estima que entre el 12 % y el 35 % del discurso en ASL corresponde a deletreo manual (Padden y Gunsauls, 2003), lo que evidencia su relevancia en la comunicación cotidiana.

Su reconocimiento automático representa un reto distinto al de la clasificación convencional de imágenes, debido a que el significado no depende únicamente de la forma de la mano, sino también de su movimiento a lo largo del tiempo. La coarticulación entre letras, la velocidad de ejecución, las diferencias entre participantes y factores como la iluminación, el fondo, el encuadre y la oclusión pueden generar variaciones importantes en los datos.

En 2023, Google, en colaboración con la Deaf Professional Arts Network, publicó en Kaggle la competencia American Sign Language Fingerspelling Recognition, cuyo conjunto de datos contiene más de tres millones de caracteres realizados por más de cien firmantes sordos. Por razones de privacidad y almacenamiento, los datos no se distribuyen como video, sino como secuencias de landmarks obtenidos mediante MediaPipe, con coordenadas de puntos de las manos, el rostro y el cuerpo para cada cuadro.

Esta representación convierte el conjunto en datos de series temporales y, al mismo tiempo, introduce posibles valores faltantes y variaciones en la calidad de los puntos detectados. Por ello, antes de definir estrategias de procesamiento o modelado, es necesario caracterizar aspectos como la duración de las secuencias, la distribución de los valores faltantes, las regiones corporales involucradas, los patrones de movimiento y las diferencias entre participantes. Sin esta exploración, decisiones como qué variables conservar, cómo normalizar las coordenadas o qué secuencias utilizar podrían basarse en supuestos no comprobados y afectar la reproducibilidad del análisis.

## Problema científico

¿Cuáles son las características, la calidad y la variabilidad de las secuencias de landmarks de manos, rostro y pose del conjunto de datos ASL Fingerspelling, y qué aspectos deben considerarse para su limpieza, preprocesamiento y posterior uso en la transcripción automática de frases deletreadas?

De la pregunta principal se derivan las siguientes preguntas específicas:

1. ¿Qué variables y observaciones componen el conjunto, de qué tipo son y en
   qué unidades están expresadas?
2. ¿Cómo se distribuyen la duración de las secuencias (número de cuadros) y la
   longitud de las frases objetivo (número de caracteres), y existe entre ambas
   una relación estable que permita anticipar cuántos cuadros corresponden a
   cada carácter?
3. ¿Qué proporción de valores faltantes presenta cada región anatómica (mano
   izquierda, mano derecha, rostro, pose) y cómo se distribuyen esos faltantes
   a lo largo de cada secuencia?
4. ¿Existen diferencias sistemáticas entre participantes —en duración, en
   mano dominante o en disponibilidad de landmarks— lo bastante grandes como
   para exigir una normalización previa al modelado?
5. ¿Qué observaciones deben excluirse o corregirse antes de modelar, y bajo
   qué criterio explícito?

Delimitación: El estudio se limita al análisis exploratorio de los datos. No incluye el entrenamiento ni la evaluación de modelos de transcripción.

## Objetivos

### Objetivo general

Caracterizar, mediante un análisis exploratorio de datos, la estructura, la
calidad y la variabilidad de las secuencias de landmarks del conjunto
*American Sign Language Fingerspelling Recognition* (Google, 2023), con el fin
de determinar y justificar con evidencia empírica las transformaciones de
limpieza y preprocesamiento que el conjunto requiere antes de la construcción
de un modelo de transcripción automática del deletreo manual.

### Objetivos específicos

1. **Inventariar la estructura del conjunto de datos**, reportando el número
   de observaciones, el número y el tipo de cada variable, y los conteos de
   participantes, secuencias y cuadros de la muestra analizada.
   *Criterio de cumplimiento:* una tabla de metadatos con nombre, tipo, rango
   y unidad de cada variable, y los conteos anteriores documentados.

2. **Cuantificar la completitud de los datos**, calculando el porcentaje de
   valores faltantes por región anatómica (mano izquierda, mano derecha,
   rostro y pose) y su distribución a lo largo de la secuencia.
   *Criterio de cumplimiento:* porcentajes de faltantes por región y por
   posición relativa en la secuencia, y un umbral explícito a partir del cual
   una secuencia se considera inutilizable.

3. **Describir estadísticamente las variables de interés** —duración de la
   secuencia, longitud de la frase objetivo y número de secuencias por
   participante— mediante medidas de tendencia central y dispersión (media,
   mediana, desviación estándar, cuartiles y rango intercuartílico) y tablas
   de frecuencia para las variables categóricas, identificando valores
   atípicos bajo un criterio declarado (regla de 1.5 · RIC).
   *Criterio de cumplimiento:* tabla de resumen numérico, tablas de frecuencia
   y listado de atípicos clasificados como error de captura o variación
   legítima.

4. **Evaluar la relación entre la duración de la secuencia, la cantidad de
   cuadros con mano detectada y la longitud de la frase objetivo**, mediante
   coeficientes de correlación y análisis bivariado, para establecer si existe
   una razón cuadros-por-carácter estable y si se sostiene entre participantes.
   *Criterio de cumplimiento:* matriz de correlaciones y una conclusión
   explícita sobre la existencia o no de dicha razón.

5. **Elaborar gráficos exploratorios** (histogramas, diagramas de caja y
   bigotes, diagramas de dispersión y visualización de la trayectoria de la
   mano en el tiempo) que evidencien la forma de las distribuciones, los
   valores atípicos y los patrones de datos faltantes.
   *Criterio de cumplimiento:* al menos un gráfico por cada uno de los
   objetivos 2, 3 y 4, exportado a `reports/figures/` e interpretado en el
   texto.

### Trazabilidad de los objetivos

| Objetivo | Actividad de la guía | Archivo del repositorio |
|---|---|---|
| 1 | 4a — variables, observaciones y tipos | `notebooks/01_estructura_datos.ipynb` |
| 2 | 3 — limpieza y preprocesamiento | `notebooks/02_limpieza_preprocesamiento.ipynb` |
| 3 | 4b — resumen numérico y frecuencias | `notebooks/03_analisis_univariado.ipynb` |
| 4 | 4c — cruce de variables | `notebooks/04_analisis_bivariado.ipynb` |
| 5 | 4d — gráficos exploratorios | `notebooks/05_visualizaciones.ipynb` |


## Referencias

- Google & Deaf Professional Arts Network. (2023). *Google — American Sign
  Language Fingerspelling Recognition* [Competencia de Kaggle].
  https://www.kaggle.com/competitions/asl-fingerspelling
- Graves, A., Fernández, S., Gomez, F. y Schmidhuber, J. (2006). Connectionist
  Temporal Classification: Labelling unsegmented sequence data with recurrent
  neural networks. *Proceedings of the 23rd International Conference on
  Machine Learning (ICML)*, 369–376.
- Lugaresi, C., Tang, J., Nash, H., McClanahan, C., Uboweja, E., Hays, M.,
  et al. (2019). MediaPipe: A framework for building perception pipelines.
  *arXiv:1906.08172*.
- Padden, C. y Gunsauls, D. C. (2003). How the alphabet came to be used in a
  sign language. *Sign Language Studies*, 4(1), 10–33.
- Shi, B., Del Rio, A. M., Keane, J., Michaux, J., Brentari, D., Shakhnarovich,
  G. y Livescu, K. (2018). American Sign Language fingerspelling recognition in
  the wild. *IEEE Spoken Language Technology Workshop (SLT)*, 145–152.
- Zhang, F., Bazarevsky, V., Vakunov, A., Tkachenka, A., Sung, G., Chang, C.-L.
  y Grundmann, M. (2020). MediaPipe Hands: On-device real-time hand tracking.
  *arXiv:2006.10214*.