# Planteamiento del problema

## Situación problemática

En lengua de señas, el deletreo de letras o fingerspelling es un componente del lenguaje de señas estadounidense (ASL). Se usa para decir nombres propios, lugares, términos técnicos y otras palabras que no tengan una seña en específico.

Reconocer automaticamente el deletreo manual puede ser difícil porque las señas forman parte de secuencias de movimientos, y no imágenes independientes. Su significado varía por velocidad, posición, orientación y estilo, mientras que las frases y secuencias pueden llegar a ser muy variables. Durante la captura pueden existir cortes en imagenes o señas poco confiables debido a bloqueos o movimientos demasiado rápidos.

## Problema científico

¿Es posbile caracterizar las secuencias de manos, rostro y pose extraídas con MediaPipe para determinar si cuentan con la información y calidad suficiente para transcribir a texto frases deletreadas con ASL, considerando la variabilidad entre participantes, diferencias de duraciones y señalizaciones faltantes? 

## Objetivos

### Objetivo general

Realizar un análisis exploratorio (EDA) de los datos obtenidos a través de la [competencia Google - Amercian Sign Language Fingerspelling Recognition](https://www.kaggle.com/competitions/asl-fingerspelling) para caracterizar la estructura, distribución y calidad de las secuencias de señalizaciones e identificar datos que deberán ser considerados en la etapa de limpieza, pre procesamiento y posterior modelado para transcripción automática del deletreo.

### Objetivos específicos

1. Describir la estructura del conjunto de datos mediante la identificación de sus variables, tipos de datos, cantidad de participantes, frases, secuencias, frames y puntos corporales clave disponibles.
2. Analizar las distribuciones de la duración de las secuencias y de la longitud de las frases objetivo, identificando valores atípicos y diferencias relevantes entre participantes.
3. Cuantificar y visualizar la presencia de valores faltantes en los landmarks de manos, rostro y pose, con el fin de reconocer patrones de pérdida de información y determinar qué regiones corporales presentan mayor disponibilidad de datos.
