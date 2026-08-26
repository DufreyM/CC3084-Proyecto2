# Investigación del tema (Actividad 1 de la guía)

**Responsable principal:** Jose Donado

> Archivo independiente — no depende de datos ni de código, se puede avanzar
> desde el día 1. Sirve de base conceptual para interpretar lo que se
> encuentre en el EDA.

## 1. ¿Qué es el deletreo manual (fingerspelling) en ASL?

<!-- TODO(equipo): Explicar en qué consiste el fingerspelling dentro del
lenguaje de señas americano (ASL): quiénes lo usan y para qué (nombres
propios, palabras sin seña establecida, deletreo de siglas), cómo se compone
cada letra con la mano, y por qué es un problema distinto al reconocimiento
de señas de palabra completa. -->

TODO

## 2. Cómo se capturan los datos: MediaPipe y landmarks

<!-- TODO(equipo): Investigar cómo funciona MediaPipe Holistic (o el modelo
usado por la competencia) para extraer landmarks de manos, cara y pose a
partir de video. Explicar qué representa cada landmark (x, y, z), por qué
puede haber landmarks faltantes (mano fuera de cuadro, oclusión), y por qué
se usan landmarks en vez de los frames de video crudos (privacidad, tamaño
de los datos, dato ya "pre-procesado" para el modelo). -->

TODO

## 3. Técnicas para reconocer patrones en secuencias (landmarks -> texto)

<!-- TODO(equipo): Investigar qué técnicas se usan típicamente para este tipo
de problema de secuencia-a-secuencia / secuencia-a-texto, por ejemplo:
- Modelos recurrentes (LSTM/GRU) sobre secuencias de landmarks
- Transformers / attention aplicados a secuencias temporales
- CTC loss (Connectionist Temporal Classification), muy usado cuando no hay
  alineación frame-a-caracter exacta (como en este reto)
- Normalización y aumentación de datos de landmarks (rotación, escala,
  reflejo especular para manos zurdas/diestras)
Enfocarse en qué patrones debe aprender el algoritmo (trayectoria de los
dedos en el tiempo) más que en la implementación exacta. -->

TODO

## 4. Referencias

<!-- TODO(equipo): Listar aquí los artículos, posts o documentación de
Kaggle/MediaPipe consultados. -->

- TODO
