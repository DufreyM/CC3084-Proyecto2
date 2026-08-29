# Investigación del tema (Actividad 1 de la guía)

**Responsable principal:** Jose Donado

> Archivo independiente — no depende de datos ni de código, se puede avanzar
> desde el día 1. Sirve de base conceptual para interpretar lo que se
> encuentre en el EDA.

## 1. ¿Qué es el deletreo manual (fingerspelling) en ASL?
Según la Asociación de Niños Sordos (ASDC), el deletreo manual en ASL se refiere al uso de las manos para representar individualmente las letras del alfabeto del idioma inglés. 

El deletreo manual se utiliza cotidianamente por la comunidad sorda, intérpretes y estudiantes de ASL. Es parte del lenguaje de señas (ASL), el cual se compone de reglas gramaticales y su vocabulario.

Se utiliza para tres propósitos:
- Nombres propios: común para referirse a países, personas, películas, libros o ciudades.
- Términos técnicos o palabras sin seña establecida.
- Deletreo de siglas y acrónimos.

Por otro lado, el alfabeto dactilológico se realiza con una sola mano colocada a la altura del hombro y con la palma generalmente viendo hacia el receptor del mensaje.
Cada letra tiene tres elementos:
- Configuración de la mano: La forma específica que adoptan los dedos (flexionados, extendidos o cruzados). Por ejemplo, la A es un puño cerrado con el pulgar a un lado; la V extiende los dedos índice y medio en forma de victoria.
- Orientación de la palma: Hacia dónde apunta la palma de la mano. La mayoría miran al frente, pero letras como la G y la H apuntan de lado, y la P apunta hacia abajo.
- Movimiento: La gran mayoría de las letras son estáticas, pero la J dibuja una curva en el aire con el dedo meñique y la Z dibuja la forma de la letra con el dedo índice.


A diferencia del reconocimiento de señas comunes, el problema es que el deletreo es rapidísimo: de 5 a 6 letras por segundo. Por ello, se dificulta su lectura cuando se traslapan los dedos entre sí y es difícil de interpretar para algoritmos de aprendizaje debido a la limitación de ser entrenados con videos en dos dimensiones. 

Hay letras similares, así como la M, N y T que agravan la situación. Pero sobre todo, se requiere de una mayor resolución de imagen y fotogramas por segundo para analizar las señales a nivel casi microscópico dado a que no se tiene la forma del cuerpo, sino simplemente de una porción de la mano.

<!-- TODO(equipo): Explicar en qué consiste el fingerspelling dentro del
lenguaje de señas americano (ASL): quiénes lo usan y para qué (nombres
propios, palabras sin seña establecida, deletreo de siglas), cómo se compone
cada letra con la mano, y por qué es un problema distinto al reconocimiento

TODO

## 2. Cómo se capturan los datos: MediaPipe y landmarks

<!-- TODO(equipo): Investigar cómo funciona MediaPipe Holistic (o el modelo
usado por la competencia) para extraer landmarks de manos, cara y pose a
partir de video. Explicar qué representa cada landmark (x, y, z), por qué
puede haber landmarks faltantes (mano fuera de cuadro, oclusión), y por qué
se usan landmarks en vez de los frames de video crudos (privacidad, tamaño

MediaPipe es un marco de trabajo de código abierto de Google que permite aplicar modelos de Inteligencia Artificial para el procesamiento de video en tiempo real.Su principal ventaja es la eficiencia: es tan ligero que puede ejecutarse directamente en dispositivos móviles, páginas web o computadoras sin necesidad de tarjetas gráficas (GPU) de alta potencia.


Para entender el movimiento de la mano, MediaPipe utiliza un modelo de DL que detecta una estructura geométrica llamada Hand Landmarks (puntos de referencia).El sistema localiza exactamente 21 puntos tridimensionales (X, Y, Z) en cada mano, distribuidos estratégicamente en las articulaciones clave:Punto 0: La muñeca (origen de la mano). 4 puntos por dedo: Cada uno de los 5 dedos (pulgar, índice, medio, anular y meñique) tiene asignados 4 puntos que corresponden a la base, las articulaciones intermedias y la punta del dedo.


El flujo de captura de datos se divide en tres etapas continuas que ocurren en milisegundos:
- Detección de la Palma (Palm Detection): El algoritmo primero analiza la imagen completa de la cámara para encontrar una mano. Como las palmas y los puños son zonas relativamente estables y fáciles de identificar, el sistema recorta esa región exacta de la imagen y descarta el fondo (muebles, ropa, etc.).
- Predicción de Puntos (Hand Landmark Model): Sobre la región recortada de la mano, el modelo predice la ubicación exacta de los 21 puntos de referencia.
- Extracción de Coordenadas 3D: Por cada uno de los 21 puntos, MediaPipe genera tres valores:X e Y: La posición horizontal y vertical del punto dentro de la imagen (coordenadas de la pantalla). Z: La profundidad relativa. 

Con esto, logra estimar qué tan cerca o lejos está cada articulación respecto a la muñeca (basándose en el tamaño de la mano). Al transformar el video en un flujo continuo de coordenadas numéricas, los algoritmos de IA ya no ven colores ni luces; solo ven un "esqueleto" que se mueve en el tiempo, facilitando el entrenamiento de modelos para identificar las letras del fingerspelling.

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

TODO

## 4. Referencias

<!-- TODO(equipo): Listar aquí los artículos, posts o documentación de

- TODO
