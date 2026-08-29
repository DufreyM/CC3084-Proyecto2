# Investigación del tema (Actividad 1)

**Responsable principal:** José Donado

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

## 2. Cómo se capturan los datos: MediaPipe y landmarks

<!-- TODO(equipo): Investigar cómo funciona MediaPipe Holistic (o el modelo
usado por la competencia) para extraer landmarks de manos, cara y pose a
partir de video. Explicar qué representa cada landmark (x, y, z), por qué
puede haber landmarks faltantes (mano fuera de cuadro, oclusión), y por qué
se usan landmarks en vez de los frames de video crudos (privacidad, tamaño

MediaPipe es un marco de trabajo de código abierto de Google que permite aplicar modelos de Inteligencia Artificial para el procesamiento de video en tiempo real.Su principal ventaja es la eficiencia: es tan ligero que puede ejecutarse directamente en dispositivos móviles, páginas web o computadoras sin necesidad de tarjetas gráficas (GPU) de alta potencia.


Gomaa y El-Khoribi (2026) explican que para entender el movimiento de la mano, MediaPipe utiliza un modelo de DL que detecta una estructura geométrica llamada Hand Landmarks (puntos de referencia).El sistema localiza exactamente 21 puntos tridimensionales (X, Y, Z) en cada mano, distribuidos estratégicamente en las articulaciones clave:Punto 0: La muñeca (origen de la mano). 4 puntos por dedo: Cada uno de los 5 dedos (pulgar, índice, medio, anular y meñique) tiene asignados 4 puntos que corresponden a la base, las articulaciones intermedias y la punta del dedo.


El flujo de captura de datos se divide en tres etapas continuas que ocurren en milisegundos:
- Detección de la Palma (Palm Detection): El algoritmo primero analiza la imagen completa de la cámara para encontrar una mano. Como las palmas y los puños son zonas relativamente estables y fáciles de identificar, el sistema recorta esa región exacta de la imagen y descarta el fondo (muebles, ropa, etc.).
- Predicción de Puntos (Hand Landmark Model): Sobre la región recortada de la mano, el modelo predice la ubicación exacta de los 21 puntos de referencia.
- Extracción de Coordenadas 3D: Por cada uno de los 21 puntos, MediaPipe genera tres valores:X e Y: La posición horizontal y vertical del punto dentro de la imagen (coordenadas de la pantalla). Z: La profundidad relativa. 

Con esto, logra estimar qué tan cerca o lejos está cada articulación respecto a la muñeca (basándose en el tamaño de la mano). Al transformar el video en un flujo continuo de coordenadas numéricas, los algoritmos de IA ya no ven colores ni luces; solo ven un "esqueleto" que se mueve en el tiempo, facilitando el entrenamiento de modelos para identificar las letras del fingerspelling.

## 3. Técnicas para reconocer patrones en secuencias (landmarks -> texto)

Con el esqueleto se pasa al procesamiento de secuencias de forma continua. Para ello la IA utiliza estas estrategias:
1. RNN y LSTM (Redes Neuronales Recurrentes)
RNN son redes que t ienen una "memoria" interna que les permite recordar lo que pasó en fotogramas anteriores. LSTM (Long Short-Term Memory) es la variante más utilizada. Resuelve el problema de las RNN básicas, que olvidan rápidamente la información del pasado. Una LSTM mantiene una "línea de vida" que preserva la información relevante a largo plazo. Luego, la red recibe las coordenadas de la mano fotograma por fotograma. Al procesar la letra actual, la LSTM "recuerda" la posición de la mano un segundo antes. Esto es crucial para detectar la coarticulación (saber si un dedo se está doblando debido a la letra anterior).

2. CNN 1D (Redes Neuronales Convolucionales Unidimensionales)
Las CNN 1D se aplican sobre datos que se desplazan en una sola dirección temporal.En lugar de buscar patrones visuales en una foto (como ojos o narices), la CNN 1D desliza "filtros" a lo largo de la línea de tiempo de las coordenadas de los landmarks.
Con ello ya captura micro-movimientos locales muy rápidos. Por ejemplo, identifica la velocidad exacta con la que baja un dedo para formar la letra J. Al ser operaciones matemáticas simples, las CNN 1D son extremadamente rápidas y eficientes para ejecutarse en tiempo real.

3. Transformers (Modelos basados en Atención)

Es la arquitectura de CHATGPT. Los Transformers han reemplazado en gran medida a las LSTM porque no procesan los datos paso a paso, sino que analizan toda la secuencia al mismo tiempo.

Esto permite al modelo calcular matemáticamente qué partes de la secuencia se relacionan más entre sí, sin importar qué tan separadas estén en el tiempo.

El Transformer puede analizar una palabra entera deletreada de corrido. Entiende el contexto global del movimiento y puede corregir errores basándose en las letras vecinas. Si el sistema detecta con un 90% de certeza las letras H-O-U-S- y la última letra está muy distorsionada entre una E y una O, el mecanismo de atención sabrá que contextualmente la palabra más probable en inglés es HOUSE, corrigiendo la salida de texto automáticamente (Al-Qaderi & El-Sabaa, 2026).

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

Al-Qaderi, M., & El-Sabaa, H. (2026). American Sign Language recognition for alphabets using MediaPipe and LSTM [Reconocimiento de alfabetos en la Lengua de Señas Americana utilizando MediaPipe y LSTM]. ResearchGate. https://www.researchgate.net/publication/366722112_American_Sign_Language_Recognition_for_Alphabets_Using_MediaPipe_and_LSTM


Pitsikalis, V., Katsamanis, A., & Maragos, P. (2024). Tracking and recognition of fingerspelling from videos [Seguimiento y reconocimiento de deletreo manual a partir de videos]. University of Thessaly Institutional Repository. https://ir.lib.uth.gr/xmlui/bitstream/handle/11615/59441/25386.pdf

<!-- TODO(equipo): Listar aquí los artículos, posts o documentación de
