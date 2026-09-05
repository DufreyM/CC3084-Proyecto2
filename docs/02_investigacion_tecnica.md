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

MediaPipe es un marco de trabajo de código abierto de Google que permite aplicar modelos de Inteligencia Artificial para el procesamiento de video en tiempo real.Su principal ventaja es la eficiencia: es tan ligero que puede ejecutarse directamente en dispositivos móviles, páginas web o computadoras sin necesidad de tarjetas gráficas (GPU) de alta potencia.


Gomaa y El-Khoribi (2026) explican que para entender el movimiento de la mano, MediaPipe utiliza un modelo de DL que detecta una estructura geométrica llamada Hand Landmarks (puntos de referencia).El sistema localiza exactamente 21 puntos tridimensionales (X, Y, Z) en cada mano, distribuidos estratégicamente en las articulaciones clave:Punto 0: La muñeca (origen de la mano). 4 puntos por dedo: Cada uno de los 5 dedos (pulgar, índice, medio, anular y meñique) tiene asignados 4 puntos que corresponden a la base, las articulaciones intermedias y la punta del dedo.


El flujo de captura de datos se divide en tres etapas continuas que ocurren en milisegundos:
- Detección de la Palma (Palm Detection): El algoritmo primero analiza la imagen completa de la cámara para encontrar una mano. Como las palmas y los puños son zonas relativamente estables y fáciles de identificar, el sistema recorta esa región exacta de la imagen y descarta el fondo (muebles, ropa, etc.).
- Predicción de Puntos (Hand Landmark Model): Sobre la región recortada de la mano, el modelo predice la ubicación exacta de los 21 puntos de referencia.
- Extracción de Coordenadas 3D: Por cada uno de los 21 puntos, MediaPipe genera tres valores:X e Y: La posición horizontal y vertical del punto dentro de la imagen (coordenadas de la pantalla). Z: La profundidad relativa. 

Con esto, logra estimar qué tan cerca o lejos está cada articulación respecto a la muñeca (basándose en el tamaño de la mano). Al transformar el video en un flujo continuo de coordenadas numéricas, los algoritmos de IA ya no ven colores ni luces; solo ven un "esqueleto" que se mueve en el tiempo, facilitando el entrenamiento de modelos para identificar las letras del fingerspelling.

### Por qué hay landmarks faltantes

MediaPipe no devuelve una coordenada por defecto: si el detector no supera su umbral mínimo de confianza para una mano en un fotograma determinado, simplemente no reporta ningún punto y esa fila queda vacía. Las causas más comunes en este tipo de grabaciones son que la mano salga del encuadre (los videos se capturan con la cámara frontal de un teléfono, y el participante no siempre mantiene la mano dentro del cuadro), la oclusión (una mano tapa a la otra o al rostro), el desenfoque por movimiento rápido —recordando que el deletreo va a 5 o 6 letras por segundo— y las condiciones de iluminación.

Esto es importante para el análisis porque los valores faltantes no son un error de captura ni datos corruptos: son ausencia de detección. En la muestra usada en este proyecto, el rostro aparece casi siempre (alrededor del 0.7% de valores faltantes) mientras que la mano derecha falta en más de la mitad de los fotogramas (cerca del 54.5%), o sea que el faltante se concentra justo en la variable más informativa para el deletreo. Por eso no se rellenan con cero, ya que eso ubicaría la mano en el origen de la imagen e introduciría un movimiento que nunca ocurrió; el tratamiento aplicado se documenta en el notebook `02_limpieza_preprocesamiento.ipynb`.

### Por qué se usan landmarks y no los fotogramas de video

La competencia entrega coordenadas y no el video original por cuatro razones:

- **Privacidad:** los landmarks son coordenadas numéricas anónimas. Aunque incluyen puntos del rostro, no permiten reconstruir la imagen ni identificar a las personas que participaron en la grabación.
- **Tamaño:** aun siendo solo coordenadas, el conjunto completo pesa 158 GB. El video equivalente sería de un orden de magnitud mucho mayor, inviable de distribuir en una competencia abierta.
- **Costo de cómputo:** al partir de un vector numérico por fotograma, los participantes se saltan por completo la etapa de visión por computadora y pueden concentrarse en el modelado de la secuencia.
- **Invarianza:** al descartar píxeles se eliminan el fondo, la ropa, la iluminación y el tono de piel, lo que reduce el riesgo de que el modelo aprenda condiciones de grabación en lugar de la forma de la mano.

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

4. CTC (Connectionist Temporal Classification)

Las arquitecturas anteriores producen una salida por fotograma, pero la etiqueta que se tiene es la frase completa: `train.csv` dice que una secuencia corresponde a "3 creekhouse", sin indicar en qué fotograma empieza y termina cada letra. Etiquetar eso a mano sería carísimo, y además el límite entre una letra y la siguiente es difuso porque los dedos ya se están acomodando para la letra que viene.

CTC (Graves et al., 2006) resuelve exactamente ese problema. Agrega un símbolo especial en blanco al alfabeto y, en lugar de exigir una alineación fija, calcula la probabilidad de la frase objetivo sumando todos los alineamientos posibles que colapsan al mismo texto: repeticiones consecutivas de una letra se fusionan y los blancos se descartan. Así, una secuencia de 200 fotogramas puede producir una frase de 12 caracteres sin que nadie haya tenido que marcar dónde termina cada seña. Por eso es la función de pérdida estándar en reconocimiento de voz y de deletreo manual, y es la más usada en las soluciones de esta competencia.

### Normalización y aumentación de los landmarks

Antes de entrenar, las coordenadas se suelen normalizar. Como MediaPipe entrega la posición dentro de la imagen, dos personas que hacen exactamente la misma letra dan números distintos según dónde estén sentadas y qué tan lejos de la cámara. Lo habitual es reexpresar los puntos respecto a un origen anatómico (la muñeca para la mano, o los hombros para el cuerpo) y escalarlos por una distancia de referencia, de modo que el modelo vea la forma de la mano y no su ubicación en el cuadro.

La aumentación de datos busca lo mismo desde otro ángulo: generar variantes plausibles de cada secuencia para que el modelo no se sobreajuste a la forma de grabar de unos pocos participantes. Las transformaciones típicas sobre landmarks son rotaciones, escalados y traslaciones pequeñas, el reflejo especular horizontal (que convierte una seña hecha con la derecha en su equivalente con la izquierda, útil porque en el conjunto hay participantes zurdos y diestros), cambios de velocidad mediante interpolación temporal, y el descarte aleatorio de fotogramas o de puntos, que además imita las no detecciones que ya ocurren de forma natural en los datos.


## 4. Referencias

Al-Qaderi, M., & El-Sabaa, H. (2026). American Sign Language recognition for alphabets using MediaPipe and LSTM [Reconocimiento de alfabetos en la Lengua de Señas Americana utilizando MediaPipe y LSTM]. ResearchGate. https://www.researchgate.net/publication/366722112_American_Sign_Language_Recognition_for_Alphabets_Using_MediaPipe_and_LSTM


Pitsikalis, V., Katsamanis, A., & Maragos, P. (2024). Tracking and recognition of fingerspelling from videos [Seguimiento y reconocimiento de deletreo manual a partir de videos]. University of Thessaly Institutional Repository. https://ir.lib.uth.gr/xmlui/bitstream/handle/11615/59441/25386.pdf
