# 1. Investigación del tema

## 1.0 Qué es el deletreo manual (fingerspelling) en ASL
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

## 1.1 Cómo se capturan los datos: MediaPipe y landmarks
MediaPipe es un marco de trabajo de código abierto de Google que permite aplicar modelos de Inteligencia Artificial para el procesamiento de video en tiempo real.Su principal ventaja es la eficiencia: es tan ligero que puede ejecutarse directamente en dispositivos móviles, páginas web o computadoras sin necesidad de tarjetas gráficas (GPU) de alta potencia.


Gomaa y El-Khoribi (2026) explican que para entender el movimiento de la mano, MediaPipe utiliza un modelo de DL que detecta una estructura geométrica llamada Hand Landmarks (puntos de referencia).El sistema localiza exactamente 21 puntos tridimensionales (X, Y, Z) en cada mano, distribuidos estratégicamente en las articulaciones clave:Punto 0: La muñeca (origen de la mano). 4 puntos por dedo: Cada uno de los 5 dedos (pulgar, índice, medio, anular y meñique) tiene asignados 4 puntos que corresponden a la base, las articulaciones intermedias y la punta del dedo.


El flujo de captura de datos se divide en tres etapas continuas que ocurren en milisegundos:

- Detección de la Palma (Palm Detection): El algoritmo primero analiza la imagen completa de la cámara para encontrar una mano. Como las palmas y los puños son zonas relativamente estables y fáciles de identificar, el sistema recorta esa región exacta de la imagen y descarta el fondo (muebles, ropa, etc.).
- Predicción de Puntos (Hand Landmark Model): Sobre la región recortada de la mano, el modelo predice la ubicación exacta de los 21 puntos de referencia.
- Extracción de Coordenadas 3D: Por cada uno de los 21 puntos, MediaPipe genera tres valores:X e Y: La posición horizontal y vertical del punto dentro de la imagen (coordenadas de la pantalla). Z: La profundidad relativa. 

Con esto, logra estimar qué tan cerca o lejos está cada articulación respecto a la muñeca (basándose en el tamaño de la mano). Al transformar el video en un flujo continuo de coordenadas numéricas, los algoritmos de IA ya no ven colores ni luces; solo ven un "esqueleto" que se mueve en el tiempo, facilitando el entrenamiento de modelos para identificar las letras del fingerspelling.

### Por qué hay landmarks faltantes

MediaPipe no devuelve una coordenada por defecto: si el detector no supera su umbral mínimo de confianza para una mano en un fotograma determinado, simplemente no reporta ningún punto y esa fila queda vacía. Las causas más comunes en este tipo de grabaciones son que la mano salga del encuadre (los videos se capturan con la cámara frontal de un teléfono, y el participante no siempre mantiene la mano dentro del cuadro), la oclusión (una mano tapa a la otra o al rostro), el desenfoque por movimiento rápido, recordando que el deletreo va a 5 o 6 letras por segundo, y las condiciones de iluminación.

Esto es importante para el análisis porque los valores faltantes no son un error de captura ni datos corruptos: son ausencia de detección. En la muestra usada en este proyecto, el rostro aparece casi siempre (alrededor del 0.7% de valores faltantes) mientras que la mano derecha falta en más de la mitad de los fotogramas (cerca del 54.5%), o sea que el faltante se concentra justo en la variable más informativa para el deletreo. Por eso no se rellenan con cero, ya que eso ubicaría la mano en el origen de la imagen e introduciría un movimiento que nunca ocurrió; el tratamiento aplicado se documenta en el notebook 02_limpieza_preprocesamiento.ipynb.

### Por qué se usan landmarks y no los fotogramas de video

La competencia entrega coordenadas y no el video original por cuatro razones:

- **Privacidad:** los landmarks son coordenadas numéricas anónimas. Aunque incluyen puntos del rostro, no permiten reconstruir la imagen ni identificar a las personas que participaron en la grabación.
- **Tamaño:** aun siendo solo coordenadas, el conjunto completo pesa 158 GB. El video equivalente sería de un orden de magnitud mucho mayor, inviable de distribuir en una competencia abierta.
- **Costo de cómputo:** al partir de un vector numérico por fotograma, los participantes se saltan por completo la etapa de visión por computadora y pueden concentrarse en el modelado de la secuencia.
- **Invarianza:** al descartar píxeles se eliminan el fondo, la ropa, la iluminación y el tono de piel, lo que reduce el riesgo de que el modelo aprenda condiciones de grabación en lugar de la forma de la mano.

## 1.2 Técnicas para reconocer patrones en secuencias de landmarks
Con el esqueleto se pasa al procesamiento de secuencias de forma continua. Para ello la IA utiliza estas estrategias:
**1. RNN y LSTM (Redes Neuronales Recurrentes)**

RNN son redes que t ienen una "memoria" interna que les permite recordar lo que pasó en fotogramas anteriores. LSTM (Long Short-Term Memory) es la variante más utilizada. Resuelve el problema de las RNN básicas, que olvidan rápidamente la información del pasado. Una LSTM mantiene una "línea de vida" que preserva la información relevante a largo plazo. Luego, la red recibe las coordenadas de la mano fotograma por fotograma. Al procesar la letra actual, la LSTM "recuerda" la posición de la mano un segundo antes. Esto es crucial para detectar la coarticulación (saber si un dedo se está doblando debido a la letra anterior).

**2. CNN 1D (Redes Neuronales Convolucionales Unidimensionales)**

Las CNN 1D se aplican sobre datos que se desplazan en una sola dirección temporal.En lugar de buscar patrones visuales en una foto (como ojos o narices), la CNN 1D desliza "filtros" a lo largo de la línea de tiempo de las coordenadas de los landmarks.
Con ello ya captura micro-movimientos locales muy rápidos. Por ejemplo, identifica la velocidad exacta con la que baja un dedo para formar la letra J. Al ser operaciones matemáticas simples, las CNN 1D son extremadamente rápidas y eficientes para ejecutarse en tiempo real.

**3. Transformers (Modelos basados en Atención)**


Es la arquitectura de CHATGPT. Los Transformers han reemplazado en gran medida a las LSTM porque no procesan los datos paso a paso, sino que analizan toda la secuencia al mismo tiempo.

Esto permite al modelo calcular matemáticamente qué partes de la secuencia se relacionan más entre sí, sin importar qué tan separadas estén en el tiempo.

El Transformer puede analizar una palabra entera deletreada de corrido. Entiende el contexto global del movimiento y puede corregir errores basándose en las letras vecinas. Si el sistema detecta con un 90% de certeza las letras H-O-U-S- y la última letra está muy distorsionada entre una E y una O, el mecanismo de atención sabrá que contextualmente la palabra más probable en inglés es HOUSE, corrigiendo la salida de texto automáticamente (Al-Qaderi & El-Sabaa, 2026).

**4. CTC (Connectionist Temporal Classification)**


Las arquitecturas anteriores producen una salida por fotograma, pero la etiqueta que se tiene es la frase completa: train.csv dice que una secuencia corresponde a "3 creekhouse", sin indicar en qué fotograma empieza y termina cada letra. Etiquetar eso a mano sería carísimo, y además el límite entre una letra y la siguiente es difuso porque los dedos ya se están acomodando para la letra que viene.

CTC (Graves et al., 2006) resuelve exactamente ese problema. Agrega un símbolo especial en blanco al alfabeto y, en lugar de exigir una alineación fija, calcula la probabilidad de la frase objetivo sumando todos los alineamientos posibles que colapsan al mismo texto: repeticiones consecutivas de una letra se fusionan y los blancos se descartan. Así, una secuencia de 200 fotogramas puede producir una frase de 12 caracteres sin que nadie haya tenido que marcar dónde termina cada seña. Por eso es la función de pérdida estándar en reconocimiento de voz y de deletreo manual, y es la más usada en las soluciones de esta competencia.

### Normalización y aumentación de los landmarks

Antes de entrenar, las coordenadas se suelen normalizar. Como MediaPipe entrega la posición dentro de la imagen, dos personas que hacen exactamente la misma letra dan números distintos según dónde estén sentadas y qué tan lejos de la cámara. Lo habitual es reexpresar los puntos respecto a un origen anatómico (la muñeca para la mano, o los hombros para el cuerpo) y escalarlos por una distancia de referencia, de modo que el modelo vea la forma de la mano y no su ubicación en el cuadro.

La aumentación de datos busca lo mismo desde otro ángulo: generar variantes plausibles de cada secuencia para que el modelo no se sobreajuste a la forma de grabar de unos pocos participantes. Las transformaciones típicas sobre landmarks son rotaciones, escalados y traslaciones pequeñas, el reflejo especular horizontal (que convierte una seña hecha con la derecha en su equivalente con la izquierda, útil porque en el conjunto hay participantes zurdos y diestros), cambios de velocidad mediante interpolación temporal, y el descarte aleatorio de fotogramas o de puntos, que además imita las no detecciones que ya ocurren de forma natural en los datos.

# 2. Situación problemática
El deletreo manual (fingerspelling) es un componente importante del lenguaje de señas estadounidense (ASL), ya que permite representar letra por letra palabras que no cuentan con una seña propia, como nombres, direcciones, marcas, siglas y términos técnicos. Se estima que entre el 12 % y el 35 % del discurso en ASL corresponde a deletreo manual (Padden y Gunsauls, 2003), lo que evidencia su relevancia en la comunicación cotidiana.

Su reconocimiento automático representa un reto distinto al de la clasificación convencional de imágenes, debido a que el significado no depende únicamente de la forma de la mano, sino también de su movimiento a lo largo del tiempo. La coarticulación entre letras, la velocidad de ejecución, las diferencias entre participantes y factores como la iluminación, el fondo, el encuadre y la oclusión pueden generar variaciones importantes en los datos.

En 2023, Google, en colaboración con la Deaf Professional Arts Network, publicó en Kaggle la competencia American Sign Language Fingerspelling Recognition, cuyo conjunto de datos contiene más de tres millones de caracteres realizados por más de cien firmantes sordos. Por razones de privacidad y almacenamiento, los datos no se distribuyen como video, sino como secuencias de landmarks obtenidos mediante MediaPipe, con coordenadas de puntos de las manos, el rostro y el cuerpo para cada cuadro.

Esta representación convierte el conjunto en datos de series temporales y, al mismo tiempo, introduce posibles valores faltantes y variaciones en la calidad de los puntos detectados. Por ello, antes de definir estrategias de procesamiento o modelado, es necesario caracterizar aspectos como la duración de las secuencias, la distribución de los valores faltantes, las regiones corporales involucradas, los patrones de movimiento y las diferencias entre participantes. Sin esta exploración, decisiones como qué variables conservar, cómo normalizar las coordenadas o qué secuencias utilizar podrían basarse en supuestos no comprobados y afectar la reproducibilidad del análisis.

# 3. Problema científico
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
4. ¿Existen diferencias sistemáticas entre participantes, en duración, en
   mano dominante o en disponibilidad de landmarks, lo bastante grandes como
   para exigir una normalización previa al modelado?
5. ¿Qué observaciones deben excluirse o corregirse antes de modelar, y bajo
   qué criterio explícito?

Delimitación: El estudio se limita al análisis exploratorio de los datos. No incluye el entrenamiento ni la evaluación de modelos de transcripción.

# 4. Objetivos
## 4.1 Objetivo general
Caracterizar, mediante un análisis exploratorio de datos, la estructura, la
calidad y la variabilidad de las secuencias de landmarks del conjunto
*American Sign Language Fingerspelling Recognition* (Google, 2023), con el fin
de determinar y justificar con evidencia empírica las transformaciones de
limpieza y preprocesamiento que el conjunto requiere antes de la construcción
de un modelo de transcripción automática del deletreo manual.

## 4.2 Objetivos específicos
**1.** **Inventariar la estructura del conjunto de datos**, reportando el número
   de observaciones, el número y el tipo de cada variable, y los conteos de
   participantes, secuencias y cuadros de la muestra analizada.
   *Criterio de cumplimiento:* una tabla de metadatos con nombre, tipo, rango
   y unidad de cada variable, y los conteos anteriores documentados.

**2.** **Cuantificar la completitud de los datos**, calculando el porcentaje de
   valores faltantes por región anatómica (mano izquierda, mano derecha,
   rostro y pose) y su distribución a lo largo de la secuencia.
   *Criterio de cumplimiento:* porcentajes de faltantes por región y por
   posición relativa en la secuencia, y un umbral explícito a partir del cual
   una secuencia se considera inutilizable.

**3.** **Describir estadísticamente las variables de interés**, duración de la
   secuencia, longitud de la frase objetivo y número de secuencias por
   participante, mediante medidas de tendencia central y dispersión (media,
   mediana, desviación estándar, cuartiles y rango intercuartílico) y tablas
   de frecuencia para las variables categóricas, identificando valores
   atípicos bajo un criterio declarado (regla de 1.5 · RIC).
   *Criterio de cumplimiento:* tabla de resumen numérico, tablas de frecuencia
   y listado de atípicos clasificados como error de captura o variación
   legítima.

**4.** **Evaluar la relación entre la duración de la secuencia, la cantidad de
   cuadros con mano detectada y la longitud de la frase objetivo**, mediante
   coeficientes de correlación y análisis bivariado, para establecer si existe
   una razón cuadros-por-carácter estable y si se sostiene entre participantes.
   *Criterio de cumplimiento:* matriz de correlaciones y una conclusión
   explícita sobre la existencia o no de dicha razón.

**5.** **Elaborar gráficos exploratorios** (histogramas, diagramas de caja y
   bigotes, diagramas de dispersión y visualización de la trayectoria de la
   mano en el tiempo) que evidencien la forma de las distribuciones, los
   valores atípicos y los patrones de datos faltantes.
   *Criterio de cumplimiento:* al menos un gráfico por cada uno de los
   objetivos 2, 3 y 4, exportado a reports/figures/ e interpretado en el
   texto.

## 4.3 Trazabilidad de los objetivos
| Objetivo | Actividad de la guía | Archivo del repositorio |
|---|---|---|
| 1 | 4a, variables, observaciones y tipos | notebooks/01_estructura_datos.ipynb |
| 2 | 3, limpieza y preprocesamiento | notebooks/02_limpieza_preprocesamiento.ipynb |
| 3 | 4b, resumen numérico y frecuencias | notebooks/03_analisis_univariado.ipynb |
| 4 | 4c, cruce de variables | notebooks/04_analisis_bivariado.ipynb |
| 5 | 4d, gráficos exploratorios | notebooks/05_visualizaciones.ipynb |




# 5. Descripción de los datos

## 5.1 Variables y observaciones

El archivo train.csv trae 67,208 filas, una por secuencia de deletreo, y 5 columnas:
path y phrase son texto (la ruta al archivo de landmarks y la frase deletreada), y
file_id, sequence_id y participant_id son enteros que identifican el archivo, la
secuencia y la persona que deletreó. Ninguna de estas tres es una variable numérica
en el sentido de que se pueda promediar, son identificadores.

Participan 94 personas, pero de forma muy despareja: la que menos aporta tiene 1 sola
secuencia y la que más, 1,535 (mediana de 794). Los 10 participantes con más datos
concentran apenas el 14.5% del total, así que no hay uno o dos participantes que
dominen la muestra, pero sí hay una cola larga de participantes con muy pocas
secuencias, que quedan casi sin representación.

Aparte de train.csv, los datos de landmarks vienen en archivos parquet (uno por
grupo de secuencias). Cada archivo trae aproximadamente 1,000 secuencias,
identificadas por sequence_id, que en este caso es el índice de la tabla y no una
columna. Por cada cuadro (frame) hay 1,630 columnas: 468 corresponden al rostro, 33 a
la pose del cuerpo y 21 a cada mano, cada punto con sus tres coordenadas x, y, z.



<div class="figura"><img src="C:/Users/leome/OneDrive/Desktop/Universidad/CC3084-Proyecto2/reports/figures/01_landmarks_un_cuadro.png"><p class="leyenda">Figura 1. Landmarks de un cuadro real de una secuencia (rostro, pose y mano derecha). El rostro forma un bloque compacto arriba, pose se dispersa por el resto del cuerpo, y la mano queda agrupada cerca de la cara, algo esperable en fingerspelling.</p></div>



## 5.2 Limpieza y preprocesamiento

Como se explicó en la sección de investigación del tema, los landmarks faltantes no
son un error de captura, son ausencia de detección. En la muestra de este proyecto
el rostro está presente casi siempre (falta en un 0.7% de los cuadros) y la pose
prácticamente nunca falta, mientras que las manos son harina de otro costal: la mano
izquierda falta en casi todos los cuadros y la mano derecha falta en un 54.5% de
ellos en promedio, es decir, el problema de calidad se concentra justo en la parte
del cuerpo que más información aporta para el deletreo.



<div class="figura"><img src="C:/Users/leome/OneDrive/Desktop/Universidad/CC3084-Proyecto2/reports/figures/02_missing_landmarks_boxplot.png"><p class="leyenda">Figura 2. Proporción de valores faltantes por tipo de landmark en la muestra fija. El rostro y la pose casi no faltan; las manos, sobre todo la izquierda, sí.</p></div>



Con esto en mente se tomaron tres decisiones de limpieza. Primero, los valores
faltantes se rellenaron con el último valor válido dentro de la misma secuencia
(forward-fill) en lugar de interpolar o poner cero, porque poner cero ubicaría la
mano en el origen de la imagen e inventaría un movimiento que nunca ocurrió.
Segundo, se eliminaron las 468 columnas del rostro (86% de las 1,630 columnas
originales) porque aportan poco a la tarea de deletreo y encarecen mucho el
procesamiento; con esto la muestra quedó en 226 columnas. Tercero, se descartaron
las secuencias que tenían alguna mano detectada en menos del 10% de sus cuadros, por
ser casos donde prácticamente no hay señal útil. La muestra resultante se guardó en
data/processed/ para que el resto del análisis parta de datos ya tratados.


# 6. Análisis exploratorio

## 6.1 Variables cuantitativas

train.csv no trae ninguna variable numérica propia, así que el análisis cuantitativo
se hizo sobre tres variables derivadas: la longitud de la frase objetivo (en
caracteres), la longitud de la secuencia (en cuadros) y el porcentaje de cuadros sin
ninguna mano detectada.

La longitud de frase va de 1 a 31 caracteres, con media de 17.8 y mediana de 17: es
una distribución casi simétrica y acotada por arriba, con una moda marcada en 12
caracteres que se nota claramente en el histograma.



<div class="figura"><img src="C:/Users/leome/OneDrive/Desktop/Universidad/CC3084-Proyecto2/reports/figures/hist_phrase_len.png"><p class="leyenda">Figura 3. Distribución de la longitud de frase (en caracteres) sobre las 67,208 secuencias de train.csv.</p></div>



La longitud de secuencia se comporta distinto: la media es de 160.8 cuadros, la
mediana de 147, y hay una cola larga hacia la derecha que llega hasta 751 cuadros
(asimetría de 1.09). Frases de largo parecido pueden tomar duraciones muy distintas
según la persona y el momento.



<div class="figura"><img src="C:/Users/leome/OneDrive/Desktop/Universidad/CC3084-Proyecto2/reports/figures/hist_seq_len.png"><p class="leyenda">Figura 4. Distribución de la longitud de secuencia (en cuadros), con sesgo marcado hacia la derecha.</p></div>



El porcentaje de cuadros sin mano detectada tiene una media de 44.8% y una mediana de
42.9%, con bastante dispersión entre secuencias (rango intercuartílico de 51.6
puntos porcentuales). Esta variable resume, secuencia por secuencia, el problema de
calidad que se trató en la limpieza: entre más alto este porcentaje, menos señal útil
trae la secuencia aunque su longitud bruta sea grande.

## 6.2 Relación entre variables y correlaciones

Cruzando longitud de frase con longitud de secuencia aparece una correlación de
Pearson de 0.60, positiva y moderada: como es de esperarse, frases más largas
requieren secuencias más largas, aunque la relación no es estricta porque la
velocidad de deletreo varía entre personas.



<div class="figura"><img src="C:/Users/leome/OneDrive/Desktop/Universidad/CC3084-Proyecto2/reports/figures/scatter_seq_vs_phrase.png"><p class="leyenda">Figura 5. Relación entre longitud de frase y longitud de secuencia (r de Pearson igual a 0.60), con línea de tendencia.</p></div>



También se cruzó la longitud de secuencia con el porcentaje promedio de landmarks
faltantes, y salió una correlación negativa pero débil (r de -0.19): las secuencias
más largas tienden a tener, en promedio, un poco menos de landmarks faltantes. En
cambio, la correlación entre longitud de frase y landmarks faltantes resultó
prácticamente nula (r de -0.01), así que una variable no explica a la otra.



<div class="figura"><img src="C:/Users/leome/OneDrive/Desktop/Universidad/CC3084-Proyecto2/reports/figures/heatmap_correlacion.png"><p class="leyenda">Figura 6. Matriz de correlación entre longitud de frase, longitud de secuencia y proporción promedio de landmarks faltantes.</p></div>



También se revisó cómo varía la longitud de secuencia entre participantes: las
medianas por persona van aproximadamente de 75 a 241 cuadros, es decir, hay
diferencias reales en qué tan rápido o lento deletrea cada quien.



<div class="figura"><img src="C:/Users/leome/OneDrive/Desktop/Universidad/CC3084-Proyecto2/reports/figures/box_seq_len_by_participant.png"><p class="leyenda">Figura 7. Longitud de secuencia por participante (15 participantes con más datos), comparada contra la mediana global de 147 cuadros.</p></div>



Para detectar valores atípicos se usó el criterio de 1.5 veces el rango
intercuartílico (IQR) sobre longitud de secuencia y longitud de frase. Con eso se
encontraron 28 secuencias atípicas de un total de 998 en la muestra (cerca del 3%),
con límites normales entre 0 y 359 cuadros y entre 0 y 37 caracteres. Los límites
inferiores que da la fórmula son negativos, pero como ninguna de las dos variables
puede ser negativa, el límite inferior real se interpreta como cero: es decir, los
casos atípicos encontrados son todos de secuencias o frases inusualmente largas, no
cortas.

## 6.3 Variables categóricas

Las dos variables categóricas del proyecto son el participante que deletrea y los
caracteres que forman las frases. Ya se describió el desbalance entre participantes
en la sección de descripción de los datos.

Sobre los caracteres, en las frases aparecen 59 símbolos distintos. Agrupándolos por
tipo, las letras representan un 61.2% del total de caracteres, los dígitos un 24.6%,
los espacios un 4.9% y otros símbolos (guiones, barras, signos de puntuación) un
9.3%. El peso de los dígitos tiene sentido porque buena parte de las frases son
direcciones, teléfonos y URLs, no texto corrido. Entre las letras individuales, e, a,
o, r, n y t concentran la mayoría de las apariciones, mientras que j, q, z y varios
símbolos casi no tienen muestras.



<div class="figura"><img src="C:/Users/leome/OneDrive/Desktop/Universidad/CC3084-Proyecto2/reports/figures/bar_char_frequency.png"><p class="leyenda">Figura 8. Frecuencia absoluta de cada carácter presente en las frases objetivo, de mayor a menor.</p></div>



Por último, para ilustrar cómo se ve el movimiento real de una mano durante el
deletreo, se graficó la trayectoria de los landmarks de la mano a lo largo de todos
los cuadros de una secuencia de ejemplo. Se nota una nube de puntos con varias zonas
donde el trazo se concentra, compatible con las pausas y cambios de forma que ocurren
al pasar de una letra a otra.



<div class="figura"><img src="C:/Users/leome/OneDrive/Desktop/Universidad/CC3084-Proyecto2/reports/figures/sample_hand_trajectory.png"><p class="leyenda">Figura 9. Trayectoria en 2D de los landmarks de la mano a lo largo de una secuencia completa de ejemplo.</p></div>




# 7. Hallazgos y conclusiones

## 7.1 Síntesis de la investigación técnica
El deletreo manual en ASL implica el uso de 5 a 6 letras por segundo, similar a la rapidez encontrada en el idioma español (verbal), que según Deutsche Welle (2025) utiliza 7.82 sílabas por segundo, es decir, un poco más de 12 letras por segundo. El deletreo del ASL se utiliza para nombres propios, siglas o términos técnicos.

El presente proyecto emplea MediaPipe para extraer 21 puntos tridimensionales (*landmarks*) por mano en lugar de procesar video crudo. Esto protege la privacidad, reduce significativamente el tamaño de los datos y elimina interferencias visuales como la iluminación o el fondo.

Por otro lado, se debe tomar en cuenta que la pérdida de *landmarks* (valores nulos) no es un error imprevisto, sino que puede deberse a la ausencia de detección por movimiento rápido, oclusión o porque la mano sale del encuadre. En los datos de muestra, la mano derecha está ausente en más del 54.5 % de los fotogramas.

Para traducir estos *landmarks* a texto, las arquitecturas de aprendizaje automático ideales incluyen LSTM, CNN 1D, Transformers y CTC (*Connectionist Temporal Classification*), siendo esta última fundamental para secuencias continuas sin un límite difuso entre letras.

## 7.2 Resumen de hallazgos por notebook
**01_estructura_datos:** train.csv trae 67,208 secuencias de 94 participantes, pero están muy desbalanceadas entre sí: van de 1 a 1,535 secuencias por persona (mediana de 794), así que cualquier comparación por participante debe tomarse con cuidado. Las frases van de 1 a 31 caracteres (mediana de 17). Cada Parquet de *landmarks* trae aproximadamente 1,000 secuencias, indexadas por sequence_id (no es columna), con 1,630 columnas por cuadro: 468 de rostro, 33 de pose y 21 por mano. Al graficar un cuadro se observa que la mano queda agrupada cerca de la cara, algo esperable en *fingerspelling*.

**02_limpieza_preprocesamiento:** los datos ya contienen coordenadas normalizadas por MediaPipe, por lo que el principal problema no es la suciedad, sino las detecciones faltantes. Estas afectan especialmente a la mano derecha, con aproximadamente 54.5 % de valores NaN, frente a aproximadamente 0.7 % en el rostro. Los faltantes se trataron mediante *forward-fill* dentro de cada secuencia para evitar que la interpolación generara posiciones artificiales entre letras. Además, se redujeron las variables de 1,630 a 226 columnas eliminando los 468 puntos del rostro, que representan aproximadamente el 86 % de las columnas y aportan poca información al deletreo. Finalmente, se excluyeron las secuencias con detección de mano en menos del 10 % de sus cuadros y la muestra resultante se almacenó en data/processed/.

**03_analisis_univariado:** train.csv no trae variables numéricas propias (solo identificadores), así que el análisis se apoya en tres derivadas: longitud de frase, longitud de secuencia y porcentaje de frames sin mano detectada. La frase está acotada y es casi simétrica (1 a 31 caracteres, media 17.8), mientras que la secuencia es asimétrica a la derecha (mediana aprox. 147 frames, cola por encima de 700): dos personas deletreando la misma cantidad de caracteres pueden generar secuencias de duración muy distinta. En las categóricas se repite el mismo patrón de desbalance ya visto en 01: participantes entre 1 y 1,535 secuencias, y entre caracteres, unas pocas letras (e, a, o, r, n, t) concentran la mayoría de las apariciones mientras que j, q, z y varios símbolos casi no tienen muestras.

**04_analisis_bivariado:** se comprobó una correlación positiva, r de aproximadamente 0.5979, entre la longitud de la frase y el número de frames utilizados. Frases con mayor cantidad de caracteres generan secuencias más largas. Asimismo, se observó una relación estadísticamente negativa de r de aproximadamente -0.1883 entre la proporción de *landmarks* faltantes y la longitud de la secuencia. Esto indica que, si la cámara pierde los puntos de referencia, la secuencia registrada tiende a ser más corta.

La matriz de correlación (*heatmap*) reveló que la relación cruzada entre la longitud de la frase y el promedio de información perdida (NaN) es lo suficientemente baja como para ser descartada del enfoque principal.

Mediante el método del rango intercuartílico (IQR), se detectaron 28 secuencias atípicas sobre un total de 998, aproximadamente el 3 % de datos atípicos. Los límites estadísticos normales se establecieron entre 0 y 359.4 frames, y entre 0 y 37 caracteres, interpretando los límites inferiores negativos como un tope natural de cero.

**05_visualizaciones**: las visualizaciones confirmaron que la duración de las secuencias presenta un sesgo hacia la derecha: la mayoría contiene entre 50 y 250 frames, con una mediana global de 147 y algunos valores extremos cercanos a 750. 

También hubo diferencias importantes entre participantes, cuyas medianas varían aproximadamente entre 75.5 y 241 frames. 

La longitud de las frases se concentra entre 10 y 30 caracteres, con una moda marcada de 12, y mantiene una correlación positiva moderada con la cantidad de frames (r de aproximadamente 0.602). 

Además, la frecuencia de caracteres está desbalanceada: predominan letras como e y a, mientras que j, z y varios símbolos cuentan con pocas observaciones. 

En conjunto, estos resultados indican que un futuro modelo deberá aceptar secuencias de longitud variable, considerar las diferencias de velocidad entre participantes y prestar especial atención a los caracteres menos frecuentes.


## 7.3 Problemas de calidad de datos encontrados

- **Landmarks faltantes concentrados en las manos:** el rostro casi no falta (aprox. 0.7 %), pero la mano derecha falta en aprox. 54.5 % de los frames. No es ruido ni corrupción, es ausencia de detección (mano fuera de cuadro, oclusión, movimiento rápido) y afecta justo a la región más informativa para el deletreo.
- **Desbalance entre participantes:** de 1 a 1,535 secuencias por persona (mediana 794); los 10 participantes con más datos solo concentran el 14.5 % del total, pero los que tienen pocas secuencias quedan casi sin representación.
- **Desbalance entre caracteres:** el alfabeto efectivo tiene 59 símbolos, pero unas pocas letras (e, a, o, r, n, t) concentran la mayoría de las apariciones; j, q, z y varios símbolos aparecen en una fracción mínima.
- **Longitud de secuencia muy variable y sesgada a la derecha:** mediana aprox. 147 frames pero con cola hasta 750+, mientras que la frase objetivo es corta y casi simétrica (1-31 caracteres). El mismo contenido puede tomar duraciones muy distintas.
- **Outliers:** con el criterio de 1.5·IQR se detectaron 28 secuencias atípicas de 998 (aprox. 3 %) en la muestra, con límites normales de 0-359 frames y 0-37 caracteres.
- **Relación cruzada débil:** la correlación entre longitud de frase y proporción de landmarks faltantes resultó baja, así que no se puede usar una variable para explicar la otra.

## 7.4 Conclusiones sobre los siguientes pasos

- **Tratar los landmarks faltantes sin inventar movimiento:** se usó forward-fill dentro de cada secuencia (no interpolación) para no generar posiciones artificiales entre letras, y se descartaron secuencias con mano detectada en menos del 10 % de sus frames.
- **Reducir columnas irrelevantes:** los 468 puntos de rostro (86 % de las columnas) aportan poco al deletreo; ya se redujo la muestra de 1,630 a 226 columnas en 02_limpieza_preprocesamiento.ipynb.
- **Separar entrenamiento/validación por participante**, no de forma aleatoria, para no medir si un futuro modelo memorizó el estilo de deletreo de unos pocos participantes muy representados.
- **Aceptar longitud de secuencia variable** (recorte o relleno) en vez de asumir una duración fija, dado el sesgo y la cola larga encontrados en 01, 03 y 05.
- **Reportar desempeño por carácter y no solo en promedio**, ya que el desbalance de caracteres haría que el promedio general esconda un mal desempeño en letras poco frecuentes (j, q, z, símbolos).
- Todo esto es evidencia para la limpieza/preprocesamiento; el entrenamiento y evaluación de un modelo de transcripción quedan fuera del alcance de este análisis exploratorio.

# 8. Referencias

- Google & Deaf Professional Arts Network. (2023). *Google, American Sign
  Language Fingerspelling Recognition* [Competencia de Kaggle].
  https:/<wbr/>/<wbr/>www.kaggle.com/<wbr/>competitions/<wbr/>asl-fingerspelling

- Graves, A., Fernández, S., Gomez, F. y Schmidhuber, J. (2006). Connectionist
  Temporal Classification: Labelling unsegmented sequence data with recurrent
  neural networks. *Proceedings of the 23rd International Conference on
  Machine Learning (ICML)*, 369, 376.

- Lugaresi, C., Tang, J., Nash, H., McClanahan, C., Uboweja, E., Hays, M.,
  et al. (2019). MediaPipe: A framework for building perception pipelines.
  *arXiv:1906.08172*.

- Padden, C. y Gunsauls, D. C. (2003). How the alphabet came to be used in a
  sign language. *Sign Language Studies*, 4(1), 10, 33.

- Shi, B., Del Rio, A. M., Keane, J., Michaux, J., Brentari, D., Shakhnarovich,
  G. y Livescu, K. (2018). American Sign Language fingerspelling recognition in
  the wild. *IEEE Spoken Language Technology Workshop (SLT)*, 145, 152.

- Zhang, F., Bazarevsky, V., Vakunov, A., Tkachenka, A., Sung, G., Chang, C.-L.
  y Grundmann, M. (2020). MediaPipe Hands: On-device real-time hand tracking.
  *arXiv:2006.10214*.

Al-Qaderi, M., & El-Sabaa, H. (2026). American Sign Language recognition for alphabets using MediaPipe and LSTM [Reconocimiento de alfabetos en la Lengua de Señas Americana utilizando MediaPipe y LSTM]. ResearchGate. https:/<wbr/>/<wbr/>www.researchgate.net/<wbr/>publication/<wbr/>366722112_American_Sign_Language_Recognition_for_Alphabets_Using_MediaPipe_and_LSTM


Google. (2023). Google, American Sign Language Fingerspelling Recognition [Conjunto de datos]. Kaggle. https:/<wbr/>/<wbr/>www.kaggle.com/<wbr/>competitions/<wbr/>asl-fingerspelling

Pitsikalis, V., Katsamanis, A., & Maragos, P. (2024). Tracking and recognition of fingerspelling from videos [Seguimiento y reconocimiento de deletreo manual a partir de videos]. University of Thessaly Institutional Repository. https:/<wbr/>/<wbr/>ir.lib.uth.gr/<wbr/>xmlui/<wbr/>bitstream/<wbr/>handle/<wbr/>11615/<wbr/>59441/<wbr/>25386.pdf