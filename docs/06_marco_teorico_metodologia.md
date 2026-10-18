# Marco teórico, metodología y referencias

**Responsable:** Cindy Mishelle Gualim Perez

> Secciones del informe final. El marco teórico adapta la revisión bibliográfica del Avance 2
> ([02_investigacion_tecnica.md](02_investigacion_tecnica.md)) y la lleva a lo que se usó en el modelado. La
> metodología describe los pasos reales del proyecto, con el código en `src/`. Las referencias de esta sección
> están en formato APA; las de [05_modelo_cnn_bigru.md](05_modelo_cnn_bigru.md) se unen al final del informe.

# Marco teórico

## Deletreo manual y la tarea de transcripción

El deletreo manual (*fingerspelling*) representa palabras letra por letra con el alfabeto manual del ASL, y se
usa sobre todo para nombres propios, términos técnicos y siglas. Entre el 12 % y el 35 % del discurso en ASL es
deletreo (Padden y Gunsauls, 2003). Reconocerlo es difícil porque se ejecuta muy rápido, las letras se
fusionan entre sí (coarticulación) y varias son casi iguales (M, N y T), de modo que la forma de la mano en un
solo cuadro no basta: hay que mirar la secuencia (Shi et al., 2018).

Formalmente, cada ejemplo es una secuencia de $T$ cuadros y su etiqueta es una frase de $L$ caracteres, con
$L$ mucho menor que $T$ y sin saber en qué cuadro empieza cada letra. Es un problema de *secuencia a
secuencia sin alineación*, que es el mismo planteamiento del reconocimiento de voz.

## MediaPipe y los *landmarks*

MediaPipe es un marco de Google para construir flujos de percepción sobre video, ligero y pensado para correr en
dispositivos móviles (Lugaresi et al., 2019). Su módulo de manos primero detecta la palma y luego predice 21
puntos tridimensionales por mano: la muñeca y cuatro puntos por dedo (Zhang et al., 2020). Junto con los puntos
de rostro y de pose (33 puntos del cuerpo), cada cuadro del video queda reducido a un vector de coordenadas
$(x, y, z)$. La competencia distribuye solo esos vectores, no el video (Google y Deaf Professional Arts
Network, 2023), lo que protege la privacidad de los firmantes, reduce el tamaño del conjunto y elimina fondo,
ropa e iluminación.

Dos propiedades de este formato condicionan el modelado:

- **Las coordenadas dependen de la cámara.** $x$ e $y$ son posiciones dentro de la imagen y $z$ es una
  profundidad relativa, así que la misma letra produce números distintos según dónde esté sentada la persona y
  qué tan cerca esté. Por eso se re-expresan respecto a un origen anatómico (muñeca para la mano, hombros para
  la pose) y se escalan por una distancia de referencia.
- **Hay ausencia de detección.** Cuando MediaPipe no supera su umbral de confianza no devuelve ningún punto, y
  la celda queda vacía. En la muestra del proyecto la mano derecha falta en 54.5 % de los cuadros. Es
  información sobre la captura, no un error de los datos, y por eso se trata sin inventar movimiento y se
  entrega al modelo como una máscara.

## Redes para secuencias de *landmarks*

**Redes recurrentes (RNN, LSTM y GRU).** Procesan la secuencia cuadro por cuadro y arrastran un estado interno
que resume lo anterior. Las RNN simples olvidan con rapidez porque el gradiente se desvanece en secuencias
largas; la LSTM lo corrige con celdas de memoria y compuertas (Hochreiter y Schmidhuber, 1997), y la GRU logra
algo parecido con menos parámetros (Cho et al., 2014; Chung et al., 2014). Una versión **bidireccional**
recorre la secuencia en ambos sentidos y concatena los resultados, de modo que cada letra se decide con el
contexto previo y posterior (Schuster y Paliwal, 1997), útil con la coarticulación.

**Convoluciones 1D.** Deslizan filtros sobre el eje del tiempo y detectan movimientos locales de pocos cuadros
(LeCun et al., 1998). Con *stride* además acortan la secuencia, y las redes convolucionales para secuencias
compiten con las recurrentes en varias tareas (Bai et al., 2018). Combinadas con una parte recurrente, como en
HandReader (Korotaev et al., 2025), dan una estructura de extractor local más contexto global.

**Transformers.** Reemplazan la recurrencia por atención: cada posición se compara con todas las demás en una
sola operación, por lo que captan dependencias lejanas y se entrenan en paralelo (Vaswani et al., 2017). El
costo crece con el cuadrado del largo de la secuencia, por lo que con secuencias de cientos de cuadros se
reduce el tiempo antes de la atención.

**CTC.** *Connectionist Temporal Classification* añade un símbolo vacío (*blank*) al alfabeto y calcula la
probabilidad de la frase objetivo sumando todos los alineamientos posibles que, al fusionar repeticiones y
quitar los blancos, dan esa frase (Graves et al., 2006). Así se entrena con la frase completa sin marcar dónde
cae cada letra. Exige que la salida tenga al menos tantos pasos como la frase más las repeticiones consecutivas
de letra, lo que limita cuánto se puede acortar la secuencia con el *stride*. La predicción se obtiene con
decodificación *greedy*: el símbolo más probable por paso, colapsando repeticiones y quitando blancos.

## Aumentación de datos

Con pocas secuencias y pocos participantes, un modelo tiende a memorizar el estilo de quien firma. La
aumentación genera variantes plausibles de cada ejemplo durante el entrenamiento para reducir ese
sobreajuste (Shorten y Khoshgoftaar, 2019). En *landmarks* las transformaciones naturales son rotaciones y
escalados leves, ruido en las coordenadas, cambios de velocidad por interpolación en el tiempo, y el descarte
aleatorio de cuadros o de puntos, que además imita las no detecciones que ya ocurren de forma natural.

## Métrica: tasa de error por carácter

La métrica es el **CER** (*character error rate*): la distancia de edición de Levenshtein entre la frase real
y la predicha (sustituciones + borrados + inserciones) dividida entre la longitud de la frase real. Un CER de 0
es transcripción perfecta; puede pasar de 1 si hay muchas inserciones. La métrica de la competencia de Kaggle
se calcula como $1 - \text{CER}$. Se reporta también el error por carácter, porque el desbalance de caracteres
hace que el promedio esconda el mal desempeño en las letras poco frecuentes.

---

# Metodología

## Datos

Se usó la muestra fija del EDA: `train.csv` con la frase y el participante de cada secuencia, más dos archivos
de `train_landmarks/` (de un total de 68, 158 GB). Se descargaron con la CLI de Kaggle
(`notebooks/00_setup_datos.ipynb`). Quedaron 1,878 secuencias de 92 participantes antes del split.

## Pasos

1. **Selección de variables.** Se conservan los 21 puntos de cada mano y los 33 de pose; se eliminan los 468
   puntos del rostro (86 % de las columnas), que aportan poco al deletreo. Cada punto tiene $x$, $y$, $z$.
2. **Normalización** (`src/features.py`). Cada mano se centra en su muñeca y se escala por la distancia
   muñeca–nudillo medio; la pose se centra en el punto medio de los hombros y se escala por el ancho de
   hombros. Se usa la mediana de la secuencia como escala para que un cuadro ruidoso no la distorsione.
3. **Valores faltantes.** Se calcula una máscara de detección por mano (2 columnas), se rellena con el último
   valor válido de la misma secuencia (*forward-fill*) y lo que queda al inicio va a 0. Se descartan las
   secuencias con alguna mano detectada en menos del 10 % de sus cuadros.
4. **Vector por cuadro.** 75 puntos × 3 coordenadas + 2 de máscara = 227 valores.
5. **Longitud.** La mediana es de 147 cuadros y la cola llega a ~750. Las secuencias de más de 384 cuadros se
   submuestrean tomando cuadros equiespaciados (`max_frames = 384`); el relleno por lotes se hace con
   `collate_ctc`.
6. **Etiquetas.** Cada carácter de la frase se codifica con `character_to_prediction_index.json`, desplazado
   en uno porque el índice 0 es el *blank* de CTC (59 caracteres + blank = 60 clases).
7. **Aumentación** (`src/augment.py`). Solo en entrenamiento, aplicada a la secuencia ya normalizada y
   antes del submuestreo. Cada transformación se aplica con probabilidad independiente (0.5 por defecto):
   cambio de velocidad (factor 0.8–1.25, interpolación lineal; la máscara se vuelve a binarizar), descarte de
   cuadros (10 %), escala (±10 %) y rotación (±10°) en el plano $x$–$y$, descarte de puntos (5 %, puestos en 0)
   y ruido gaussiano ($\sigma$ = 0.01). Validación y prueba nunca se aumentan.
8. **Modelos.** Cuatro arquitecturas con CTC: CNN 1D + BiGRU (principal), Transformer pequeño, LSTM
   unidireccional como línea base (con una GRU como ablación) y BiLSTM. Todas siguen la misma interfaz
   `forward(x, x_len) -> (log_probs, out_len)` y se guardan con el mismo formato.
9. **Entrenamiento** (`src/train.py`). Igual para todos: pérdida CTC, optimizador AdamW con ciclo de tasa de
   aprendizaje *one-cycle* (Loshchilov y Hutter, 2019; Smith y Topin, 2019), recorte de gradiente, semilla 42
   y *early stopping* por CER de validación. Se cambia un hiperparámetro a la vez respecto a una
   configuración base.
10. **Evaluación.** CER en validación para elegir entre variantes y CER en prueba, una sola vez, para
    comparar modelos. Se reportan además el error por carácter, el tiempo de entrenamiento, el tiempo de
    predicción por secuencia y el número de parámetros.

## División de los datos

El split se hace **por participante**, no por secuencia: ningún participante aparece en dos particiones. Si se
repartieran secuencias al azar, el modelo podría reconocer el estilo de una persona ya vista y el CER medido
sería optimista; con este esquema se mide la capacidad de generalizar a personas nuevas. Se usó la proporción
70 / 15 / 15 con semilla 42, guardada en `data/split.json` y versionada para que los cuatro modelos usen
exactamente la misma partición:

| Partición | Participantes | Secuencias |
|-----------|--------------:|-----------:|
| Entrenamiento | 64 | 1,315 |
| Validación | 14 | 295 |
| Prueba | 14 | 268 |

## Herramientas y cómputo

| Uso | Herramienta |
|-----|-------------|
| Lenguaje y análisis | Python 3.11, NumPy, pandas, SciPy, scikit-learn |
| Lectura de datos | PyArrow (archivos Parquet) |
| Modelado | PyTorch (Paszke et al., 2019) con `nn.CTCLoss` |
| Visualización | Matplotlib y seaborn |
| Entorno de trabajo | Jupyter Notebooks y Git/GitHub |
| Aplicación | Streamlit y Plotly |

El modelo principal se entrenó en CPU (sin GPU): cerca de 48 minutos para la variante final con 60 épocas. La
parte común de `src/train.py` usa CUDA si está disponible, lo que permitió entrenar los demás modelos en
Colab o Kaggle cuando hizo falta. En todos los casos el tiempo de entrenamiento y el dispositivo quedan
registrados en `reports/metrics/<modelo>.json`.

## Reproducibilidad

Se fija la semilla de `random`, NumPy y PyTorch; el split está versionado; el preprocesamiento es
determinista y se guarda en caché (`data/processed/secuencias_modelado.npz`); y los pesos y métricas de cada
modelo se guardan con sus hiperparámetros. La aplicación usa el mismo `src/features.py` para procesar una
secuencia nueva (`app/preprocess.py`), de modo que lo que ve el modelo en producción es lo mismo que vio al
entrenar.

## Limitaciones

- Solo 1,315 secuencias de entrenamiento y 64 participantes, frente a las ~67,000 secuencias del conjunto
  completo; por eso los CER no son comparables con los de la competencia.
- La mano izquierda falta casi siempre y la derecha en más de la mitad de los cuadros, lo que limita la señal
  disponible.
- El submuestreo a 384 cuadros descarta información en las secuencias más largas (alrededor de la cola de la
  distribución).
- La decodificación es *greedy*, sin modelo de lenguaje: no corrige letras con el contexto de la palabra.

---

# Referencias

Bai, S., Kolter, J. Z., & Koltun, V. (2018). *An empirical evaluation of generic convolutional and recurrent networks for sequence modeling*. arXiv. https://arxiv.org/abs/1803.01271

Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder–decoder for statistical machine translation. In *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing* (pp. 1724–1734). Association for Computational Linguistics. https://aclanthology.org/D14-1179

Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). *Empirical evaluation of gated recurrent neural networks on sequence modeling*. arXiv. https://arxiv.org/abs/1412.3555

Google & Deaf Professional Arts Network. (2023). *Google – American Sign Language Fingerspelling Recognition* [Competencia de Kaggle]. https://www.kaggle.com/competitions/asl-fingerspelling

Graves, A., Fernández, S., Gomez, F., & Schmidhuber, J. (2006). Connectionist temporal classification: Labelling unsegmented sequence data with recurrent neural networks. In *Proceedings of the 23rd International Conference on Machine Learning* (pp. 369–376). https://www.cs.toronto.edu/~graves/icml_2006.pdf

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation, 9*(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735

Korotaev, P., Surovtsev, P., Kapitanov, A., Kvanchiani, K., & Nagaev, A. (2025). *HandReader: Advanced techniques for efficient fingerspelling recognition*. arXiv. https://arxiv.org/abs/2505.10267

LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE, 86*(11), 2278–2324. https://doi.org/10.1109/5.726791

Loshchilov, I., & Hutter, F. (2019). *Decoupled weight decay regularization*. arXiv. https://arxiv.org/abs/1711.05101

Lugaresi, C., Tang, J., Nash, H., McClanahan, C., Uboweja, E., Hays, M., Zhang, F., Chang, C.-L., Yong, M. G., Lee, J., Chang, W.-T., Hua, W., Georg, M., & Grundmann, M. (2019). *MediaPipe: A framework for building perception pipelines*. arXiv. https://arxiv.org/abs/1906.08172

Padden, C. A., & Gunsauls, D. C. (2003). How the alphabet came to be used in a sign language. *Sign Language Studies, 4*(1), 10–33. https://doi.org/10.1353/sls.2003.0026

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., … Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. In *Advances in Neural Information Processing Systems 32* (pp. 8024–8035).

Schuster, M., & Paliwal, K. K. (1997). Bidirectional recurrent neural networks. *IEEE Transactions on Signal Processing, 45*(11), 2673–2681. https://doi.org/10.1109/78.650093

Shi, B., Del Rio, A. M., Keane, J., Michaux, J., Brentari, D., Shakhnarovich, G., & Livescu, K. (2018). American Sign Language fingerspelling recognition in the wild. In *2018 IEEE Spoken Language Technology Workshop (SLT)* (pp. 145–152). IEEE. https://doi.org/10.1109/SLT.2018.8639639

Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on image data augmentation for deep learning. *Journal of Big Data, 6*, Article 60. https://doi.org/10.1186/s40537-019-0197-0

Smith, L. N., & Topin, N. (2019). *Super-convergence: Very fast training of neural networks using large learning rates*. arXiv. https://arxiv.org/abs/1708.07120

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. In *Advances in Neural Information Processing Systems 30* (pp. 5998–6008).

Zhang, F., Bazarevsky, V., Vakunov, A., Tkachenka, A., Sung, G., Chang, C.-L., & Grundmann, M. (2020). *MediaPipe Hands: On-device real-time hand tracking*. arXiv. https://arxiv.org/abs/2006.10214
