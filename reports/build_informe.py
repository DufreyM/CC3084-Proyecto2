"""Ensambla docs/ + hallazgos de los notebooks en reports/informe_final.md/.pdf."""
import re
from pathlib import Path

import markdown as md_lib
from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
FIGS = ROOT / "reports" / "figures"
REPORTS = ROOT / "reports"

MIEMBROS = [
    "María José Girón Isidro",
    "Leonardo Dufrey Mejía Mejía",
    "Cindy Mishelle Gualim Perez",
    "Daniela Ramírez de León",
    "José Donado",
]


def strip_meta(texto, encabezado_nuevo=None):
    """Quita la primera linea (# titulo), 'Responsable principal' y el bloque de cita."""
    lineas = texto.split("\n")
    lineas = [l for l in lineas if not l.startswith("**Responsable principal:**")]
    lineas = [l for l in lineas if not l.startswith(">")]
    if lineas and lineas[0].startswith("# "):
        lineas = lineas[1:]
    cuerpo = "\n".join(lineas).strip()
    if encabezado_nuevo:
        cuerpo = f"# {encabezado_nuevo}\n\n{cuerpo}"
    return cuerpo


def _reemplazar_encabezado_exacto(texto, encabezado_viejo, encabezado_nuevo):
    """Reemplaza solo si encabezado_viejo ocupa la linea completa (evita
    colisiones cuando un encabezado es substring de otro, p.ej. '## Objetivos'
    dentro de '### Objetivos específicos')."""
    patron = r"(?m)^" + re.escape(encabezado_viejo) + r"\s*$"
    return re.sub(patron, encabezado_nuevo, texto, count=1)


_fig_counter = [0]


def fig(name, caption):
    _fig_counter[0] += 1
    src = (FIGS / name).as_posix()
    return (
        f'\n\n<div class="figura">'
        f'<img src="{src}">'
        f'<p class="leyenda">Figura {_fig_counter[0]}. {caption}</p>'
        f"</div>\n\n"
    )


# ---------------------------------------------------------------------------
# 1. Investigacion del tema (docs/02)
# ---------------------------------------------------------------------------
investigacion = strip_meta(
    (DOCS / "02_investigacion_tecnica.md").read_text(encoding="utf-8"),
    encabezado_nuevo="1. Investigación del tema",
)
investigacion = investigacion.replace(
    "## 4. Referencias", ""
)  # las referencias van al final del informe, no aqui
# separar las referencias del cuerpo de la seccion 1
partes_inv = investigacion.split("Al-Qaderi, M.")
investigacion = partes_inv[0].strip()
referencias_texto = "Al-Qaderi, M." + partes_inv[1] if len(partes_inv) > 1 else ""

# renumerar subsecciones internas (2 -> 1.1, 3 -> 1.2, etc.)
investigacion = _reemplazar_encabezado_exacto(
    investigacion,
    "## 1. ¿Qué es el deletreo manual (fingerspelling) en ASL?",
    "## 1.0 Qué es el deletreo manual (fingerspelling) en ASL",
)
investigacion = _reemplazar_encabezado_exacto(
    investigacion, "## 2. Cómo se capturan los datos: MediaPipe y landmarks",
    "## 1.1 Cómo se capturan los datos: MediaPipe y landmarks",
)
investigacion = _reemplazar_encabezado_exacto(
    investigacion,
    "## 3. Técnicas para reconocer patrones en secuencias (landmarks -> texto)",
    "## 1.2 Técnicas para reconocer patrones en secuencias de landmarks",
)
# las 4 tecnicas venian como lista numerada pegada al parrafo (sin sangria de
# continuacion); se convierten a subtitulos en negrita para que el numero
# no se pierda en la conversion a PDF.
for _n, _titulo in [
    (1, "RNN y LSTM (Redes Neuronales Recurrentes)"),
    (2, "CNN 1D (Redes Neuronales Convolucionales Unidimensionales)"),
    (3, "Transformers (Modelos basados en Atención)"),
    (4, "CTC (Connectionist Temporal Classification)"),
]:
    investigacion = investigacion.replace(
        f"{_n}. {_titulo}", f"**{_n}. {_titulo}**\n"
    )

# ---------------------------------------------------------------------------
# 2, 3, 4. Planteamiento (docs/01): situacion, problema cientifico, objetivos
# ---------------------------------------------------------------------------
planteamiento = (DOCS / "01_planteamiento.md").read_text(encoding="utf-8")
planteamiento = planteamiento.replace("# Planteamiento del problema\n\n", "")
planteamiento = _reemplazar_encabezado_exacto(planteamiento, "## Situación problemática", "# 2. Situación problemática")
planteamiento = _reemplazar_encabezado_exacto(planteamiento, "## Problema científico", "# 3. Problema científico")
planteamiento = _reemplazar_encabezado_exacto(planteamiento, "## Objetivos", "# 4. Objetivos")
planteamiento = _reemplazar_encabezado_exacto(planteamiento, "### Objetivo general", "## 4.1 Objetivo general")
planteamiento = _reemplazar_encabezado_exacto(planteamiento, "### Objetivos específicos", "## 4.2 Objetivos específicos")
planteamiento = _reemplazar_encabezado_exacto(
    planteamiento, "### Trazabilidad de los objetivos", "## 4.3 Trazabilidad de los objetivos"
)
# la seccion de Referencias de docs/01 se une a las del resto del informe
planteamiento, _, refs_planteamiento = planteamiento.partition("## Referencias")

# los objetivos especificos usan lista numerada + negrita al inicio de cada
# item; esa combinacion hace que xhtml2pdf no dibuje el numero. Se saca el
# numero de la lista nativa y se deja como texto en negrita.
planteamiento = re.sub(
    r"(?m)^(\d+)\. \*\*", r"**\1.** **", planteamiento
)

# ---------------------------------------------------------------------------
# 5. Descripcion de los datos (redactado a partir de 01 y 02)
# ---------------------------------------------------------------------------
descripcion_datos = f"""# 5. Descripción de los datos

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

{fig("01_landmarks_un_cuadro.png", "Landmarks de un cuadro real de una secuencia (rostro, pose y mano derecha). El rostro forma un bloque compacto arriba, pose se dispersa por el resto del cuerpo, y la mano queda agrupada cerca de la cara, algo esperable en fingerspelling.")}

## 5.2 Limpieza y preprocesamiento

Como se explicó en la sección de investigación del tema, los landmarks faltantes no
son un error de captura, son ausencia de detección. En la muestra de este proyecto
el rostro está presente casi siempre (falta en un 0.7% de los cuadros) y la pose
prácticamente nunca falta, mientras que las manos son harina de otro costal: la mano
izquierda falta en casi todos los cuadros y la mano derecha falta en un 54.5% de
ellos en promedio, es decir, el problema de calidad se concentra justo en la parte
del cuerpo que más información aporta para el deletreo.

{fig("02_missing_landmarks_boxplot.png", "Proporción de valores faltantes por tipo de landmark en la muestra fija. El rostro y la pose casi no faltan; las manos, sobre todo la izquierda, sí.")}

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
"""

# ---------------------------------------------------------------------------
# 6. Analisis exploratorio (redactado a partir de 03, 04 y 05)
# ---------------------------------------------------------------------------
analisis_exploratorio = f"""# 6. Análisis exploratorio

## 6.1 Variables cuantitativas

train.csv no trae ninguna variable numérica propia, así que el análisis cuantitativo
se hizo sobre tres variables derivadas: la longitud de la frase objetivo (en
caracteres), la longitud de la secuencia (en cuadros) y el porcentaje de cuadros sin
ninguna mano detectada.

La longitud de frase va de 1 a 31 caracteres, con media de 17.8 y mediana de 17: es
una distribución casi simétrica y acotada por arriba, con una moda marcada en 12
caracteres que se nota claramente en el histograma.

{fig("hist_phrase_len.png", "Distribución de la longitud de frase (en caracteres) sobre las 67,208 secuencias de train.csv.")}

La longitud de secuencia se comporta distinto: la media es de 160.8 cuadros, la
mediana de 147, y hay una cola larga hacia la derecha que llega hasta 751 cuadros
(asimetría de 1.09). Frases de largo parecido pueden tomar duraciones muy distintas
según la persona y el momento.

{fig("hist_seq_len.png", "Distribución de la longitud de secuencia (en cuadros), con sesgo marcado hacia la derecha.")}

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

{fig("scatter_seq_vs_phrase.png", "Relación entre longitud de frase y longitud de secuencia (r de Pearson igual a 0.60), con línea de tendencia.")}

También se cruzó la longitud de secuencia con el porcentaje promedio de landmarks
faltantes, y salió una correlación negativa pero débil (r de -0.19): las secuencias
más largas tienden a tener, en promedio, un poco menos de landmarks faltantes. En
cambio, la correlación entre longitud de frase y landmarks faltantes resultó
prácticamente nula (r de -0.01), así que una variable no explica a la otra.

{fig("heatmap_correlacion.png", "Matriz de correlación entre longitud de frase, longitud de secuencia y proporción promedio de landmarks faltantes.")}

También se revisó cómo varía la longitud de secuencia entre participantes: las
medianas por persona van aproximadamente de 75 a 241 cuadros, es decir, hay
diferencias reales en qué tan rápido o lento deletrea cada quien.

{fig("box_seq_len_by_participant.png", "Longitud de secuencia por participante (15 participantes con más datos), comparada contra la mediana global de 147 cuadros.")}

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

{fig("bar_char_frequency.png", "Frecuencia absoluta de cada carácter presente en las frases objetivo, de mayor a menor.")}

Por último, para ilustrar cómo se ve el movimiento real de una mano durante el
deletreo, se graficó la trayectoria de los landmarks de la mano a lo largo de todos
los cuadros de una secuencia de ejemplo. Se nota una nube de puntos con varias zonas
donde el trazo se concentra, compatible con las pausas y cambios de forma que ocurren
al pasar de una letra a otra.

{fig("sample_hand_trajectory.png", "Trayectoria en 2D de los landmarks de la mano a lo largo de una secuencia completa de ejemplo.")}
"""

# ---------------------------------------------------------------------------
# 7. Hallazgos y conclusiones (docs/03)
# ---------------------------------------------------------------------------
hallazgos = strip_meta(
    (DOCS / "03_hallazgos_conclusiones.md").read_text(encoding="utf-8"),
    encabezado_nuevo="7. Hallazgos y conclusiones",
)
hallazgos = hallazgos.replace("(Actividad 5 de la guía)\n\n", "")
hallazgos = _reemplazar_encabezado_exacto(
    hallazgos, "## Síntesis de la investigación técnica", "## 7.1 Síntesis de la investigación técnica"
)
hallazgos = _reemplazar_encabezado_exacto(
    hallazgos, "## Resumen de hallazgos", "## 7.2 Resumen de hallazgos por notebook"
)
hallazgos = _reemplazar_encabezado_exacto(
    hallazgos, "## Problemas de calidad de datos encontrados", "## 7.3 Problemas de calidad de datos encontrados"
)
hallazgos = _reemplazar_encabezado_exacto(
    hallazgos, "## Conclusiones sobre los siguientes pasos", "## 7.4 Conclusiones sobre los siguientes pasos"
)
# quitar los enlaces a notebooks, se dejan solo los nombres en texto
hallazgos = re.sub(r"\[([^\]]+)\]\(\.\./notebooks/[^)]+\)", r"\1", hallazgos)
# el resumen por notebook usa "* **nombre**:" como lista; con negrita al
# inicio del item, xhtml2pdf no dibuja el marcador de forma consistente, asi
# que se deja como parrafo con negrita en vez de lista nativa.
hallazgos = re.sub(r"(?m)^\* \*\*", "**", hallazgos)
hallazgos = re.sub(r"(?m)^  (?=\S)", "", hallazgos)

# ---------------------------------------------------------------------------
# 8. Referencias (docs/01 + docs/02, sin duplicar la de Kaggle)
# ---------------------------------------------------------------------------
# docs/01 trae sus referencias como lista "- " sin linea en blanco entre
# items; se separan para que el split posterior por parrafo (doble salto de
# linea) detecte cada referencia como un elemento independiente.
refs_planteamiento = re.sub(r"\n(?=- )", "\n\n", refs_planteamiento.strip())

referencias = "# 8. Referencias\n\n" + refs_planteamiento.strip() + "\n\n" + referencias_texto.strip()
# la cita de Kaggle en docs/02 es una version mas corta de la misma fuente que
# ya aparece en docs/01 (con la Deaf Professional Arts Network); se deja solo
# la de docs/01.
referencias = referencias.replace(
    "Google. (2023). Google, American Sign Language Fingerspelling Recognition "
    "[Conjunto de datos]. Kaggle. https://www.kaggle.com/competitions/asl-fingerspelling\n\n",
    "",
)


def _clave_referencia(entrada):
    antes = entrada.split("(")[0].lower()
    antes = re.sub(r"\s+(y|and|&)\s+", " ", antes)
    return re.sub(r"[^a-z0-9]", "", antes)


lineas_ref = referencias.split("\n\n")
vistas = set()
lineas_unicas = []
for l in lineas_ref:
    clave = _clave_referencia(l)
    if clave and clave in vistas:
        continue
    if clave:
        vistas.add(clave)
    lineas_unicas.append(l)
referencias = "\n\n".join(lineas_unicas)

# ---------------------------------------------------------------------------
# Ensamblaje final
# ---------------------------------------------------------------------------
informe_md = "\n\n".join([
    investigacion,
    planteamiento,
    descripcion_datos,
    analisis_exploratorio,
    hallazgos,
    referencias,
])

# limpieza tipografica: sin backticks, sin simbolos poco comunes, sin lineas ---
informe_md = informe_md.replace("`", "")
informe_md = re.sub(r"(?m)^-{3,}\s*$", "", informe_md)
informe_md = re.sub(r"\s*[‒–—―]\s*", ", ", informe_md)
informe_md = re.sub(r"\s+--+\s+", ", ", informe_md)
# asegurar linea en blanco antes de listas "- " que vienen pegadas a un parrafo,
# pero sin separar los items de la lista entre si (evita listas "sueltas" con <p>
# dentro de cada <li>, que en xhtml2pdf pierden el bullet).
def _separar_listas(texto):
    lineas = texto.split("\n")
    salida = []
    dentro_de_lista = False
    for linea in lineas:
        es_inicio_item = bool(re.match(r"^(-\s|\d+\.\s)", linea))
        es_continuacion = linea.startswith(("   ", "\t")) and linea.strip() != ""
        if es_inicio_item:
            if not dentro_de_lista and salida and salida[-1].strip() != "":
                salida.append("")
            dentro_de_lista = True
        elif linea.strip() == "" or not es_continuacion:
            dentro_de_lista = False
        salida.append(linea)
    return "\n".join(salida)


informe_md = _separar_listas(informe_md)
informe_md = re.sub(r"~(\d)", r"aprox. \1", informe_md)
informe_md = informe_md.replace("~", "")
informe_md = re.sub(r"\\\(\s*r\s*\\approx\s*([\-0-9.]+)\s*\\\)", r"r de aproximadamente \1", informe_md)
informe_md = re.sub(r"\(\(r\s*\\approx\s*([\-0-9.]+)\)\)", r"(r de aproximadamente \1)", informe_md)
informe_md = re.sub(r"\\approx\s*", "aproximadamente ", informe_md)
informe_md = informe_md.replace("\\(", "").replace("\\)", "")
informe_md = informe_md.replace("≈", " aproximadamente ")
# permitir que las URLs largas se corten de linea (con <wbr> despues de cada
# "/"), porque xhtml2pdf no soporta word-wrap/overflow-wrap y una URL sin
# puntos de corte se sale del margen de la pagina.
informe_md = re.sub(
    r"https?://\S+", lambda m: m.group(0).replace("/", "/<wbr/>"), informe_md
)

informe_path = REPORTS / "informe_final.md"
informe_path.write_text(informe_md, encoding="utf-8")
print(f"Markdown guardado en {informe_path} ({len(informe_md)} caracteres)")

# ---------------------------------------------------------------------------
# Conversion a PDF
# ---------------------------------------------------------------------------
caratula = f"""
<div style="text-align: center; margin-top: 40pt;">
<p style="text-align: center; font-size: 15pt; font-weight: bold;">UNIVERSIDAD DEL VALLE DE GUATEMALA</p>
<p style="text-align: center; font-size: 12pt;">Facultad de Ingeniería</p>
<p style="text-align: center; font-size: 12pt;">Departamento de Ciencias de la Computación</p>
<p style="text-align: center; font-size: 13pt;">CC3084 - Data Science</p>
<p style="text-align: center; font-size: 12pt;">Semestre II - 2026</p>
<p style="text-align: center; margin-top: 55pt; font-size: 16pt; font-weight: bold;">Proyecto 2</p>
<p style="text-align: center; font-size: 14pt;">Informe de Análisis Exploratorio</p>
<p style="text-align: center; font-size: 12pt;">Google: Reconocimiento de deletreo manual del lenguaje de<br>señas estadounidense (ASL Fingerspelling)</p>
<p style="text-align: center; margin-top: 55pt; font-size: 12pt;">{"<br>".join(MIEMBROS)}</p>
<p style="text-align: center; margin-top: 55pt; font-size: 12pt;">6 de septiembre de 2026</p>
<p style="text-align: center; margin-top: 30pt; font-size: 10pt;">Repositorio: https://github.com/DufreyM/CC3084-Proyecto2</p>
</div>
<p style="page-break-after: always;"></p>
"""

html_body = md_lib.markdown(informe_md, extensions=["tables", "nl2br"])
html_full = f"""<html><head><meta charset="utf-8"><style>
@page {{
    size: letter portrait;
    margin: 2.2cm 2cm 2cm 2cm;
    @frame footer_frame {{
        -pdf-frame-content: footer_content;
        bottom: 1cm; margin-left: 2cm; margin-right: 2cm; height: 1cm;
    }}
}}
* {{ color: #000000 !important; }}
body {{ font-family: "Times New Roman", Times, serif; font-size: 12pt; line-height: 1.45;
        color: #000000; background: #ffffff; }}
h1 {{ font-size: 17pt; font-weight: bold; margin-top: 4pt; margin-bottom: 10pt;
      page-break-before: always; }}
h2 {{ font-size: 13.5pt; font-weight: bold; margin-top: 16pt; margin-bottom: 6pt; }}
h3 {{ font-size: 12pt; font-weight: bold; margin-top: 12pt; margin-bottom: 6pt; }}
p, li {{ text-align: justify; margin-bottom: 6pt; word-wrap: break-word; }}
ul {{ margin-bottom: 10pt; padding-left: 18pt; list-style-type: disc; }}
ol {{ margin-bottom: 10pt; padding-left: 18pt; list-style-type: decimal; }}
.figura {{ text-align: center; margin: 16pt 0; }}
.figura img {{ max-width: 380px; }}
.leyenda {{ font-style: italic; font-size: 10pt; text-align: center; margin-top: 4pt; }}
table {{ border-collapse: collapse; width: 100%; font-size: 10.5pt; margin-bottom: 10pt; }}
td, th {{ border: 1px solid #000000; padding: 3px; text-align: left; }}
#footer_content {{ text-align: center; font-size: 9pt; }}
</style></head><body>
{caratula}
<div id="footer_content">Página <pdf:pagenumber> de <pdf:pagecount></div>
{html_body}
</body></html>"""

pdf_path = REPORTS / "informe_final.pdf"
with open(pdf_path, "wb") as f:
    resultado = pisa.CreatePDF(html_full, dest=f)

print(f"PDF generado en {pdf_path} (errores: {resultado.err})")
