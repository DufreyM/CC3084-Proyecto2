"""Arma reports/presentacion_final.pptx: version visual del informe para exponer."""
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "reports" / "figures"
REPORTS = ROOT / "reports"

# --- paleta -----------------------------------------------------------------
NAVY = RGBColor(0x0F, 0x1E, 0x33)
NAVY_2 = RGBColor(0x17, 0x2A, 0x45)
TEAL = RGBColor(0x2C, 0xA6, 0xA4)
ORANGE = RGBColor(0xF2, 0x8C, 0x28)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
INK = RGBColor(0x1B, 0x22, 0x2D)
MUTED = RGBColor(0x5B, 0x6B, 0x7C)
BG = RGBColor(0xF6, 0xF8, 0xFA)
LINE = RGBColor(0xE3, 0xE8, 0xEC)

FONT = "Calibri"

MIEMBROS = [
    "María José Girón Isidro",
    "Leonardo Dufrey Mejía Mejía",
    "Cindy Mishelle Gualim Perez",
    "Daniela Ramírez de León",
    "José Donado",
]

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
SW, SH = prs.slide_width, prs.slide_height


def new_slide(bg=WHITE):
    slide = prs.slides.add_slide(BLANK)
    rect = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SW, SH)
    rect.fill.solid()
    rect.fill.fore_color.rgb = bg
    rect.line.fill.background()
    rect.shadow.inherit = False
    # mandar el fondo hasta atras
    spTree = slide.shapes._spTree
    spTree.remove(rect._element)
    spTree.insert(2, rect._element)
    return slide


def add_rect(slide, left, top, width, height, color, line=False):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    if not line:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = color
    shp.shadow.inherit = False
    return shp


def add_text(slide, text, left, top, width, height, size=18, color=INK,
             bold=False, align=PP_ALIGN.LEFT, font=FONT, italic=False,
             anchor=MSO_ANCHOR.TOP, line_spacing=1.0):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line_spacing
        r = p.add_run()
        r.text = line
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.name = font
        r.font.color.rgb = color
    return box


def add_bullets(slide, items, left, top, width, height, size=16, color=INK,
                 marker_color=TEAL, gap=8, font=FONT):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = 0
    tf.margin_right = 0
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        p.line_spacing = 1.08
        r = p.add_run()
        r.text = f"›  {item}"
        r.font.size = Pt(size)
        r.font.name = font
        r.font.color.rgb = color
    return box


def add_picture_fit(slide, path, left, top, max_w, max_h):
    im = Image.open(path)
    iw, ih = im.size
    ratio = min(max_w / iw, max_h / ih)
    w, h = int(iw * ratio), int(ih * ratio)
    x = left + (max_w - w) // 2
    y = top + (max_h - h) // 2
    slide.shapes.add_picture(str(path), x, y, width=w, height=h)
    return x, y, w, h


def add_page_number(slide, n):
    add_text(slide, str(n), SW - Inches(0.7), SH - Inches(0.45), Inches(0.5),
              Inches(0.35), size=11, color=MUTED, align=PP_ALIGN.RIGHT)


def add_kicker(slide, text):
    add_rect(slide, Inches(0.6), Inches(0.55), Inches(0.35), Inches(0.06), TEAL)
    add_text(slide, text.upper(), Inches(1.05), Inches(0.42), Inches(8), Inches(0.35),
              size=13, color=TEAL, bold=True)


def title_slide():
    slide = new_slide(NAVY)
    add_rect(slide, 0, 0, Inches(0.28), SH, TEAL)
    add_rect(slide, Inches(0.9), Inches(1.55), Inches(0.6), Inches(0.08), ORANGE)
    add_text(slide, "PROYECTO 2 · ANÁLISIS EXPLORATORIO", Inches(0.9), Inches(1.0),
              Inches(10), Inches(0.4), size=15, color=TEAL, bold=True)
    add_text(slide, "Reconocimiento de deletreo manual\nen lenguaje de señas (ASL Fingerspelling)",
              Inches(0.9), Inches(1.85), Inches(11.2), Inches(2.0), size=36, bold=True,
              color=WHITE, line_spacing=1.05)
    add_text(slide, "Google · American Sign Language Fingerspelling Recognition (Kaggle, 2023)",
              Inches(0.9), Inches(3.75), Inches(11), Inches(0.5), size=16, color=RGBColor(0xB8, 0xC4, 0xD0))
    add_text(slide, "\n".join(MIEMBROS), Inches(0.9), Inches(4.6), Inches(6), Inches(2.0),
              size=15, color=WHITE, line_spacing=1.35)
    add_text(slide, "CC3084 · Data Science\nUniversidad del Valle de Guatemala\n6 de septiembre de 2026",
              Inches(9.2), Inches(6.3), Inches(3.3), Inches(1.1), size=12,
              color=RGBColor(0x8A, 0x98, 0xA8), align=PP_ALIGN.RIGHT, line_spacing=1.3)


def section_slide(numero, titulo, subtitulo):
    slide = new_slide(NAVY_2)
    add_rect(slide, 0, 0, Inches(0.28), SH, ORANGE)
    add_text(slide, numero, Inches(0.9), Inches(2.5), Inches(3), Inches(1.3),
              size=64, bold=True, color=TEAL)
    add_text(slide, titulo, Inches(0.9), Inches(3.7), Inches(11), Inches(1.2),
              size=34, bold=True, color=WHITE)
    add_text(slide, subtitulo, Inches(0.9), Inches(4.6), Inches(10.5), Inches(0.8),
              size=16, color=RGBColor(0xB8, 0xC4, 0xD0))
    return slide


def content_slide(kicker, title):
    slide = new_slide(BG)
    add_rect(slide, 0, 0, SW, Inches(1.35), NAVY)
    add_rect(slide, Inches(0.6), Inches(0.42), Inches(0.35), Inches(0.06), TEAL)
    add_text(slide, kicker.upper(), Inches(1.05), Inches(0.3), Inches(9), Inches(0.3),
              size=12, color=TEAL, bold=True)
    add_text(slide, title, Inches(0.6), Inches(0.6), Inches(11.5), Inches(0.7),
              size=25, bold=True, color=WHITE)
    return slide


def stat_card(slide, left, top, width, height, numero, etiqueta):
    add_rect(slide, left, top, width, height, WHITE, line=True)
    add_text(slide, numero, left + Inches(0.15), top + Inches(0.12), width - Inches(0.3),
              height - Inches(0.75), size=30, bold=True, color=ORANGE, align=PP_ALIGN.CENTER)
    add_text(slide, etiqueta, left + Inches(0.1), top + height - Inches(0.55), width - Inches(0.2),
              Inches(0.5), size=11.5, color=MUTED, align=PP_ALIGN.CENTER, line_spacing=1.0)


n = 0

# 1. Portada -------------------------------------------------------------
title_slide()

# 2. Contexto --------------------------------------------------------------
slide = content_slide("Contexto", "¿Qué es el deletreo manual (fingerspelling)?")
add_bullets(slide, [
    "Se deletrea letra por letra lo que no tiene seña propia: nombres, siglas, términos técnicos.",
    "Es muy rápido (5-6 letras por segundo) y muy usado: 12-35% del discurso en ASL.",
    "Reconocerlo automáticamente es distinto a reconocer señas de palabra completa: el significado depende del movimiento en el tiempo, no solo de la forma de la mano.",
], Inches(0.7), Inches(1.75), Inches(6.2), Inches(4.5), size=17, gap=18)
stat_card(slide, Inches(7.3), Inches(1.8), Inches(2.6), Inches(1.9), "5-6", "letras por\nsegundo")
stat_card(slide, Inches(10.1), Inches(1.8), Inches(2.6), Inches(1.9), "12-35%", "del discurso\nen ASL")
stat_card(slide, Inches(7.3), Inches(3.9), Inches(2.6), Inches(1.9), "94", "participantes\nen la muestra")
stat_card(slide, Inches(10.1), Inches(3.9), Inches(2.6), Inches(1.9), "67,208", "secuencias en\ntrain.csv")
add_page_number(slide, 2)

# 3. Landmarks / como se capturan los datos --------------------------------
slide = content_slide("Los datos", "Landmarks, no video: así se ven los datos")
add_picture_fit(slide, FIGS / "01_landmarks_un_cuadro.png", Inches(0.6), Inches(1.5),
                 Inches(5.6), Inches(5.6))
add_bullets(slide, [
    "MediaPipe convierte cada cuadro de video en puntos (x, y, z): 468 de rostro, 33 de pose y 21 por mano.",
    "Se usan coordenadas y no video por privacidad, tamaño (158 GB el set completo) e invarianza a fondo/iluminación.",
    "En este cuadro real se ve la mano agrupada cerca de la cara, típico en fingerspelling.",
], Inches(6.5), Inches(1.9), Inches(6.2), Inches(4.5), size=16.5, gap=18)
add_page_number(slide, 3)

# 4. Descripcion de los datos / estructura ---------------------------------
slide = content_slide("Descripción de los datos", "Estructura del conjunto de datos")
add_bullets(slide, [
    "train.csv: 67,208 secuencias, 5 columnas (path, phrase, file_id, sequence_id, participant_id).",
    "94 participantes muy desbalanceados: de 1 a 1,535 secuencias por persona (mediana 794).",
    "Cada parquet de landmarks trae ~1,000 secuencias; 1,630 columnas por cuadro.",
    "Frases de 1 a 31 caracteres (mediana 17).",
], Inches(0.7), Inches(1.75), Inches(11.9), Inches(3.0), size=18, gap=14)
stat_card(slide, Inches(0.7), Inches(5.0), Inches(2.85), Inches(1.7), "5", "columnas en\ntrain.csv")
stat_card(slide, Inches(3.75), Inches(5.0), Inches(2.85), Inches(1.7), "1,630", "columnas por\ncuadro de landmarks")
stat_card(slide, Inches(6.8), Inches(5.0), Inches(2.85), Inches(1.7), "94", "participantes")
stat_card(slide, Inches(9.85), Inches(5.0), Inches(2.85), Inches(1.7), "1 - 31", "caracteres\npor frase")
add_page_number(slide, 4)

# 5. Limpieza ---------------------------------------------------------------
slide = content_slide("Limpieza y preprocesamiento", "El problema no es la suciedad: son detecciones faltantes")
add_picture_fit(slide, FIGS / "02_missing_landmarks_boxplot.png", Inches(6.7), Inches(1.6),
                 Inches(6.0), Inches(5.4))
add_bullets(slide, [
    "Rostro y pose casi no faltan; la mano derecha falta en 54.5% de los cuadros en promedio.",
    "No es error de captura: es ausencia de detección (mano fuera de cuadro, oclusión, movimiento rápido).",
    "Se aplicó forward-fill dentro de cada secuencia (no interpolación, no ceros).",
    "Se eliminaron las 468 columnas de rostro (86% de las columnas) y las secuencias con mano detectada en menos del 10% de sus cuadros.",
    "De 1,630 columnas se pasó a 226.",
], Inches(0.6), Inches(1.85), Inches(5.8), Inches(5.2), size=16, gap=16)
add_page_number(slide, 5)

# 6. Distribuciones univariadas ---------------------------------------------
slide = content_slide("Análisis exploratorio", "Longitud de frase y de secuencia")
add_picture_fit(slide, FIGS / "hist_phrase_len.png", Inches(0.5), Inches(1.6), Inches(6.0), Inches(4.6))
add_picture_fit(slide, FIGS / "hist_seq_len.png", Inches(6.7), Inches(1.6), Inches(6.0), Inches(4.6))
add_bullets(slide, [
    "Frase: 1-31 caracteres, casi simétrica, moda marcada en 12.",
    "Secuencia: sesgada a la derecha, mediana 147 cuadros, cola hasta 751.",
], Inches(0.6), Inches(6.3), Inches(12.1), Inches(0.9), size=15.5, gap=6)
add_page_number(slide, 6)

# 7. Relacion frase-secuencia -------------------------------------------
slide = content_slide("Análisis exploratorio", "A mayor frase, secuencias más largas")
add_picture_fit(slide, FIGS / "scatter_seq_vs_phrase.png", Inches(0.6), Inches(1.5), Inches(7.5), Inches(5.6))
stat_card(slide, Inches(8.6), Inches(2.4), Inches(3.9), Inches(2.0), "r = 0.60", "correlación de Pearson\nfrase vs. secuencia")
add_bullets(slide, [
    "Correlación positiva moderada, no estricta.",
    "La velocidad de deletreo varía entre personas.",
], Inches(8.6), Inches(4.7), Inches(3.9), Inches(1.8), size=15.5, gap=10)
add_page_number(slide, 7)

# 8. Correlaciones -----------------------------------------------------
slide = content_slide("Análisis exploratorio", "Matriz de correlación")
add_picture_fit(slide, FIGS / "heatmap_correlacion.png", Inches(0.6), Inches(1.5), Inches(7.3), Inches(5.6))
add_bullets(slide, [
    "Frase vs. secuencia: r = 0.60 (positiva moderada).",
    "Landmarks faltantes vs. secuencia: r = -0.19 (negativa débil).",
    "Frase vs. landmarks faltantes: r = -0.01 (prácticamente nula).",
], Inches(8.2), Inches(2.2), Inches(4.4), Inches(3.0), size=16.5, gap=16)
add_page_number(slide, 8)

# 9. Participantes -------------------------------------------------------
slide = content_slide("Análisis exploratorio", "Diferencias entre participantes")
add_picture_fit(slide, FIGS / "box_seq_len_by_participant.png", Inches(0.5), Inches(1.55), Inches(12.3), Inches(4.6))
add_bullets(slide, [
    "Las medianas por participante van de ~75 a ~241 cuadros: hay quienes deletrean más rápido o más lento.",
], Inches(0.6), Inches(6.3), Inches(12.1), Inches(0.9), size=16, gap=6)
add_page_number(slide, 9)

# 10. Caracteres -------------------------------------------------------
slide = content_slide("Análisis exploratorio", "Frecuencia de caracteres: alfabeto desbalanceado")
add_picture_fit(slide, FIGS / "bar_char_frequency.png", Inches(0.5), Inches(1.55), Inches(8.1), Inches(4.9))
add_bullets(slide, [
    "59 símbolos distintos: 61% letras, 25% dígitos, 5% espacios, 9% otros símbolos.",
    "e, a, o, r, n, t concentran la mayoría de apariciones.",
    "j, q, z y varios símbolos casi no tienen muestras.",
], Inches(8.9), Inches(2.3), Inches(3.9), Inches(3.5), size=15.5, gap=14)
add_page_number(slide, 10)

# 11. Trayectoria --------------------------------------------------------
slide = content_slide("Análisis exploratorio", "Cómo se mueve una mano al deletrear")
add_picture_fit(slide, FIGS / "sample_hand_trajectory.png", Inches(0.6), Inches(1.5), Inches(7.5), Inches(5.6))
add_bullets(slide, [
    "Trayectoria 2D de los landmarks de una mano en una secuencia completa.",
    "Se ven zonas donde el trazo se concentra: pausas y cambios de forma entre letras.",
    "Refuerza por qué el orden temporal importa tanto como la posición.",
], Inches(8.3), Inches(2.3), Inches(4.3), Inches(3.5), size=15.5, gap=16)
add_page_number(slide, 11)

# 12. Hallazgos principales ------------------------------------------------
_slide_hallazgos = section_slide("", "Hallazgos principales", "Lo que encontramos en el análisis exploratorio")
add_page_number(_slide_hallazgos, 12)

slide = content_slide("Hallazgos", "Resumen de hallazgos")
add_bullets(slide, [
    "Los datos son secuencias de landmarks, no video: 1,630 columnas por cuadro, con la mano derecha ausente en más de la mitad de los cuadros en promedio.",
    "94 participantes y 59 caracteres muy desbalanceados entre sí: cualquier evaluación futura debe separarse por participante y reportarse por carácter.",
    "La longitud de secuencia (147 cuadros de mediana, sesgada a la derecha) no es tan predecible como la longitud de frase; un futuro modelo necesita aceptar longitudes variables.",
    "La correlación frase-secuencia (r = 0.60) es la relación más fuerte encontrada; el resto de cruces son débiles o nulos.",
    "Se detectaron 28 secuencias atípicas (~3%) con el criterio de 1.5·IQR, todas por longitud inusualmente grande.",
], Inches(0.7), Inches(1.9), Inches(11.9), Inches(4.9), size=18, gap=20)
add_page_number(slide, 13)

# 13. Conclusiones y siguientes pasos --------------------------------------
slide = content_slide("Conclusiones", "Próximos pasos para el modelado")
add_bullets(slide, [
    "Tratar landmarks faltantes con forward-fill, no interpolación ni ceros.",
    "Descartar las 468 columnas de rostro: aportan poco y encarecen el procesamiento.",
    "Separar entrenamiento y validación por participante, no al azar.",
    "Aceptar longitud de secuencia variable (recorte o relleno).",
    "Reportar desempeño por carácter, no solo en promedio.",
], Inches(0.7), Inches(1.9), Inches(11.9), Inches(4.9), size=18, gap=20)
add_page_number(slide, 14)

# 14. Cierre -----------------------------------------------------------
slide = new_slide(NAVY)
add_rect(slide, 0, 0, Inches(0.28), SH, TEAL)
add_text(slide, "Gracias", Inches(0.9), Inches(2.7), Inches(8), Inches(1.3), size=48, bold=True, color=WHITE)
add_text(slide, "Preguntas y comentarios", Inches(0.9), Inches(3.75), Inches(8), Inches(0.6), size=18, color=RGBColor(0xB8, 0xC4, 0xD0))
add_text(slide, "Repositorio: https://github.com/DufreyM/CC3084-Proyecto2", Inches(0.9), Inches(6.6),
          Inches(9), Inches(0.4), size=13, color=TEAL)

out_path = REPORTS / "presentacion_final.pptx"
prs.save(out_path)
print(f"Presentacion guardada en {out_path} ({len(prs.slides)} slides)")
