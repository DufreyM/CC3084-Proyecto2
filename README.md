# CC3084 - Proyecto 2: Análisis Exploratorio

**Universidad del Valle de Guatemala** — Facultad de Ingeniería
Departamento de Ciencias de la Computación — CC3084 Data Science — Semestre II 2026

## Grupo 5

- María José Girón Isidro
- Leonardo Dufrey Mejía Mejía
- Cindy Mishelle Gualim Perez
- Daniela Ramírez de León
- José Donado

## Reto seleccionado

**#15 — Google: Reconocimiento de deletreo manual del lenguaje de señas estadounidense (ASL Fingerspelling)**

Tema: Visión Artificial

Competencia de Kaggle: [ASL Fingerspelling](https://www.kaggle.com/competitions/asl-fingerspelling)

### Sobre el reto

El objetivo de la competencia es transcribir a texto secuencias de deletreo manual (fingerspelling)
en el lenguaje de señas americano (ASL), a partir de datos de landmarks (puntos de referencia de
manos, cara y pose) capturados con MediaPipe, en lugar de video/imagen crudo. Los datos vienen en
archivos `.parquet`, con coordenadas (x, y, z) de cada landmark por frame, agrupados por secuencia
(`sequence_id`) y asociados a una frase objetivo (`phrase`).

## Estructura del repositorio

```
CC3084-Proyecto2/
├── data/
│   ├── raw/            # Datos originales de Kaggle (no versionados, ver .gitignore)
│   └── processed/       # Datos limpios/transformados listos para EDA
├── notebooks/           # Notebooks de Jupyter/Colab para exploración y análisis
├── src/                 # Scripts reutilizables (carga de datos, limpieza, utilidades)
├── reports/
│   └── figures/         # Gráficos exportados para el informe y la presentación
├── requirements.txt
└── README.md
```

## Plan de trabajo y asignaciones

Cada tarea vive en un archivo independiente (notebook o markdown) para que cada
integrante pueda avanzar y **commitear su parte sin generar conflictos** con
el trabajo de los demás.

`notebooks/00_setup_datos.ipynb` (descarga de datos, entorno) es **trabajo
general/compartido**, no se cuenta en la carga individual de nadie — cualquiera
lo puede correr y ajustar. Las 7 tareas de EDA propiamente dichas no se
dividen exacto entre 5 personas (7/5), así que la repartición más pareja
posible es 2 archivos para dos integrantes y 1 archivo para las otras tres:

| # | Archivo | Actividad de la guía | Responsable |
|---|---------|----------------------|-------------|
| 0 | [notebooks/00_setup_datos.ipynb](notebooks/00_setup_datos.ipynb) | Setup: descarga de datos con `kagglehub` y entorno | General / compartido (todos) |
| 1 | [docs/01_planteamiento.md](docs/01_planteamiento.md) | Situación problemática, problema científico, objetivos (30 pts) | María José Girón Isidro |
| 2 | [docs/02_investigacion_tecnica.md](docs/02_investigacion_tecnica.md) | Actividad 1: investigación del tema | José Donado |
| 3 | [notebooks/01_estructura_datos.ipynb](notebooks/01_estructura_datos.ipynb) | Actividad 4a + Descripción de los datos | Leonardo Dufrey Mejía Mejía |
| 4 | [notebooks/02_limpieza_preprocesamiento.ipynb](notebooks/02_limpieza_preprocesamiento.ipynb) | Actividad 3: limpieza y preprocesamiento | Cindy Mishelle Gualim Perez |
| 5 | [notebooks/03_analisis_univariado.ipynb](notebooks/03_analisis_univariado.ipynb) | Actividad 4b: resumen numérico y tablas de frecuencia | Daniela Ramírez de León |
| 6 | [notebooks/04_analisis_bivariado.ipynb](notebooks/04_analisis_bivariado.ipynb) | Actividad 4c: cruce de variables y correlaciones | María José Girón Isidro |
| 7 | [notebooks/05_visualizaciones.ipynb](notebooks/05_visualizaciones.ipynb) | Actividad 4d: gráficos exploratorios | José Donado |
| 8 | [docs/03_hallazgos_conclusiones.md](docs/03_hallazgos_conclusiones.md) | Actividad 5: hallazgos y conclusiones | Todo el equipo (al final, una vez cerrados 1-7) |

`src/config.py` y `src/data_loading.py` son utilidades **compartidas** (rutas
y funciones de carga de `train.csv` / parquet de landmarks) — están en
progreso, no completas. Si necesitas una función nueva de carga de datos,
agrégala ahí en vez de duplicar código en tu notebook.

### Flujo de git sugerido (para evitar conflictos)

1. `git pull` antes de empezar a trabajar.
2. Trabajar **solo** en tu(s) archivo(s) asignado(s) de la tabla de arriba.
3. Commits pequeños y frecuentes, con mensajes claros, por ejemplo:
   `git commit -m "01_estructura_datos: describir columnas de train.csv"`.
4. `git push` seguido. Como cada quien toca archivos distintos, no debería
   haber conflictos de merge.
5. Si necesitas modificar un archivo de otra persona (p. ej. una función en
   `src/`), avisa en el grupo antes de hacerlo.

## Cómo obtener los datos

El dataset completo de la competencia son **158 GB** — no hace falta para un
EDA. Todo el equipo descarga la misma **muestra fija** (misma metadata
completa + 2 archivos de landmarks), corriendo
[notebooks/00_setup_datos.ipynb](notebooks/00_setup_datos.ipynb):

1. Aceptar las reglas de la competencia en Kaggle: https://www.kaggle.com/competitions/asl-fingerspelling/rules
2. Configurar credenciales de Kaggle (`~/.kaggle/kaggle.json`, API token desde https://www.kaggle.com/settings).
3. Correr `00_setup_datos.ipynb`, que descarga con la CLI de `kaggle` (no
   `kagglehub`: para archivos individuales devuelve el contenido comprimido en
   zip pero nombrado sin `.zip`, lo cual rompe la lectura en silencio):
   - `train.csv`, `supplemental_metadata.csv`, `character_to_prediction_index.json` (completos, pocos MB)
   - los 2 archivos de `train_landmarks/` listados en `config.SAMPLE_LANDMARK_PATHS`
     (~1.5 GB cada uno, ~1000 secuencias cada uno — de sobra para EDA)

Si en algún momento se necesitara el dataset completo (no para este
entregable), es `kagglehub.competition_download("asl-fingerspelling")` —
descarga los 158 GB en un solo archivo no reanudable.

**Importante:** todos deben usar la misma muestra (`config.SAMPLE_LANDMARK_PATHS`).
Si cada quien descarga archivos distintos, los notebooks que referencian esa
muestra fallan o dan resultados distintos según la máquina.

## Entorno de trabajo

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Objetivos (borrador)

**Objetivo general:**
Realizar un análisis exploratorio de los datos de la competencia ASL Fingerspelling para
comprender la estructura de las secuencias de landmarks y las características que permitan
plantear un modelo de reconocimiento de deletreo manual.

**Objetivos específicos:**
- Describir la estructura de los datos (variables, tipos, cantidad de secuencias/frames,
  distribución de longitudes de frase y de secuencia).
- Identificar patrones y problemas de calidad de datos (valores faltantes en landmarks,
  outliers, landmarks con baja confianza) que deban tratarse antes del modelado.
- Explorar relaciones entre la longitud de la secuencia, el número de landmarks disponibles
  y la longitud de la frase objetivo.

## Entregables

- Informe de análisis exploratorio (PDF).
- Presentación de PowerPoint.
- Este repositorio, versionado con las contribuciones de cada integrante.

**Fecha de entrega:** 11 de septiembre de 2025 (según guía del proyecto).
