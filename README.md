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

## Cómo obtener los datos

Los datos son grandes y no se versionan en GitHub. Para descargarlos:

1. Aceptar las reglas de la competencia en Kaggle: https://www.kaggle.com/competitions/asl-fingerspelling/rules
2. Configurar la API de Kaggle (`~/.kaggle/kaggle.json` con tu API token).
3. Descargar y descomprimir en `data/raw/`:

   ```bash
   kaggle competitions download -c asl-fingerspelling -p data/raw/
   unzip data/raw/asl-fingerspelling.zip -d data/raw/
   ```

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
