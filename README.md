# Trabajo Práctico 1 - Aprendizaje Automático II

## Descripción General

Este proyecto corresponde al **Trabajo Práctico 1** de la asignatura **Aprendizaje Automático II** de la Universidad Nacional de Rosario, Facultad de Ciencias Exactas, Ingeniería y Agrimensura. El trabajo incluye el uso de redes neuronales densas y convolucionales para resolver diferentes problemas de predicción y clasificación.

### Autores
- **Mateo Gravi Fiorino**
- **Lucas Gauto**

---

## Contenido del Proyecto

### Problema 1: Predicción del Índice de Rendimiento Académico

El objetivo de este problema es construir un modelo de regresión utilizando redes neuronales para predecir el rendimiento académico de los estudiantes en función de un conjunto de características proporcionadas (horas de estudio, resultados anteriores, participación en actividades extracurriculares, etc.).

**Scripts:**
- `analisis.py`: Realiza el análisis de los datos y construye el modelo de regresión.
  - Para ejecutar el script: `streamlit run analisis.py`

**Entradas:**
- Dataset con información académica de los estudiantes (horas de estudio, horas de sueño, actividades extracurriculares, etc.).

**Modelos utilizados:**
- Red neuronal con capas densas.

---

### Problema 2: Clasificación de Gestos de Mano (Piedra, Papel o Tijeras)

Este problema tiene como objetivo implementar un sistema de clasificación de gestos de mano (piedra, papel o tijeras) utilizando MediaPipe para detectar los puntos clave de la mano (landmarks) y entrenar un modelo de red neuronal densa para clasificar los gestos.

**Scripts:**
1. `record-dataset.py`: Captura el dataset de gestos utilizando una cámara web y MediaPipe para detectar los landmarks de la mano.
2. `train-gesture-classifier.py`: Entrena el modelo de red neuronal utilizando los datos capturados.
3. `rock-paper-scissors.py`: Ejecuta la clasificación de gestos en tiempo real utilizando la cámara web y el modelo entrenado.

**Dataset:**
- Las imágenes de gestos de manos son capturadas en tiempo real y etiquetadas como "piedra", "papel" o "tijeras", almacenadas en archivos `.npy` para su uso en el entrenamiento del modelo.

**Modelos utilizados:**
- Red neuronal densa.

---

### Problema 3: Clasificación de Imágenes de Escenas Naturales

El objetivo de este problema es construir un modelo de clasificación utilizando redes neuronales convolucionales para clasificar imágenes de escenas naturales en una de las seis categorías: edificios, bosques, glaciares, montañas, mar y calles.

**Scripts:**
- Se proporcionan notebooks en `Google Colab` con el código para la clasificación de imágenes.

**Dataset:**
- 25,000 imágenes de escenas naturales, divididas en las categorías mencionadas.

**Modelos utilizados:**
- Red neuronal con capas densas.
- Red neuronal con capas convolucionales.
- Red neuronal con bloques residuales.
- Red neuronal con transferencia de aprendizaje (*backbone*).

---

## Estructura del Proyecto

TP1-AAII-2C-2024/
│
├── Problema 1/
│   ├── analisis.py
│
├── Problema 2/
│   ├── record-dataset.py
│   ├── train-gesture-classifier.py
│   ├── rock-paper-scissors.py
│
└── Problema 3/
    └── Notebooks en Google Colab

---

## Requisitos

- Python 3.x
- Librerías necesarias: TensorFlow, Keras, MediaPipe, Streamlit, NumPy, OpenCV

**Instalación de dependencias:**
```bash
pip install -r requirements.txt
