import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp

# Cargar el modelo entrenado
model = load_model('Problema 2/rps_model.h5')

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Función para procesar la imagen y predecir el gesto
def preprocess_and_predict(image):
    # Convertir la imagen a RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Dibujar los landmarks
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extraer los landmarks y formatearlos para el modelo
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append(landmark.x)
                landmarks.append(landmark.y)
            landmarks = np.array(landmarks).reshape(1, -1)

            # Realizar la predicción
            prediction = model.predict(landmarks)
            class_id = np.argmax(prediction)
            gestures = ['rock', 'paper', 'scissors']
            gesture = gestures[class_id]

            # Mostrar el gesto reconocido en la imagen
            cv2.putText(image, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return image

# Captura de vídeo en tiempo real
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Procesar y predecir el gesto
    output_frame = preprocess_and_predict(frame)

    # Mostrar la imagen procesada
    cv2.imshow('Rock Paper Scissors', output_frame)

    # Salir del bucle si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()

import os
import time

# Crear una carpeta para guardar las imágenes
output_dir = 'Problema 2/captured_gestures'
os.makedirs(output_dir, exist_ok=True)

# Guardar la imagen si se presiona 's'
if cv2.waitKey(1) & 0xFF == ord('s'):
    img_name = f"{output_dir}/gesture_{int(time.time())}.png"
    cv2.imwrite(img_name, output_frame)
    print(f"Imagen guardada: {img_name}")
