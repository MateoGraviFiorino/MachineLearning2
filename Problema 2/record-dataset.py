import cv2
import numpy as np
import mediapipe as mp
import os

# Inicializar MediaPipe para detección de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Crear una carpeta para guardar las imágenes capturadas
def create_directory(label):
    gesture_names = {0: "piedra", 1: "papel", 2: "tijeras"}
    directory = gesture_names[label]
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    return directory

# Función para capturar las imágenes y detectar los landmarks
def capture_hand_landmarks(label, dataset_file='rps_dataset.npy', labels_file='rps_labels.npy'):
    # Inicializa la webcam
    cap = cv2.VideoCapture(0)

    landmarks_list = []
    labels_list = []

    directory = create_directory(label)
    image_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al acceder a la cámara.")
            break

        # Convertir la imagen a RGB porque MediaPipe usa RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesar la imagen para obtener los landmarks de la mano
        result = hands.process(image_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Dibujar los landmarks en la imagen
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extraer coordenadas x, y de los 21 landmarks
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y])
                landmarks = np.array(landmarks).flatten()  # Aplanar la lista para guardar

                # Añadir los landmarks a la lista
                landmarks_list.append(landmarks)
                labels_list.append(label)

                # Guardar la imagen capturada en la carpeta correspondiente
                image_path = os.path.join(directory, f"gesture_{image_counter}.jpg")
                cv2.imwrite(image_path, frame)
                image_counter += 1

        # Mostrar la imagen con los landmarks dibujados
        cv2.imshow("Captura de Gestos - Presiona 'q' para salir", frame)

        # Presionar 'q' para detener la captura
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar la cámara y cerrar ventanas
    cap.release()
    cv2.destroyAllWindows()

    # Guardar los landmarks y etiquetas en archivos .npy
    if os.path.exists(dataset_file) and os.path.exists(labels_file):
        existing_data = np.load(dataset_file)
        existing_labels = np.load(labels_file)
        landmarks_list = np.vstack((existing_data, landmarks_list))  # Concatenar con datos existentes
        labels_list = np.hstack((existing_labels, labels_list))  # Concatenar etiquetas existentes

    # Guardar los datos y etiquetas
    np.save(dataset_file, landmarks_list)
    np.save(labels_file, labels_list)

    print(f"Se han guardado {len(landmarks_list)} muestras en {dataset_file} y {labels_file}")
    print(f"Se han capturado {image_counter} imágenes en la carpeta '{directory}'.")

# Llamada para capturar los gestos y etiquetarlos:
# 0: Piedra, 1: Papel, 2: Tijeras

print("Capturando gestos de PIEDRA. Presiona 'q' cuando hayas terminado.")
capture_hand_landmarks(label=0)

print("Capturando gestos de PAPEL. Presiona 'q' cuando hayas terminado.")
capture_hand_landmarks(label=1)

print("Capturando gestos de TIJERAS. Presiona 'q' cuando hayas terminado.")
capture_hand_landmarks(label=2)
