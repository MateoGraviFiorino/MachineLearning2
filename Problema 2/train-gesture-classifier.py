import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Cargar el dataset y las etiquetas
dataset = np.load('Problema 2/rps_dataset.npy')  # Reemplaza con el nombre correcto de tu archivo
labels = np.load('Problema 2/rps_labels.npy')  # Reemplaza con el nombre correcto de tu archivo

# Normalizar los datos
dataset = dataset / np.max(dataset)  # Normaliza los datos de entrada si es necesario

# Definir la arquitectura de la red neuronal
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(42,)),  # 42 entradas (21 puntos x 2)
    layers.Dropout(0.5),  # Regularización por Dropout
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),  # Regularización por Dropout
    layers.Dense(3, activation='softmax')  # 3 clases (piedra, papel, tijeras)
])

# Compilar el modelo
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Definir el callback para detener el entrenamiento temprano
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Dividir el dataset en conjunto de entrenamiento y validación (80-20)
X_train, X_val, y_train, y_val = train_test_split(dataset, labels, test_size=0.2, random_state=42)

# Entrenar el modelo
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Guardar el modelo entrenado
model.save('rps_model.h5')  # Guardar en formato .h5

print("Modelo entrenado y guardado como 'rps_model.h5'")
