import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from typing import Tuple
import matplotlib.pyplot as plt
from keras.layers import Dropout

class NeuralNetworkModel:
    def __init__(self, input_dim: int):
        """Inicializa el modelo de red neuronal."""
        self.model = Sequential()
        self.model.add(Dense(input_dim, activation='relu'))  # Aumentar neuronas
        self.model.add(Dense(10, activation='relu'))  # Capa oculta
        self.model.add(Dense(1, activation='linear'))  # Capa de salida

    def compile_model(self) -> None:
        """Compila el modelo."""
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, validation_split: float = 0.2, epochs: int = 100, batch_size: int = 64) -> None:
        """Entrena el modelo con los datos proporcionados."""
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        self.history = self.model.fit(X_train, y_train, validation_split=validation_split, 
                                       epochs=epochs, batch_size=batch_size, 
                                       callbacks=[early_stopping])

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
        """Evalúa el modelo con datos de prueba y retorna la pérdida y el MAE."""
        loss, mae = self.model.evaluate(X_test, y_test)
        print(f'Error absoluto medio en el conjunto de prueba: {mae}')
        return loss, mae

    def plot_training_history(self) -> None:
        """Visualiza el historial de entrenamiento."""

        plt.figure(figsize=(12, 6))
        plt.plot(self.history.history['loss'], label='Pérdida de entrenamiento')
        plt.plot(self.history.history['val_loss'], label='Pérdida de validación')
        plt.title('Pérdida durante el entrenamiento')
        plt.xlabel('Épocas')
        plt.ylabel('Pérdida')
        plt.legend()
        plt.show()


