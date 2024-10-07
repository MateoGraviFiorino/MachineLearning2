import matplotlib.pyplot as plt
from typing import Tuple
from keras.models import Model
from redes import NeuralNetworkModel 
import pandas as pd

class ModelEvaluator:
    def __init__(self, model: Model, history: dict):
        """Inicializa el evaluador con el modelo y el historial de entrenamiento."""
        self.model = model
        self.history = history

    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[float, float]:
        """Evalúa el modelo y retorna la pérdida y el MAE."""
        loss, mae = self.model.evaluate(X_test, y_test)
        print(f'Error absoluto medio en el conjunto de prueba: {mae}')
        return loss, mae

    def plot_training_history(self) -> None:
        """Visualiza el historial de entrenamiento del modelo."""
        plt.figure(figsize=(12, 6))
        plt.plot(self.history.history['loss'], label='Pérdida de entrenamiento')
        plt.plot(self.history.history['val_loss'], label='Pérdida de validación')
        plt.title('Pérdida durante el entrenamiento')
        plt.xlabel('Épocas')
        plt.ylabel('Pérdida')
        plt.legend()
        plt.show()
