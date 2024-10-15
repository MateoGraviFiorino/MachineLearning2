import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple
from procesamiento import DataPreprocessor
from redes import NeuralNetworkModel
from eval import ModelEvaluator
from data_set import data

def main() -> None:
    # Crear instancia de DataPreprocessor
    preprocessor = DataPreprocessor(data)  # Usar el DataFrame directamente

    # Verificación de nulos y duplicados
    nulls = preprocessor.check_nulls()
    duplicates = preprocessor.check_duplicates()

    # Manejo de nulos
    if nulls.sum() > 0:
        preprocessor.data.fillna(preprocessor.data.mean(), inplace=True)
    
    duplicates = preprocessor.check_duplicates()
    if duplicates > 0:
        preprocessor.data.drop_duplicates(inplace=True)
    # Obtener estadísticas descriptivas
    preprocessor.get_descriptive_stats()

    # Preprocesar los datos
    X, y = preprocessor.preprocess_data()

    # Dividir los datos
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)

    # Escalar características
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)

    # Crear y entrenar el modelo
    model = NeuralNetworkModel(input_dim=X_train.shape[1])
    model.compile_model()
    model.train_model(X_train_scaled, y_train, epochs=100)  # Ajustar según sea necesario

    # Evaluar el modelo
    evaluator = ModelEvaluator(model=model.model, history=model.history)
    evaluator.evaluate_model(X_test_scaled, y_test)

    # Visualizar predicciones
    evaluator.plot_training_history()
    evaluator.plot_predictions(X_test_scaled, y_test)

if __name__ == "__main__":
    main()
