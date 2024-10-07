import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
from data_set import data

class DataPreprocessor:
    def __init__(self, data: Optional[pd.DataFrame] = None):
        """Inicializa la clase con un DataFrame."""
        if data is not None:
            self.data = data
        else:
            raise ValueError("Se debe proporcionar un DataFrame")

    def check_nulls(self) -> pd.Series:
        """Verifica y retorna los valores nulos en el dataset."""
        nulls = self.data.isnull().sum()
        print("\nValores nulos en el dataset:")
        print(nulls)
        return nulls

    def check_duplicates(self) -> int:
        """Verifica y retorna el número de filas duplicadas en el dataset."""
        duplicates = self.data.duplicated().sum()
        print(f"\nNúmero de filas duplicadas en el dataset: {duplicates}")
        return duplicates

    def get_descriptive_stats(self) -> pd.DataFrame:
        """Retorna estadísticas descriptivas del dataset."""
        stats = self.data.describe()
        print("\nEstadísticas descriptivas:")
        print(stats)
        return stats

    def preprocess_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Convierte variables categóricas y separa características de la variable objetivo."""
        # Verificar valores únicos antes de la conversión
        print("Valores únicos en 'Extracurricular Activities':", self.data['Extracurricular Activities'].unique())
        
        # Convertir 'Extracurricular Activities' a variable numérica
        if 'Extracurricular Activities' in self.data.columns:
            self.data['Extracurricular Activities'] = self.data['Extracurricular Activities'].map({'Sí': 1, 'No': 0}).fillna(0)
        else:
            print("Columna 'Extracurricular Activities' no encontrada.")

        # Verificamos si la conversión se realizó correctamente
        print("\nDatos después de la conversión:")
        print(self.data[['Extracurricular Activities']].value_counts())

        # Separar características y variable objetivo
        if 'Performance Index' in self.data.columns:
            X = self.data.drop('Performance Index', axis=1)  # Características
            y = self.data['Performance Index']  # Variable objetivo
        else:
            raise ValueError("Columna 'Performance Index' no encontrada en el dataset.")

        return X, y
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Divide el dataset en conjunto de entrenamiento y prueba."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"\nTamaño del conjunto de entrenamiento: {X_train.shape}")
        print(f"Tamaño del conjunto de prueba: {X_test.shape}")
        return X_train, X_test, y_train, y_test

    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Escala las características usando StandardScaler."""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

