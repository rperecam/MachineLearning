import os
import cloudpickle
import pandas as pd
import numpy as np
import warnings
from sklearn import set_config

# Configuración
set_config(transform_output="pandas")
warnings.filterwarnings('ignore')


def get_X():
    """Carga y preprocesa los datos para inferencia, siguiendo el mismo procesamiento del entrenamiento."""
    print("Cargando datos de inferencia...")

    # Cargar datos
    inference = pd.read_csv(os.environ.get("INFERENCE_DATA_PATH", "data/bookings_train.csv"))
    hotels = pd.read_csv(os.environ.get("HOTELS_DATA_PATH", "data/hotels.csv"))

    # Unir datos
    data = pd.merge(inference, hotels, on='hotel_id', how='left')
    #Quito los registros con reservation_status = Canceled, Check-in o No-Show para hacer inferencia solo en Booked
    data = data[data['reservation_status'].isin(['No-Show', 'Canceled', 'Check-in'])].copy()

    # Convertir fechas
    for col in ['arrival_date', 'booking_date', 'reservation_status_date']:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col])

    # Calcular días antes de llegada si están las columnas necesarias
    if 'arrival_date' in data.columns and 'reservation_status_date' in data.columns:
        data['days_before_arrival'] = (data['arrival_date'] - data['reservation_status_date']).dt.days

    # Crear características clave (igual que en entrenamiento)
    data['lead_time'] = (data['arrival_date'] - data['booking_date']).dt.days
    data['is_high_season'] = data['arrival_date'].dt.month.isin([6, 7, 8, 12]).astype(int)
    data['is_weekend_arrival'] = data['arrival_date'].dt.dayofweek.isin([4, 5]).astype(int)
    data['price_per_night'] = data['rate'] / np.maximum(data['stay_nights'], 1)
    data['stay_duration_category'] = pd.cut(data['stay_nights'],
                                            bins=[-1, 1, 3, 7, 14, float('inf')],
                                            labels=['1_night', '2-3_nights', '4-7_nights', '8-14_nights', '15+_nights'])
    data['has_special_requests'] = (data['special_requests'] > 0).astype(int)

    # Característica de cliente extranjero (importante para predecir cancelaciones)
    country_col_x = 'country_x'
    country_col_y = 'country_y'
    if country_col_x in data.columns and country_col_y in data.columns:
        data['is_foreign'] = (data[country_col_x].astype(str) != data[country_col_y].astype(str)).astype(int)
        data.loc[data[country_col_x].isna() | data[country_col_y].isna(), 'is_foreign'] = 0

    # Eliminar columnas que podrían causar data leakage
    columns_to_drop = [
        'reservation_status', 'reservation_status_date', 'days_before_arrival',
        'arrival_date', 'booking_date''special_requests', 'stay_nights', 'country_y'
    ]
    columns_to_drop = [col for col in columns_to_drop if col in data.columns]

    # Manejar valores nulos en todo el dataset
    for col in data.select_dtypes(include=['int64', 'float64']).columns:
        if data[col].isna().any():
            data[col] = data[col].fillna(data[col].median())

    for col in data.select_dtypes(include=['object', 'category']).columns:
        if col not in columns_to_drop and data[col].isna().any():
            data[col] = data[col].fillna(data[col].mode()[0])

    # Preparar X
    X = data.drop(columns=columns_to_drop)

    print(f"Datos preprocesados para inferencia: {X.shape[0]} registros, {X.shape[1]} características")
    return X


def get_pipeline():
    """Carga el modelo entrenado con ThresholdClassifier."""
    print("Cargando el modelo entrenado...")
    model_path = os.environ.get("MODEL_PATH", "models/xgboost_model.pkl")

    with open(model_path, mode="rb") as f:
        model = cloudpickle.load(f)

    print("Modelo cargado correctamente.")
    return model


class ThresholdClassifier:
    """Reimplementación de la clase ThresholdClassifier para asegurar compatibilidad."""
    def __init__(self, classifier, threshold=0.5):
        self.classifier = classifier
        self.threshold = threshold

    def predict(self, X):
        return (self.classifier.predict_proba(X)[:, 1] >= self.threshold).astype(int)

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)


def get_predictions(pipeline, X=None):
    """Realiza predicciones usando el modelo cargado."""
    if X is None:
        X = get_X()

    print("Realizando predicciones...")

    # Si tenemos hotel_id en X, eliminarlo para la predicción
    if 'hotel_id' in X.columns:
        X = X.drop(columns=['hotel_id'])

    try:
        # Usar predict del ThresholdClassifier, que ya aplica el umbral óptimo
        predictions = pipeline.predict(X)

    except Exception as e:
        print(f"Error al realizar predicciones: {e}")

        # Si el modelo no es un ThresholdClassifier pero tiene los atributos necesarios
        try:
            if hasattr(pipeline, 'classifier') and hasattr(pipeline, 'threshold'):
                probas = pipeline.classifier.predict_proba(X)[:, 1]
                predictions = (probas >= pipeline.threshold).astype(int)
            else:
                # Último recurso: usar predict_proba con umbral 0.5
                probas = pipeline.predict_proba(X)[:, 1]
                predictions = (probas >= 0.5).astype(int)
        except Exception as e2:
            print(f"Error secundario: {e2}")
            raise ValueError("No se pudieron generar predicciones con el modelo")

    # Convertir a Series con formato correcto
    if isinstance(predictions, np.ndarray):
        predictions = pd.Series(predictions, name='prediction')
    else:
        predictions = pd.Series(predictions, name='prediction')

    print("Predicciones realizadas con éxito.")
    return predictions


if __name__ == "__main__":
    X = get_X()
    pipe = get_pipeline()
    preds = get_predictions(pipe, X)

    # NO CAMBIAR LA RUTA DE SALIDA NI EL FORMATO. UNA ÚNICA COLUMNA CON LAS PREDICCIONES 0/1
    preds.to_csv("data/output_predictions.csv", index=False)