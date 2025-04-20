import os
import cloudpickle
import pandas as pd
import numpy as np
from sklearn import set_config

set_config(transform_output="pandas")

def get_X():

    #df = pd.read_csv(os.environ["INFERENCE_DATA_PATH"])

    # Cargar datos de entrada localmente
    df = pd.read_csv("data/bookings_train.csv")

    # Convertir columnas de fecha a datetime
    date_columns = ['arrival_date', 'booking_date', 'reservation_status_date']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    # Crear características temporales
    if 'arrival_date' in df.columns and 'booking_date' in df.columns:
        df['lead_time'] = (df['arrival_date'] - df['booking_date']).dt.days
        df['lead_time_category'] = pd.cut(df['lead_time'],
                                          bins=[-1, 7, 30, 90, 180, float('inf')],
                                          labels=['last_minute', 'short', 'medium', 'long', 'very_long'])
        df['is_high_season'] = df['arrival_date'].dt.month.isin([6, 7, 8, 12]).astype(int)
        df['is_weekend_arrival'] = df['arrival_date'].dt.dayofweek.isin([4, 5]).astype(int)
        df['arrival_month'] = df['arrival_date'].dt.month
        df['arrival_dayofweek'] = df['arrival_date'].dt.dayofweek
        df['booking_month'] = df['booking_date'].dt.month

    # Características derivadas de precio
    if 'rate' in df.columns:
        df['price_per_night'] = df['rate'] / np.maximum(df['stay_nights'], 1)
        df['price_per_person'] = df['rate'] / np.maximum(df['total_guests'], 1)

        # Limitar valores extremos
        price_cap = np.percentile(df['price_per_night'].dropna(), 99)
        df['price_per_night'] = df['price_per_night'].clip(upper=price_cap)
        price_person_cap = np.percentile(df['price_per_person'].dropna(), 99)
        df['price_per_person'] = df['price_per_person'].clip(upper=price_person_cap)

    # Características de duración de estancia
    if 'stay_nights' in df.columns:
        df['stay_duration_category'] = pd.cut(df['stay_nights'],
                                              bins=[-1, 1, 3, 7, 14, float('inf')],
                                              labels=['1_night', '2-3_nights', '4-7_nights', '8-14_nights', '15+_nights'])

    # Características de solicitudes especiales
    if 'special_requests' in df.columns:
        df['has_special_requests'] = (df['special_requests'] > 0).astype(int)
        df['special_requests_ratio'] = df['special_requests'] / np.maximum(df['total_guests'], 1)

    # Características de ubicación
    if 'country_x' in df.columns and 'country_y' in df.columns:
        df['is_foreign'] = (df['country_x'] != df['country_y']).astype(int)
        df.loc[df['country_x'].isna() | df['country_y'].isna(), 'is_foreign'] = 0

    # Características interactivas
    if 'total_guests' in df.columns and 'stay_nights' in df.columns:
        df['guest_nights'] = df['total_guests'] * df['stay_nights']
        df['booking_lead_ratio'] = df.get('lead_time', 0) / np.maximum(df['stay_nights'], 1)

    # Características de transporte
    if 'required_car_parking_spaces' in df.columns:
        df['requested_parking'] = (df['required_car_parking_spaces'] > 0).astype(int)

    return df

def get_pipeline():

    # Cargar el pipeline desde un archivo local
    with open("models/xgboost_model.pkl", mode="rb") as f:
        pipe = cloudpickle.load(f)

    #with open(os.environ["MODEL_PATH"], mode="rb") as f:
        #pipe = cloudpickle.load(f)

    return pipe

def get_predictions(pipe, X):
    """
    Realiza predicciones usando el modelo cargado.

    Args:
        pipe: Pipeline entrenada (CustomPipeline o similar)
        X: DataFrame con datos para predicción

    Returns:
        Series con predicciones 0/1
    """
    # Detectar qué tipo de pipeline estamos usando
    if hasattr(pipe, 'predict'):
        # Nuestra CustomPipeline tiene método predict
        predictions = pipe.predict(X)
    else:
        # Fallback a formas alternativas
        try:
            # Si es un pipeline estándar
            predictions = pipe.predict(X)
        except:
            try:
                # Si es un diccionario o estructura con modelo
                if hasattr(pipe, 'model'):
                    X_processed = pipe.preprocessor.transform(X)
                    predictions = pipe.model.predict(X_processed)
                else:
                    raise ValueError("Estructura de pipeline no reconocida")
            except:
                raise ValueError("No se pudieron generar predicciones con el modelo")

    # Asegurarnos de devolver Series con el formato correcto
    if isinstance(predictions, np.ndarray):
        return pd.Series(predictions, index=X.index, name='prediction')
    else:
        return predictions.rename('prediction')

if __name__ == "__main__":
    # Cargar datos
    X = get_X()

    # Cargar pipeline
    pipe = get_pipeline()

    # Obtener predicciones
    preds = get_predictions(pipe, X)

    # NO CAMBIAR LA RUTA DE SALIDA NI EL FORMATO. UNA ÚNICA COLUMNA CON LAS PREDICCIONES 0/1
    preds.to_csv("output_predictions.csv", header=True)