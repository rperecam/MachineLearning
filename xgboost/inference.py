import os
import cloudpickle
import pandas as pd
import numpy as np
from sklearn import set_config

set_config(transform_output="pandas")

def get_X():
    """
    Carga y preprocesa los datos para inferencia.

    Returns:
        DataFrame con características preparadas
    """
    print("Cargando datos de inferencia...")
    # Usar las variables de entorno para cargar los datos
    df = pd.read_csv(os.environ.get("INFERENCE_DATA_PATH", "data/bookings_train.csv"))
    hotels = pd.read_csv(os.environ.get("HOTELS_DATA_PATH", "data/hotels.csv"))

    # Combinar datos
    merged = pd.merge(df, hotels, on='hotel_id', how='left')
    df = merged.copy()

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

    # Manejar valores infinitos y nulos en características numéricas
    for col in ['price_per_night', 'price_per_person', 'special_requests_ratio', 'booking_lead_ratio']:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    print("Datos preprocesados para inferencia.")
    return df

def get_pipeline():
    """
    Carga el pipeline entrenado desde la ruta definida en variables de entorno.

    Returns:
        Pipeline entrenada
    """
    print("Cargando el pipeline entrenado...")
    # Usar la variable de entorno para la ruta del modelo
    model_path = os.environ.get("MODEL_PATH", "models/xgboost_model.pkl")

    with open(model_path, mode="rb") as f:
        pipe = cloudpickle.load(f)

    print("Pipeline cargado.")
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
    print("Realizando predicciones...")
    try:
        # Intentar usar el pipeline directamente
        if hasattr(pipe, 'predict'):
            # Si es CustomPipeline, forzar is_fitted=True si es necesario
            if hasattr(pipe, 'is_fitted'):
                pipe.is_fitted = True
            predictions = pipe.predict(X)
        else:
            # Si es un pipeline estándar
            predictions = pipe.predict(X)
    except Exception as e:
        print(f"Error usando el método predict: {e}")
        try:
            # Intento alternativo con preprocessor y model directamente
            if hasattr(pipe, 'preprocessor') and hasattr(pipe, 'model'):
                X_processed = pipe.preprocessor.transform(X)
                probas = pipe.model.predict_proba(X_processed)[:, 1]
                threshold = getattr(pipe, 'best_threshold', 0.5)
                predictions = (probas >= threshold).astype(int)
            else:
                raise ValueError("No se pudo acceder a los componentes del pipeline")
        except Exception as e:
            print(f"Error en aproximación alternativa: {e}")
            raise ValueError("No se pudieron generar predicciones con el modelo")

    # Asegurarnos de devolver Series con el formato correcto
    if isinstance(predictions, np.ndarray):
        predictions = pd.Series(predictions, index=X.index, name='prediction')
    else:
        predictions = predictions.rename('prediction')

    print("Predicciones realizadas.")
    return predictions

if __name__ == "__main__":
    # Cargar datos
    X = get_X()

    # Cargar pipeline
    pipe = get_pipeline()

    # Obtener predicciones
    preds = get_predictions(pipe, X)

    # NO CAMBIAR LA RUTA DE SALIDA NI EL FORMATO. UNA ÚNICA COLUMNA CON LAS PREDICCIONES 0/1
    print("Guardando predicciones en data/output_predictions.csv...")
    preds.to_csv("data/output_predictions.csv", header=True)
    print("Predicciones guardadas.")
