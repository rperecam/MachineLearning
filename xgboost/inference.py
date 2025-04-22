import os
import cloudpickle
import pandas as pd
import numpy as np
from sklearn import set_config

set_config(transform_output="pandas")


def get_X():
    print("Cargando datos de inferencia...")

    # Cargar datos
    inference = pd.read_csv(os.environ.get("INFERENCE_DATA_PATH", "data/bookings_train.csv"))
    hotels = pd.read_csv(os.environ.get("HOTELS_DATA_PATH", "data/hotels.csv"))

    # Unir datos
    df = inference.merge(hotels, how='left', on='hotel_id')
    df['reservation_status'] = df['reservation_status'].replace('No Show', np.nan)

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

    # Características de precio
    if 'rate' in df.columns:
        df['price_per_night'] = df['rate'] / np.maximum(df['stay_nights'], 1)
        df['price_per_person'] = df['rate'] / np.maximum(df['total_guests'], 1)

        # Limitar valores extremos
        for col in ['price_per_night', 'price_per_person']:
            if col in df.columns:
                cap = np.percentile(df[col].dropna(), 99)
                df[col] = df[col].clip(upper=cap)

    # Características de estancia
    if 'stay_nights' in df.columns:
        df['stay_duration_category'] = pd.cut(df['stay_nights'],
                                              bins=[-1, 1, 3, 7, 14, float('inf')],
                                              labels=['1_night', '2-3_nights', '4-7_nights', '8-14_nights',
                                                      '15+_nights'])

    # Características de solicitudes
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

    # Manejar valores infinitos y nulos
    for col in ['price_per_night', 'price_per_person', 'special_requests_ratio', 'booking_lead_ratio']:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    print("Datos preprocesados para inferencia.")
    return df


def get_pipeline():
    print("Cargando el pipeline entrenado...")
    model_path = os.environ.get("MODEL_PATH", "models/pipeline.cloudpkl")

    with open(model_path, mode="rb") as f:
        pipeline = cloudpickle.load(f)

    print("Pipeline cargado correctamente.")
    return pipeline


def get_predictions(pipeline, X=None):
    if X is None:
        X = get_X()

    print("Realizando predicciones...")
    try:
        # Intentar usar el pipeline directamente
        if hasattr(pipeline, 'predict'):
            # Si es CustomPipeline, asegurar que is_fitted=True
            if hasattr(pipeline, 'is_fitted'):
                pipeline.is_fitted = True
            predictions = pipeline.predict(X)
        else:
            predictions = pipeline.predict(X)
    except Exception as e:
        print(f"Error usando predict: {e}")
        try:
            # Enfoque alternativo usando componentes del pipeline
            if hasattr(pipeline, 'preprocessor') and hasattr(pipeline, 'model'):
                X_processed = pipeline.preprocessor.transform(X)
                probas = pipeline.model.predict_proba(X_processed)[:, 1]
                threshold = getattr(pipeline, 'best_threshold', 0.5)
                predictions = (probas >= threshold).astype(int)
            else:
                raise ValueError("No se pudo acceder a los componentes del pipeline")
        except Exception as e:
            print(f"Error en enfoque alternativo: {e}")
            raise ValueError("No se pudieron generar predicciones con el modelo")

    # Convertir a Series con formato correcto
    if isinstance(predictions, np.ndarray):
        predictions = pd.Series(predictions, index=X.index, name='prediction')
    else:
        predictions = predictions.rename('prediction')

    print("Predicciones realizadas con éxito.")
    return predictions


if __name__ == "__main__":
    X = get_X()
    pipe = get_pipeline()
    preds = get_predictions(pipe, X)

    # NO CAMBIAR LA RUTA DE SALIDA NI EL FORMATO. UNA ÚNICA COLUMNA CON LAS PREDICCIONES 0/1
    preds.to_csv("data/output_predictions.csv", index=False)