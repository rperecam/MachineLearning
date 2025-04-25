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
    """Carga y preprocesa los datos para inferencia."""
    print("Cargando datos de inferencia...")

    # Cargar datos
    inference = pd.read_csv(os.environ.get("INFERENCE_DATA_PATH"))
    hotels = pd.read_csv(os.environ.get("HOTELS_DATA_PATH"))

    # Unir datos
    data = pd.merge(inference, hotels, on='hotel_id', how='left')

    # Filtrar solo reservas en estado 'Booked'
    data = data[data['reservation_status'] == 'Booked'].copy()

    # Convertir fechas
    date_columns = ['arrival_date', 'booking_date', 'reservation_status_date']
    for col in date_columns:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col])

    # Crear características clave
    data['lead_time'] = (data['arrival_date'] - data['booking_date']).dt.days
    data['is_high_season'] = data['arrival_date'].dt.month.isin([6, 7, 8, 12]).astype(int)
    data['is_weekend_arrival'] = data['arrival_date'].dt.dayofweek.isin([4, 5]).astype(int)
    data['price_per_night'] = data['rate'] / np.maximum(data['stay_nights'], 1)
    data['has_special_requests'] = (data['special_requests'] > 0).astype(int)

    # Característica de cliente extranjero
    if 'country_x' in data.columns and 'country_y' in data.columns:
        data['is_foreign'] = (data['country_x'].astype(str) != data['country_y'].astype(str)).astype(int)
        data.loc[data['country_x'].isna() | data['country_y'].isna(), 'is_foreign'] = 0

    # Eliminar columnas que podrían causar data leakage
    columns_to_drop = [
        'reservation_status', 'reservation_status_date',
        'arrival_date', 'booking_date', 'special_requests', 'stay_nights',
        'country_x', 'country_y'
    ]
    columns_to_drop = [col for col in columns_to_drop if col in data.columns]

    # Preparar X
    X = data.drop(columns=columns_to_drop)

    print(f"Datos preprocesados para inferencia: {X.shape[0]} registros, {X.shape[1]} características")
    return X

def get_pipeline():
    """Carga el modelo entrenado (StackingEnsemble)."""
    print("Cargando el modelo entrenado...")
    model_path = os.environ.get("MODEL_PATH")

    with open(model_path, mode="rb") as f:
        pipe = cloudpickle.load(f)

    print("Modelo cargado correctamente.")
    return pipe

def get_predictions(pipe, X=None):
    """Realiza predicciones usando el modelo cargado."""
    if X is None:
        X = get_X()

    print("Realizando predicciones...")

    try:
        predictions = pipe.predict(X)
    except Exception as e:
        print(f"Error al realizar predicciones: {e}")
        raise ValueError("No se pudieron generar predicciones con el modelo")

    # Convertir a Series con formato correcto
    predictions = pd.Series(predictions, name='prediction')

    print("Predicciones realizadas con éxito.")
    return predictions

if __name__ == "__main__":
    # Cargar datos y modelo
    X = get_X()
    pipe = get_pipeline()

    # Realizar predicciones
    predictions = get_predictions(pipe, X)

    # Guardar predicciones en el formato requerido
    output_path = "output_predictions.csv"
    predictions.to_csv(output_path, index=False)
    print(f"Predicciones guardadas en {output_path}: {predictions.sum()} cancelaciones de {len(predictions)} reservas ({predictions.mean() * 100:.1f}%)")
