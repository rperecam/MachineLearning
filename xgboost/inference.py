import os
import warnings
import cloudpickle
import pandas as pd
from sklearn import set_config

# Configuración global
set_config(transform_output="pandas")
warnings.filterwarnings("ignore")


def get_X():
    """Carga y preprocesa los datos para inferencia."""
    print("Cargando datos de inferencia...")

    # Cargar datos
    inference = pd.read_csv(os.environ.get("INFERENCE_DATA_PATH", "data/bookings_test.csv"))
    hotels = pd.read_csv(os.environ.get("HOTELS_DATA_PATH", "data/hotels.csv"))

    # Unir reservas con información del hotel
    data = pd.merge(inference, hotels, on="hotel_id", how="left")

    # Reemplazar 'No-Show' por 'Check-Out'
    if 'reservation_status' in data.columns:
        data["reservation_status"] = data["reservation_status"].replace("No-Show", "Check-Out")

    # Convertir columnas de fecha
    for col in ["arrival_date", "booking_date", "reservation_status_date"]:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col])

    # Calcular lead_time (coherente con boost_train.py)
    if 'arrival_date' in data.columns and 'booking_date' in data.columns:
        data['lead_time'] = (data['arrival_date'] - data['booking_date']).dt.days

    # Eliminar columnas que causan data leakage
    columns_to_drop = ["reservation_status", "reservation_status_date",
                       "days_before_arrival", "arrival_date", "booking_date"]
    X = data.drop(columns=[col for col in columns_to_drop if col in data.columns])

    print(f"Datos preprocesados: {X.shape[0]} registros, {X.shape[1]} características.")
    return X, data['reservation_id'] if 'reservation_id' in data.columns else None


def load_model():
    """Carga el modelo entrenado y el umbral óptimo."""
    print("Cargando el modelo entrenado...")
    model_path = os.environ.get("MODEL_PATH", "models/pipeline.cloudpkl")

    with open(model_path, "rb") as f:
        model_package = cloudpickle.load(f)

    print("Modelo cargado correctamente.")
    return model_package["pipeline"], model_package["threshold"]


def predict_probabilities(pipeline, X):
    """Genera predicciones de probabilidad usando el pipeline."""
    print("Generando probabilidades de cancelación...")

    # Obtener probabilidades
    y_proba = pipeline.predict_proba(X)[:, 1]

    print("Probabilidades calculadas.")
    return y_proba


def main():
    # Obtener datos y modelo
    X, reservation_ids = get_X()
    pipeline, threshold = load_model()

    # Generar probabilidades
    probabilities = predict_probabilities(pipeline, X)

    # Crear DataFrame con resultados
    results = pd.DataFrame()

    # Incluir ID de reserva si está disponible
    if reservation_ids is not None:
        results['reservation_id'] = reservation_ids

    # Añadir probabilidades
    results['cancellation_probability'] = probabilities

    # Añadir predicción binaria como referencia (opcional)
    results['predicted_cancellation'] = (probabilities >= threshold).astype(int)

    # Guardar resultados
    output_path = os.environ.get("OUTPUT_PATH", "data/output_predictions.csv")
    results.to_csv(output_path, index=False)

    print(f"Probabilidades guardadas en {output_path}:")
    print(f"→ Probabilidad media de cancelación: {probabilities.mean() * 100:.2f}%")
    print(f"→ {(probabilities >= threshold).sum()} cancelaciones previstas de {len(probabilities)} reservas "
          f"({(probabilities >= threshold).mean() * 100:.1f}% usando umbral {threshold:.2f})")


if __name__ == "__main__":
    main()