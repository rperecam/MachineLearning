import os
import warnings
import cloudpickle
import pandas as pd
import numpy as np
from datetime import datetime

# Silenciar advertencias
warnings.filterwarnings("ignore")

# ----- VALORES Y CONSTANTES -----

# Valores por defecto para imputación (exactamente como especifica el PDF)
DEFAULT_NIGHTS = 2
DEFAULT_GUESTS = 2
DEFAULT_RATE = 250
DEFAULT_BOARD = "SC"  # Añadido según el PDF

# Costes de servicios
COST = {
    'A': 4,  # Desayuno (por persona/día)
    'B': 0.09,  # Mejora habitación (% del precio total)
    'C': 7,  # Parking (por día)
    'D': 9  # Spa (por persona)
}

# Coste adicional de campaña por reserva impactada
CAMPAIGN_COST = 5

# Umbral óptimo del modelo
THRESHOLD = 0.4088


# ----- FUNCIONES DE INFERENCIA DE CANCELACIONES -----

def get_X():
    """Carga y preprocesa los datos para inferencia."""
    print("Cargando datos de inferencia...")

    # Cargar datos
    inference = pd.read_csv(os.environ.get("INFERENCE_DATA_PATH", "data/bookings_test.csv"))
    hotels = pd.read_csv(os.environ.get("HOTELS_DATA_PATH", "data/hotels.csv"))

    # Unir reservas con información del hotel
    data = pd.merge(inference, hotels, on="hotel_id", how="left")

    # Reemplazar 'No-Show' por 'Check-Out' si existe
    if 'reservation_status' in data.columns:
        data["reservation_status"] = data["reservation_status"].replace("No-Show", "Check-Out")

    # Convertir columnas de fecha
    for col in ["arrival_date", "booking_date", "reservation_status_date"]:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col])

    # Calcular lead_time
    if 'arrival_date' in data.columns and 'booking_date' in data.columns:
        data['lead_time'] = (data['arrival_date'] - data['booking_date']).dt.days

    # Eliminar columnas que causan data leakage para predicción
    columns_to_drop = ["reservation_status", "reservation_status_date",
                       "days_before_arrival"]
    X = data.drop(columns=[col for col in columns_to_drop if col in data.columns])

    print(f"Datos preprocesados: {X.shape[0]} registros, {X.shape[1]} características.")
    return X, data


def load_model():
    """Carga el modelo entrenado y el umbral óptimo."""
    print("Cargando el modelo entrenado...")
    model_path = os.environ.get("MODEL_PATH", "models/pipeline.cloudpkl")

    with open(model_path, "rb") as f:
        model_package = cloudpickle.load(f)

    print("Modelo cargado correctamente.")
    return model_package["pipeline"], model_package.get("threshold", THRESHOLD)


def predict_probabilities(pipeline, X):
    """Genera predicciones de probabilidad usando el pipeline."""
    print("Generando probabilidades de cancelación...")

    # Obtener probabilidades
    y_proba = pipeline.predict_proba(X)[:, 1]

    print("Probabilidades calculadas.")
    return y_proba


# ----- FUNCIONES DE ESTRATEGIA DE REGALOS -----

def impute_data(df):
    """Imputa valores faltantes con valores predefinidos según el PDF."""
    df_clean = df.copy()

    # Imputar valores específicos según el PDF
    if 'stay_nights' in df_clean.columns:
        df_clean['stay_nights'].fillna(DEFAULT_NIGHTS, inplace=True)
    if 'total_guests' in df_clean.columns:
        df_clean['total_guests'].fillna(DEFAULT_GUESTS, inplace=True)
    if 'rate' in df_clean.columns:
        df_clean['rate'].fillna(DEFAULT_RATE, inplace=True)
    if 'board' in df_clean.columns:
        df_clean['board'].fillna(DEFAULT_BOARD, inplace=True)

    # Para otras columnas: mediana para numéricas y moda para categóricas
    for col in df_clean.columns:
        if df_clean[col].isnull().any() and col not in ['stay_nights', 'total_guests', 'rate', 'board']:
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            else:
                mode_val = df_clean[col].mode()
                if not mode_val.empty:
                    df_clean[col].fillna(mode_val.iloc[0], inplace=True)

    return df_clean


def calc_replacement_rate(row):
    """
    Calcula la tasa de reemplazo según las fórmulas exactas del PDF:
    - Factor tiempo: 50% para reservas cortas de fin de semana, 35% resto
    - Factor hotel: 70% ciudad, 55% resort
    - Factor review: 20% * avg_review
    """
    arrival_date = pd.to_datetime(row['arrival_date'])
    stay_nights = row.get('stay_nights', DEFAULT_NIGHTS)
    avg_review = row.get('avg_review', 5)
    hotel_type = row.get('hotel_type', '').lower()

    # Factor tiempo: fin de semana (jueves-sábado) con estancia corta (1-3 noches)
    is_weekend = arrival_date.weekday() >= 3 and arrival_date.weekday() <= 5  # Jueves(3), Viernes(4), Sábado(5)
    is_short_stay = 1 <= stay_nights <= 3
    time_factor = 0.5 if (is_weekend and is_short_stay) else 0.35

    # Factor hotel: ciudad vs resort
    hotel_factor = 0.7 if 'city' in hotel_type else 0.55

    # Factor valoración: exactamente 20% * avg_review (limitado a 1.0 máximo)
    review_factor = min(1.0, 0.2 * avg_review)

    # La fórmula final es el producto de los tres factores
    return time_factor * hotel_factor * review_factor


def is_gift_eligible(row, gift_type):
    """Verifica si una reserva es elegible para un regalo específico según el PDF."""
    if gift_type == 'A':  # Desayuno
        return (row.get('board', '') not in ['BB', 'FB', 'HB'] and
                row.get('restaurant', 0) == 1)
    elif gift_type == 'B':  # Mejora habitación
        return row.get('total_rooms', 0) > 80
    elif gift_type == 'C':  # Parking
        return (row.get('parking', 0) == 1 and
                row.get('required_car_parking_spaces', 0) > 0)
    elif gift_type == 'D':  # Spa
        return row.get('pool_and_spa', 0) == 1
    return False


def calc_success_rate(row, gift_type):
    """Calcula la probabilidad de éxito para un regalo según las fórmulas exactas del PDF."""
    nights = row.get('stay_nights', DEFAULT_NIGHTS)
    guests = row.get('total_guests', DEFAULT_GUESTS)
    hotel_type = row.get('hotel_type', '').lower()

    if gift_type == 'A':  # Desayuno: min(1, sqrt(total_guests * stay_nights) / 4)
        return min(1.0, np.sqrt(guests * nights) / 4)
    elif gift_type == 'B':  # Mejora habitación: 80% ciudad, 65% resort
        return 0.8 if 'city' in hotel_type else 0.65
    elif gift_type == 'C':  # Parking: exactamente 50% fijo
        return 0.5
    elif gift_type == 'D':  # Spa: 70% resort, 55% ciudad (corregido según PDF)
        return 0.55 if 'city' in hotel_type else 0.7
    return 0


def calc_gift_cost(row, gift_type):
    """Calcula el coste de ofrecer un regalo específico según el PDF."""
    nights = row.get('stay_nights', DEFAULT_NIGHTS)
    guests = row.get('total_guests', DEFAULT_GUESTS)
    rate = row.get('rate', DEFAULT_RATE)

    if gift_type == 'A':  # Desayuno: 4€/persona/día
        return COST['A'] * guests * nights
    elif gift_type == 'B':  # Mejora habitación: 9% del precio total
        return COST['B'] * rate * nights
    elif gift_type == 'C':  # Parking: 7€/día
        return COST['C'] * nights
    elif gift_type == 'D':  # Spa: 9€/persona
        return COST['D'] * guests
    return 0


def select_best_gift(row):
    """
    Selecciona el mejor regalo para maximizar la facturación esperada.
    Solo para reservas con alta probabilidad de cancelación (>= umbral).
    """
    # Verificar si la reserva está en riesgo de cancelación
    cancel_prob = row.get('cancellation_probability', 0)
    if cancel_prob < THRESHOLD:
        return np.nan

    # Datos de la reserva
    rate = row.get('rate', DEFAULT_RATE)
    nights = row.get('stay_nights', DEFAULT_NIGHTS)
    reservation_value = rate * nights

    # Calcular valor esperado con reemplazo si se cancela
    replacement_rate = calc_replacement_rate(row)
    expected_if_replaced = reservation_value * replacement_rate

    # Valor esperado sin ninguna intervención
    expected_without_gift = reservation_value * (1 - cancel_prob) + expected_if_replaced * cancel_prob

    # Evaluar cada regalo posible
    best_gift = np.nan
    best_expected_value = expected_without_gift  # Valor esperado sin intervención

    for gift in ['A', 'B', 'C', 'D']:
        # Verificar elegibilidad
        if not is_gift_eligible(row, gift):
            continue

        # Calcular coste y beneficio esperado
        gift_cost = calc_gift_cost(row, gift) + CAMPAIGN_COST
        success_rate = calc_success_rate(row, gift)

        # Valor esperado con regalo = probabilidad de éxito * valor reserva - coste del regalo
        # Nota: el éxito del regalo convierte una cancelación probable en una reserva confirmada
        expected_with_gift = reservation_value * ((1 - cancel_prob) + cancel_prob * success_rate) - gift_cost

        # Seleccionar el regalo que maximice el valor esperado
        if expected_with_gift > best_expected_value:
            best_expected_value = expected_with_gift
            best_gift = gift

    return best_gift


# ----- FUNCIÓN PRINCIPAL -----

def main():
    """Función principal que ejecuta todo el flujo del proceso."""
    try:
        print(f"Iniciando proceso de inferencia y estrategia de regalos...")
        start_time = datetime.now()

        # 1. Cargar y preprocesar datos
        X, raw_data = get_X()

        # 2. Cargar modelo y obtener predicciones
        pipeline, threshold = load_model()
        probabilities = predict_probabilities(pipeline, X)

        # 3. Preparar datos para estrategia de regalos
        bookings = raw_data.copy()
        bookings = impute_data(bookings)
        bookings['cancellation_probability'] = probabilities
        bookings['prediction'] = (probabilities >= threshold).astype(int)  # Convertir a 0/1 entero

        # 4. Estadísticas básicas
        n_bookings = len(bookings)
        n_predicted_cancel = bookings['prediction'].sum()
        print(
            f"Reservas con predicción de cancelación: {n_predicted_cancel} de {n_bookings} ({n_predicted_cancel / n_bookings * 100:.1f}%)")

        # 5. Aplicar estrategia de regalos a cada reserva
        print("\nAplicando estrategia de regalos...")
        bookings['gift'] = bookings.apply(lambda row: select_best_gift(row), axis=1)

        # 6. Analizar distribución de regalos
        gift_counts = bookings['gift'].value_counts(dropna=False)
        print("\nDistribución de regalos:")
        gift_names = {
            'A': 'Desayuno',
            'B': 'Mejora habitación',
            'C': 'Parking',
            'D': 'Spa',
            np.nan: 'Sin regalo'
        }
        for gift, count in gift_counts.items():
            gift_name = gift_names.get(gift, 'Sin regalo')
            print(f"- {gift_name}: {count} ({count / n_bookings * 100:.1f}%)")

        # 7. Crear y guardar archivo de predicciones con formato solicitado
        output_dir = os.environ.get("OUTPUT_DIR", "data")
        os.makedirs(output_dir, exist_ok=True)

        # Crear el DataFrame final con las tres columnas solicitadas
        output_df = pd.DataFrame({
            'prediction': bookings['prediction'],
            'cancellation_probability': bookings['cancellation_probability'],
            'gift': bookings['gift']
        })

        # Guardar resultados en CSV
        output_predictions_path = os.path.join(output_dir, "output_predictions.csv")
        output_df.to_csv(output_predictions_path, index=False)
        print(f"\nPredicciones y asignación de regalos guardadas en {output_predictions_path}")

        # 8. Tiempo total de ejecución
        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()
        print(f"\nProceso completado en {elapsed_time:.2f} segundos")

    except Exception as e:
        print(f"Error en el proceso: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    main()