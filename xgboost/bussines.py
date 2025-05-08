import pandas as pd
import numpy as np

# Valores por defecto para imputación
DEFAULT_NIGHTS = 2
DEFAULT_GUESTS = 2
DEFAULT_RATE = 250

# Costes de servicios
COST = {
    'A': 4,  # Desayuno (por persona/día)
    'B': 0.09,  # Mejora habitación (% del precio total)
    'C': 7,  # Parking (por día)
    'D': 9  # Spa (por persona)
}
CAMPAIGN_COST = 5

# Umbral óptimo del modelo
THRESHOLD = 0.4088


# --- FUNCIONES DE PROCESAMIENTO DE DATOS ---
def impute_data(df):
    """Imputa valores faltantes con valores predefinidos o estadísticos."""
    df_clean = df.copy()

    # Imputar valores específicos según enunciado
    if 'stay_nights' in df_clean.columns:
        df_clean['stay_nights'].fillna(DEFAULT_NIGHTS, inplace=True)

    if 'total_guests' in df_clean.columns:
        df_clean['total_guests'].fillna(DEFAULT_GUESTS, inplace=True)

    if 'rate' in df_clean.columns:
        df_clean['rate'].fillna(DEFAULT_RATE, inplace=True)

    # Para otras columnas: mediana para numéricas y moda para categóricas
    for col in df_clean.columns:
        if df_clean[col].isnull().any() and col not in ['stay_nights', 'total_guests', 'rate']:
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            else:
                mode_val = df_clean[col].mode()
                if not mode_val.empty:
                    df_clean[col].fillna(mode_val.iloc[0], inplace=True)

    return df_clean


# --- FUNCIONES DE ANÁLISIS ---
def calc_replacement_rate(row):
    """Calcula la tasa de reemplazo si la reserva se cancela."""
    arrival_date = pd.to_datetime(row['arrival_date'])
    stay_nights = row.get('stay_nights', DEFAULT_NIGHTS)
    avg_review = row.get('avg_review', 5)
    hotel_type = row.get('hotel_type', '').lower()

    # Factor tiempo: fin de semana (jue-sáb) con estancia corta (1-3 noches)
    is_weekend = 3 <= arrival_date.weekday() <= 5
    is_short_stay = 1 <= stay_nights <= 3
    time_factor = 0.5 if (is_weekend and is_short_stay) else 0.35

    # Factor hotel: ciudad vs resort
    hotel_factor = 0.7 if 'city' in hotel_type else 0.55

    # Factor valoración: 20% * puntuación (máx 1.0)
    review_factor = min(1.0, 0.2 * avg_review)

    return time_factor * hotel_factor * review_factor


def is_gift_eligible(row, gift_type):
    """Verifica si una reserva es elegible para un regalo específico."""
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
    """Calcula la probabilidad de éxito para un regalo específico."""
    nights = row.get('stay_nights', DEFAULT_NIGHTS)
    guests = row.get('total_guests', DEFAULT_GUESTS)
    hotel_type = row.get('hotel_type', '').lower()

    if gift_type == 'A':  # Desayuno: proporcional a raíz(personas*noches)/4
        return min(1.0, np.sqrt(guests * nights) / 4)

    elif gift_type == 'B':  # Mejora habitación: 80% ciudad, 65% resort
        return 0.8 if 'city' in hotel_type else 0.65

    elif gift_type == 'C':  # Parking: 50% fijo
        return 0.5

    elif gift_type == 'D':  # Spa: 55% ciudad, 70% resort
        return 0.55 if 'city' in hotel_type else 0.7

    return 0


def calc_gift_cost(row, gift_type):
    """Calcula el coste de ofrecer un regalo específico."""
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


# --- ESTRATEGIA DE RETENCIÓN ---
def select_best_gift(row, prob_threshold=0.43, roi_min=2.5):
    """
    Selecciona el mejor regalo para una reserva con riesgo de cancelación.
    Combina análisis de valor, ROI y comparación con tasa de reemplazo.
    """
    # Verificar si la reserva está en riesgo de cancelar
    cancel_prob = row.get('cancellation_probability', 0)
    if cancel_prob < prob_threshold:
        return np.nan

    # Datos de la reserva
    rate = row.get('rate', DEFAULT_RATE)
    nights = row.get('stay_nights', DEFAULT_NIGHTS)
    booking_value = rate * nights

    # Ajustar umbral ROI según valor de la reserva
    if booking_value > 1500:
        roi_min *= 0.85
    elif booking_value < 300:
        roi_min *= 1.2

    # Calcular tasa de reemplazo y valor esperado si se cancela y reemplaza
    replacement_rate = calc_replacement_rate(row)
    expected_if_replaced = booking_value * replacement_rate * cancel_prob

    # Evaluar cada regalo posible
    best_gift = np.nan
    best_roi = 0

    for gift in ['A', 'B', 'C', 'D']:
        # Verificar elegibilidad
        if not is_gift_eligible(row, gift):
            continue

        # Calcular coste y beneficio esperado
        cost = calc_gift_cost(row, gift) + CAMPAIGN_COST
        success_rate = calc_success_rate(row, gift)
        expected_with_gift = booking_value * success_rate * cancel_prob

        # Calcular beneficio neto y ROI
        net_benefit = expected_with_gift - expected_if_replaced
        roi = net_benefit / cost if cost > 0 else 0

        # Seleccionar el regalo con mejor ROI que supere el mínimo
        if roi > roi_min and expected_with_gift > expected_if_replaced:
            if roi > best_roi:
                best_roi = roi
                best_gift = gift

    return best_gift


# --- EVALUACIÓN DE ESTRATEGIA ---
def evaluate_strategy(df, strategy_func, **params):
    """
    Evalúa el rendimiento económico de una estrategia de retención.
    Calcula ingresos con/sin campaña, coste y beneficio neto.
    """
    df_eval = df.copy()

    # Aplicar estrategia a cada reserva
    df_eval['gift'] = df_eval.apply(lambda row: strategy_func(row, **params), axis=1)

    # Contadores
    revenue_without = 0  # Ingresos sin campaña
    revenue_with = 0  # Ingresos con campaña
    total_cost = 0  # Coste total de campaña
    total_gifts = 0  # Número de regalos asignados

    # Evaluar cada reserva
    for _, row in df_eval.iterrows():
        rate = row.get('rate', DEFAULT_RATE)
        nights = row.get('stay_nights', DEFAULT_NIGHTS)
        cancel_prob = row.get('cancellation_probability', 0)
        booking_value = rate * nights

        # Calcular tasa de reemplazo si cancela
        replacement_rate = calc_replacement_rate(row)

        # Ingresos esperados sin campaña
        expected_without = ((1 - cancel_prob) * booking_value +
                            cancel_prob * replacement_rate * booking_value)
        revenue_without += expected_without

        # Evaluar impacto del regalo si se asignó
        gift = row['gift']
        if pd.notna(gift):
            total_gifts += 1

            # Coste del regalo y campaña
            gift_cost = calc_gift_cost(row, gift) + CAMPAIGN_COST
            total_cost += gift_cost

            # Probabilidad de éxito del regalo
            success_rate = calc_success_rate(row, gift)

            # Ingresos esperados con campaña
            expected_with = ((1 - cancel_prob) * booking_value +
                             cancel_prob * (success_rate * booking_value +
                                            (1 - success_rate) * replacement_rate * booking_value))
            revenue_with += expected_with
        else:
            # Sin regalo, ingresos iguales a sin campaña
            revenue_with += expected_without

    # Calcular métricas finales
    net_benefit = revenue_with - revenue_without - total_cost
    roi = net_benefit / total_cost if total_cost > 0 else 0

    # Mostrar resultados
    print(f"\n--- Evaluación de la Estrategia ---")
    print(f"Ingresos sin campaña: {revenue_without:.2f}€")
    print(f"Ingresos con campaña: {revenue_with:.2f}€")
    print(f"Coste total: {total_cost:.2f}€")
    print(f"Regalos asignados: {total_gifts}")
    print(f"Beneficio neto: {net_benefit:.2f}€")
    print(f"ROI: {roi:.2f}")

    return {
        'net_benefit': net_benefit,
        'ROI': roi,
        'total_cost': total_cost,
        'total_gifts': total_gifts
    }


# --- FUNCIÓN PRINCIPAL ---
def main():
    # Cargar datos
    try:
        bookings = pd.read_csv("data/bookings_test.csv")
        predictions = pd.read_csv("data/output_predictions.csv")
        hotels = pd.read_csv("data/hotels.csv")
    except FileNotFoundError as e:
        print(f"Error al cargar datos: {e}")
        return

    # Unir datos de reservas y hoteles
    bookings = bookings.merge(hotels, on='hotel_id', how='left')

    # Convertir fechas a datetime
    bookings['arrival_date'] = pd.to_datetime(bookings['arrival_date'])
    if 'booking_date' in bookings.columns:
        bookings['booking_date'] = pd.to_datetime(bookings['booking_date'])

    # Imputar valores nulos
    bookings = impute_data(bookings)

    # Unir con predicciones
    bookings = bookings.merge(predictions, left_index=True, right_index=True)

    # Añadir predicción binaria y probabilidad
    bookings['prediction'] = (bookings['predicted_cancellation'] >= THRESHOLD)
    bookings['cancellation_probability'] = bookings['predicted_cancellation']

    print(f"Reservas con predicción de cancelación: {bookings['prediction'].sum()} de {len(bookings)}")

    # Evaluar estrategia
    print("\nEvaluando estrategia de retención...\n")
    results = evaluate_strategy(bookings, select_best_gift)

    print(f"\nResultados finales: Beneficio = {results['net_benefit']:.2f}€, ROI = {results['ROI']:.2f}")

    # Aplicar estrategia a todas las reservas
    bookings['final_gift'] = bookings.apply(lambda row: select_best_gift(row), axis=1)

    # Guardar resultados
    try:
        bookings[['final_gift']].to_csv("data/gift_predictions.csv", index=False, header=False)
        print("\nResultados guardados en data/gift_predictions.csv")
    except Exception as e:
        print(f"Error al guardar resultados: {e}")


if __name__ == "__main__":
    main()