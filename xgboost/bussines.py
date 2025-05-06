import pandas as pd
import numpy as np

# Valores por defecto para imputación (fijos según el enunciado)
DEFAULT_STAY_NIGHTS = 2
DEFAULT_TOTAL_GUESTS = 2
DEFAULT_RATE = 250

# Costes de los servicios
COST_BREAKFAST = 4
COST_ROOM_UPGRADE = 0.09
COST_PARKING = 7
COST_SPA = 9
CAMPAIGN_COST = 5

# Umbral óptimo del modelo de predicción (según resultados de entrenamiento)
OPTIMAL_THRESHOLD = 0.4088


# --- FUNCIONES AUXILIARES ---
def impute_missing_values(df):
    """Imputes missing values with defined defaults or statistics."""
    df_copy = df.copy()

    # Imputar valores fijos específicos según enunciado
    if 'stay_nights' in df_copy.columns and df_copy['stay_nights'].isnull().any():
        df_copy['stay_nights'].fillna(DEFAULT_STAY_NIGHTS, inplace=True)

    if 'total_guests' in df_copy.columns and df_copy['total_guests'].isnull().any():
        df_copy['total_guests'].fillna(DEFAULT_TOTAL_GUESTS, inplace=True)

    if 'rate' in df_copy.columns and df_copy['rate'].isnull().any():
        df_copy['rate'].fillna(DEFAULT_RATE, inplace=True)

    # Para el resto de columnas, usar mediana para numéricas y moda para categóricas
    for col in df_copy.columns:
        if df_copy[col].isnull().any():
            if col not in ['stay_nights', 'total_guests', 'rate']:  # Excluir las ya imputadas
                if pd.api.types.is_numeric_dtype(df_copy[col]):
                    df_copy[col].fillna(df_copy[col].median(), inplace=True)  # Usar mediana en vez de media
                else:
                    mode_val = df_copy[col].mode()
                    if not mode_val.empty:
                        df_copy[col].fillna(mode_val.iloc[0], inplace=True)

    return df_copy


def calculate_replacement_rate(row):
    """Calcula la tasa de reemplazo según los factores especificados en el enunciado."""
    arrival_date = pd.to_datetime(row['arrival_date'])
    stay_nights = row.get('stay_nights', DEFAULT_STAY_NIGHTS)
    avg_review = row.get('avg_review', 5)
    hotel_type = row.get('hotel_type', '')

    # Factor tiempo: 50% en fin de semana con estancias cortas, 35% para el resto
    if 3 <= arrival_date.weekday() <= 5 and 1 <= stay_nights <= 3:
        time_factor = 0.5
    else:
        time_factor = 0.35

    # Factor tipo hotel: 70% en ciudad, 55% en resort
    hotel_factor = 0.7 if hotel_type == 'city' else 0.55

    # Factor valoración: 20% * avg_review (máximo 1.0)
    review_factor = min(1.0, 0.2 * avg_review)

    # La tasa final es el producto de los tres factores
    return time_factor * hotel_factor * review_factor


def is_eligible_for_gift(row, gift_type):
    """Verifica si una reserva es elegible para un tipo específico de regalo."""
    if gift_type == 'A':  # Desayuno
        return row.get('board', '') not in ['BB', 'FB', 'HB'] and row.get('restaurant', 0) == 1
    elif gift_type == 'B':  # Mejora de habitación
        return row.get('total_rooms', 0) > 80
    elif gift_type == 'C':  # Parking
        # Verificar si hay parking disponible en el hotel y si el cliente tiene coche
        return row.get('parking', 0) == 1 and row.get('required_car_parking_spaces', 0) > 0
    elif gift_type == 'D':  # Spa
        return row.get('pool_and_spa', 0) == 1
    return False


def calculate_success_rate(row, gift_type):
    """Calcula la tasa de éxito esperada para un regalo según el enunciado."""
    stay_nights = row.get('stay_nights', DEFAULT_STAY_NIGHTS)
    total_guests = row.get('total_guests', DEFAULT_TOTAL_GUESTS)
    hotel_type = row.get('hotel_type', '')

    if gift_type == 'A':  # Desayuno: proporcional a sqrt(personas*noches)/4
        return min(1, np.sqrt(total_guests * stay_nights) / 4)
    elif gift_type == 'B':  # Mejora habitación: 80% ciudad, 65% resort
        return 0.8 if hotel_type == 'city' else 0.65
    elif gift_type == 'C':  # Parking: 50% fijo
        return 0.5
    elif gift_type == 'D':  # Spa: 55% ciudad, 70% resort
        return 0.55 if hotel_type == 'city' else 0.7
    return 0


def calculate_gift_cost(row, gift_type):
    """Calcula el coste de ofrecer un regalo específico para una reserva."""
    stay_nights = row.get('stay_nights', DEFAULT_STAY_NIGHTS)
    total_guests = row.get('total_guests', DEFAULT_TOTAL_GUESTS)
    rate = row.get('rate', DEFAULT_RATE)

    if gift_type == 'A':  # Desayuno: 4€/persona/día
        return COST_BREAKFAST * total_guests * stay_nights
    elif gift_type == 'B':  # Mejora habitación: 9% del precio de la estancia
        return COST_ROOM_UPGRADE * rate * stay_nights
    elif gift_type == 'C':  # Parking: 7€/día
        return COST_PARKING * stay_nights
    elif gift_type == 'D':  # Spa: 9€/persona
        return COST_SPA * total_guests
    return 0


# --- ESTRATEGIAS OPTIMIZADAS ---

def strategy_probabilistic_optimized(row, threshold=OPTIMAL_THRESHOLD, min_roi=0.5):
    """
    Estrategia basada en la probabilidad de cancelación, optimizada para usar
    el umbral óptimo del modelo y un ROI mínimo más estricto.
    """
    # Verificar si la probabilidad de cancelación supera el umbral
    cancel_prob = row.get('cancellation_probability', 0)
    if cancel_prob < threshold:
        return np.nan

    # Para alta probabilidad de cancelación, ser más agresivo
    if cancel_prob > 0.7:
        min_roi = 0.3  # ROI mínimo más bajo para casos de alta probabilidad

    # Calcular el valor de la reserva
    rate = row.get('rate', DEFAULT_RATE)
    stay_nights = row.get('stay_nights', DEFAULT_STAY_NIGHTS)
    booking_value = rate * stay_nights

    # Priorizar reservas de alto valor
    if booking_value > 1000:
        min_roi *= 0.8  # Ser más flexible con el ROI para reservas valiosas

    # Evaluar cada tipo de regalo
    gifts = ['A', 'B', 'C', 'D']
    best_gift = np.nan
    best_roi = min_roi  # Umbral mínimo para considerar un ROI viable

    for gift in gifts:
        if is_eligible_for_gift(row, gift):
            # Calcular costes y beneficios del regalo
            gift_cost = calculate_gift_cost(row, gift) + CAMPAIGN_COST
            success_rate = calculate_success_rate(row, gift)

            # Calcular ROI: (beneficio esperado - coste) / coste
            expected_benefit = booking_value * success_rate * cancel_prob
            roi = (expected_benefit - gift_cost) / gift_cost if gift_cost > 0 else 0

            if roi > best_roi:
                best_roi = roi
                best_gift = gift

    return best_gift


def strategy_temporal_optimized(row, min_benefit_ratio=1.5):
    """
    Estrategia temporal optimizada que evalúa más cuidadosamente
    el balance entre el coste del regalo y el beneficio esperado.
    """
    # Calcular tasa de reemplazo y valor de la reserva
    replacement_rate = calculate_replacement_rate(row)

    rate = row.get('rate', DEFAULT_RATE)
    stay_nights = row.get('stay_nights', DEFAULT_STAY_NIGHTS)
    booking_value = rate * stay_nights

    # Valor esperado si se cancela y se reemplaza
    cancel_prob = row.get('cancellation_probability', 0)

    # Ignorar reservas con baja probabilidad de cancelación
    if cancel_prob < OPTIMAL_THRESHOLD * 0.8:
        return np.nan

    expected_value_if_cancelled = booking_value * replacement_rate * cancel_prob

    # Evaluar cada tipo de regalo
    gifts = ['A', 'B', 'C', 'D']
    best_gift = np.nan
    best_benefit_ratio = min_benefit_ratio  # Mínimo ratio beneficio/coste

    for gift in gifts:
        if is_eligible_for_gift(row, gift):
            gift_cost = calculate_gift_cost(row, gift) + CAMPAIGN_COST
            success_rate = calculate_success_rate(row, gift)

            # No considerar regalos demasiado costosos en relación al valor
            if gift_cost > booking_value * 0.15:  # 15% del valor de la reserva como máximo
                continue

            # Valor esperado si se ofrece el regalo
            expected_value_with_gift = booking_value * success_rate * cancel_prob

            # Beneficio neto: valor con regalo vs valor con reemplazo
            net_benefit = expected_value_with_gift - expected_value_if_cancelled

            # Calcular ratio beneficio/coste
            benefit_ratio = net_benefit / gift_cost if gift_cost > 0 else 0

            if benefit_ratio > best_benefit_ratio:
                best_benefit_ratio = benefit_ratio
                best_gift = gift

    # Casos especiales para maximizar beneficio
    if pd.isna(best_gift) and cancel_prob > 0.75 and booking_value > 800:
        # Para reservas de alto valor con alta probabilidad de cancelación
        # ser más agresivo incluso con ratios de beneficio más bajos
        for gift in gifts:
            if is_eligible_for_gift(row, gift):
                success_rate = calculate_success_rate(row, gift)
                if success_rate > 0.6:  # Si es un regalo con alta tasa de éxito
                    return gift

    return best_gift


def strategy_hybrid_improved(row, prob_threshold=OPTIMAL_THRESHOLD * 0.85, base_roi_threshold=3.0):
    """
    Estrategia híbrida mejorada que combina elementos de todas las estrategias anteriores
    y añade segmentación por valor de reserva y tipo de hotel.
    """
    # Obtener datos principales
    cancel_prob = row.get('cancellation_probability', 0)
    rate = row.get('rate', DEFAULT_RATE)
    stay_nights = row.get('stay_nights', DEFAULT_STAY_NIGHTS)
    booking_value = rate * stay_nights
    hotel_type = row.get('hotel_type', '')
    total_guests = row.get('total_guests', DEFAULT_TOTAL_GUESTS)

    # Filtrar reservas con baja probabilidad de cancelación
    if cancel_prob < prob_threshold:
        return np.nan

    # Segmentación por valor de reserva
    if booking_value > 1500:
        segment = "premium"
        roi_threshold = base_roi_threshold * 0.7  # Más permisivo con reservas premium
    elif booking_value > 800:
        segment = "high"
        roi_threshold = base_roi_threshold * 0.85
    elif booking_value > 400:
        segment = "medium"
        roi_threshold = base_roi_threshold * 1.0
    else:
        segment = "low"
        roi_threshold = base_roi_threshold * 1.2  # Más estricto con reservas económicas

    # Cálculo de tasa de reemplazo
    replacement_rate = calculate_replacement_rate(row)
    expected_value_if_cancelled = booking_value * replacement_rate * cancel_prob

    # Variable para registrar el mejor regalo y sus métricas
    best_gift = np.nan
    best_score = 0

    # Evaluar cada tipo de regalo con sistema de puntuación
    gifts = ['A', 'B', 'C', 'D']
    for gift in gifts:
        if is_eligible_for_gift(row, gift):
            gift_cost = calculate_gift_cost(row, gift) + CAMPAIGN_COST
            success_rate = calculate_success_rate(row, gift)

            # Evitar regalos demasiado costosos
            if gift_cost > booking_value * 0.18:  # Límite de 18% del valor de reserva
                continue

            # Valor esperado con el regalo
            expected_value_with_gift = booking_value * success_rate * cancel_prob

            # Beneficio neto vs reemplazo
            net_benefit = expected_value_with_gift - expected_value_if_cancelled
            roi = net_benefit / gift_cost if gift_cost > 0 else 0

            # Sistema de puntuación ponderado
            # (combina ROI, valor absoluto del beneficio y factores específicos)
            score = (roi * 2)  # Base: doble peso al ROI

            # Factores adicionales según segmento y tipo de hotel
            if segment == "premium":
                # Para reservas premium, valorar más el éxito que el coste
                score += success_rate * 2.5
            elif segment == "low":
                # Para reservas económicas, valorar más la eficiencia del coste
                score += (10 / gift_cost) if gift_cost > 0 else 0

            # Bonificaciones específicas por tipo de hotel
            if hotel_type == 'city':
                if gift == 'B':  # Mejora de habitación tiene mejor rendimiento en ciudad
                    score *= 1.15
                elif gift == 'C' and row.get('required_car_parking_spaces', 0) > 1:
                    # Parking más valorado si tienen varios coches
                    score *= 1.2
            else:  # Resort
                if gift == 'D':  # Spa tiene mejor rendimiento en resort
                    score *= 1.2
                if gift == 'A' and total_guests > 2:
                    # Desayuno más valorado para grupos grandes
                    score *= 1.15

            # Penalización por alta relación coste/beneficio
            cost_benefit_ratio = gift_cost / net_benefit if net_benefit > 0 else float('inf')
            if cost_benefit_ratio > 0.5:
                score *= (0.5 / cost_benefit_ratio) if cost_benefit_ratio > 0 else 1

            # Actualizar mejor regalo si mejora la puntuación
            if score > best_score and roi > roi_threshold:
                best_score = score
                best_gift = gift

    # Casos especiales para maximizar la retención
    if pd.isna(best_gift) and cancel_prob > 0.8 and segment in ["premium", "high"]:
        # Ultimo recurso para reservas valiosas con alta probabilidad de cancelación
        for gift in gifts:
            if is_eligible_for_gift(row, gift):
                success_rate = calculate_success_rate(row, gift)
                if success_rate > 0.65:  # Regalo con alta probabilidad de éxito
                    return gift

    return best_gift


# --- EVALUACIÓN ---
def evaluate_strategy(df, strategy_func, **kwargs):
    """
    Evalúa el rendimiento económico de una estrategia de retención.
    Calcula ingresos con y sin campaña, coste total y beneficio neto.
    """
    df_eval = df.copy()

    # Aplicar la estrategia a todas las reservas
    df_eval['gift'] = df_eval.apply(lambda row: strategy_func(row, **kwargs), axis=1)

    # Contadores e inicialización
    total_revenue_without = 0
    total_revenue_with = 0
    total_cost = 0
    total_gifts = 0

    # Evaluar cada reserva
    for _, row in df_eval.iterrows():
        rate = row.get('rate', DEFAULT_RATE)
        stay_nights = row.get('stay_nights', DEFAULT_STAY_NIGHTS)
        cancel_prob = row.get('cancellation_probability', 0)

        # Valor total de la reserva
        booking_value = rate * stay_nights

        # Tasa de reemplazo si se cancela
        replacement_rate = calculate_replacement_rate(row)

        # Ingresos esperados sin campaña
        expected_without = (1 - cancel_prob) * booking_value + cancel_prob * replacement_rate * booking_value
        total_revenue_without += expected_without

        # Evaluar el impacto del regalo si se asignó uno
        gift = row['gift']
        if pd.notna(gift):
            total_gifts += 1

            # Coste del regalo
            gift_cost = calculate_gift_cost(row, gift) + CAMPAIGN_COST
            total_cost += gift_cost

            # Tasa de éxito del regalo
            success_rate = calculate_success_rate(row, gift)

            # Ingresos esperados con campaña
            expected_with = (1 - cancel_prob) * booking_value + \
                            cancel_prob * (success_rate * booking_value + \
                                           (1 - success_rate) * replacement_rate * booking_value)

            total_revenue_with += expected_with
        else:
            # Sin regalo, los ingresos son iguales a los de sin campaña
            total_revenue_with += expected_without

    # Calcular métricas de rendimiento
    net_benefit = total_revenue_with - total_revenue_without - total_cost
    roi = net_benefit / total_cost if total_cost > 0 else 0

    # Mostrar resultados
    print(f"\n--- Evaluación de la estrategia ---")
    print(f"Ingresos sin campaña: {total_revenue_without:.2f}€")
    print(f"Ingresos con campaña: {total_revenue_with:.2f}€")
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

    # Convertir a datetime
    bookings['arrival_date'] = pd.to_datetime(bookings['arrival_date'])
    if 'booking_date' in bookings.columns:
        bookings['booking_date'] = pd.to_datetime(bookings['booking_date'])

    # Imputar valores nulos según requisitos
    bookings = impute_missing_values(bookings)

    # Uno el df de bookings con el de predicciones por el indice
    bookings = bookings.merge(predictions, left_index=True, right_index=True)

    # Añadir predicción binaria basada en el umbral óptimo
    bookings['prediction'] = (bookings['predicted_cancellation'] >= OPTIMAL_THRESHOLD)
    bookings['cancellation_probability'] = bookings['predicted_cancellation']

    print(f"Reservas con predicción positiva: {bookings['prediction'].sum()} de {len(bookings)} reservas")

    # Evaluar cada estrategia
    print("\nEvaluando estrategias optimizadas...\n")

    # Estrategia 1: Probabilística optimizada
    eval_prob = evaluate_strategy(bookings, strategy_probabilistic_optimized)

    # Estrategia 2: Temporal optimizada
    eval_temp = evaluate_strategy(bookings, strategy_temporal_optimized)

    # Estrategia 3: Híbrida mejorada
    eval_hybrid = evaluate_strategy(bookings, strategy_hybrid_improved)

    # Comparar resultados
    strategies = {
        "Probabilística Optimizada": eval_prob,
        "Temporal Optimizada": eval_temp,
        "Híbrida Mejorada": eval_hybrid
    }

    print("\nComparación de estrategias optimizadas:")
    for name, results in strategies.items():
        print(f"- {name}: Beneficio = {results['net_benefit']:.2f}€, ROI = {results['ROI']:.2f}")

    # Seleccionar la mejor estrategia
    best_strategy = max(strategies.items(), key=lambda x: x[1]['net_benefit'])
    print(f"\nMejor estrategia: {best_strategy[0]} con beneficio neto de {best_strategy[1]['net_benefit']:.2f}€")

    # Aplicar la mejor estrategia
    if best_strategy[0] == "Probabilística Optimizada":
        bookings['final_gift'] = bookings.apply(lambda row: strategy_probabilistic_optimized(row), axis=1)
    elif best_strategy[0] == "Temporal Optimizada":
        bookings['final_gift'] = bookings.apply(lambda row: strategy_temporal_optimized(row), axis=1)
    else:  # Híbrida Mejorada
        bookings['final_gift'] = bookings.apply(lambda row: strategy_hybrid_improved(row), axis=1)

    # Guardar resultados
    try:
        bookings[['final_gift']].to_csv("data/gift_predictions.csv", index=False, header=False)
        print("\nResultados guardados en data/gift_predictions.csv")
    except Exception as e:
        print(f"Error al guardar resultados: {e}")


if __name__ == "__main__":
    main()