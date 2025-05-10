import os
import warnings
import cloudpickle
import pandas as pd
import numpy as np
from datetime import datetime

# Silenciar advertencias
warnings.filterwarnings("ignore")

# Valores por defecto
DEFAULT_NIGHTS = 2
DEFAULT_GUESTS = 2
DEFAULT_RATE = 250.0
DEFAULT_BOARD = "SC"
CAMPAIGN_COST_PER_OFFER = 5.0
DEFAULT_FALLBACK_THRESHOLD = 0.4912

# Costes de regalos
COST = {
    'A': 4.0,  # Desayuno (por persona/día)
    'B': 0.09,  # Mejora habitación (% del precio total)
    'C': 7.0,  # Parking (por día)
    'D': 9.0  # Spa (por persona)
}

def get_X():
    """Carga los datos para predecir cancelaciones."""
    inference_path = os.environ.get("INFERENCE_DATA_PATH", "data/bookings_test.csv")
    hotels_path = os.environ.get("HOTELS_DATA_PATH", "data/hotels.csv")

    inference_data = pd.read_csv(inference_path)
    hotels_data = pd.read_csv(hotels_path)

    data = pd.merge(inference_data, hotels_data, on="hotel_id", how="left")
    original_data_for_gifting = data.copy()

    data['reservation_status'].replace('No-Show', 'Check-Out', inplace=True)

    date_cols = ["arrival_date", "booking_date"]
    for col in date_cols:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors='coerce')

    if 'arrival_date' in data.columns and 'booking_date' in data.columns:
        data['lead_time'] = (data['arrival_date'] - data['booking_date']).dt.days

    if 'required_car_parking_spaces' in data.columns:
        data['required_car_parking_spaces'] = data['required_car_parking_spaces'].fillna(0)

    return data, original_data_for_gifting

def get_pipeline():
    """Carga el modelo y umbral para predecir cancelaciones."""
    model_path = os.environ.get("MODEL_PATH", "models/pipeline.cloudpkl")

    with open(model_path, "rb") as f:
        model_package = cloudpickle.load(f)

    pipeline = model_package["pipeline"]
    threshold = model_package.get("threshold", DEFAULT_FALLBACK_THRESHOLD)

    return pipeline, threshold

def get_predictions(pipeline, X_inference_processed):
    """Predice probabilidades de cancelación."""
    return pipeline.predict_proba(X_inference_processed)[:, 1]

def impute_booking_data_for_gifting(df_bookings):
    """Completa datos faltantes en las reservas según valores por defecto."""
    df_clean = df_bookings.copy()

    df_clean['stay_nights'] = df_clean['stay_nights'].fillna(DEFAULT_NIGHTS).astype(int)
    df_clean['total_guests'] = df_clean['total_guests'].fillna(DEFAULT_GUESTS).astype(int)
    df_clean['rate'] = df_clean['rate'].fillna(DEFAULT_RATE).astype(float)
    df_clean['board'] = df_clean['board'].fillna(DEFAULT_BOARD)

    if 'required_car_parking_spaces' in df_clean.columns:
        df_clean['required_car_parking_spaces'].fillna(0, inplace=True)

    for column in df_clean.columns:
        if df_clean[column].isnull().any():
            if column not in ['stay_nights', 'total_guests', 'rate', 'board',
                              'required_car_parking_spaces', 'arrival_date',
                              'booking_date', 'lead_time', 'hotel_id']:
                if pd.api.types.is_numeric_dtype(df_clean[column]):
                    df_clean[column].fillna(df_clean[column].mean(), inplace=True)
                elif pd.api.types.is_object_dtype(df_clean[column]):
                    mode_val = df_clean[column].mode()
                    if not mode_val.empty:
                        df_clean[column].fillna(mode_val[0], inplace=True)
                    else:
                        df_clean[column].fillna('Unknown', inplace=True)

    if 'arrival_date' in df_clean.columns:
        if not pd.api.types.is_datetime64_any_dtype(df_clean['arrival_date']):
            df_clean['arrival_date'] = pd.to_datetime(df_clean['arrival_date'], errors='coerce')
        if df_clean['arrival_date'].isnull().any():
            df_clean['arrival_date'].fillna(pd.Timestamp.now() + pd.Timedelta(days=30), inplace=True)
    else:
        df_clean['arrival_date'] = pd.Timestamp.now() + pd.Timedelta(days=30)

    return df_clean

def calculate_replacement_rate(row):
    """Calcula probabilidad de que una habitación cancelada sea reservada de nuevo."""
    arrival_date = row.get('arrival_date')
    if pd.isnull(arrival_date):
        arrival_date = pd.Timestamp.now() + pd.Timedelta(days=30)

    stay_nights = row.get('stay_nights', DEFAULT_NIGHTS)
    avg_review = row.get('avg_review', 5.0)
    hotel_type = str(row.get('hotel_type', 'unknown')).lower()

    is_weekend = arrival_date.weekday() >= 3 and arrival_date.weekday() <= 5
    is_short = 1 <= stay_nights <= 3
    time_factor = 0.50 if (is_weekend and is_short) else 0.35

    hotel_factor = 0.70 if 'City Hotel' in hotel_type else 0.55
    review_factor = min(1.0, 0.20 * avg_review)

    return time_factor * hotel_factor * review_factor

def check_gift_eligibility(row, gift_type):
    """Verifica si una reserva puede recibir un regalo específico."""
    if gift_type == 'A':
        return (row.get('board', DEFAULT_BOARD) not in ['BB', 'FB', 'HB'] and
                row.get('restaurant', 0) == True)

    elif gift_type == 'B':
        return row.get('total_rooms', 0) > 80

    elif gift_type == 'C':
        return row.get('parking', 0) == True

    elif gift_type == 'D':
        return row.get('pool_and_spa', 0) == True

    return False

def calculate_gift_success_probability(row, gift_type):
    """Calcula probabilidad de éxito de un regalo (cliente acepta y no cancela)."""
    nights = row.get('stay_nights', DEFAULT_NIGHTS)
    guests = row.get('total_guests', DEFAULT_GUESTS)
    hotel_type = str(row.get('hotel_type', 'unknown')).lower()

    if gift_type == 'A':
        return min(1.0, np.sqrt(guests * nights) / 4.0)

    elif gift_type == 'B':
        return 0.80 if 'City Hotel' in hotel_type else 0.65

    elif gift_type == 'C':
        return 0.50

    elif gift_type == 'D':
        return 0.70 if 'Resort Hotel' in hotel_type else 0.55

    return 0.0

def calculate_gift_cost(row, gift_type):
    """Calcula coste total de ofrecer un regalo."""
    nights = row.get('stay_nights', DEFAULT_NIGHTS)
    guests = row.get('total_guests', DEFAULT_GUESTS)
    rate = row.get('rate', DEFAULT_RATE)

    if gift_type == 'A':
        cost = COST['A'] * guests * nights
    elif gift_type == 'B':
        cost = COST['B'] * rate
    elif gift_type == 'C':
        cost = COST['C'] * nights
    elif gift_type == 'D':
        cost = COST['D'] * guests
    else:
        cost = 0.0

    return cost + CAMPAIGN_COST_PER_OFFER

def select_optimal_gift(row_with_proba, current_threshold):
    """Selecciona el mejor regalo para maximizar el beneficio neto esperado."""
    rate = row_with_proba.get('rate', DEFAULT_RATE)

    prob_cancel = row_with_proba['cancellation_probability']
    prob_no_cancel = 1.0 - prob_cancel

    replacement_rate = calculate_replacement_rate(row_with_proba)
    value_if_replaced = rate * replacement_rate

    expected_value_no_gift = (prob_no_cancel * rate) + (prob_cancel * value_if_replaced)

    if prob_cancel < current_threshold:
        return np.nan, 0.0, expected_value_no_gift, expected_value_no_gift

    best_gift = np.nan
    best_expected_value = expected_value_no_gift
    best_gift_cost = 0.0

    for gift_code in ['A', 'B', 'C', 'D']:
        if not check_gift_eligibility(row_with_proba, gift_code):
            continue

        gift_cost = calculate_gift_cost(row_with_proba, gift_code)
        success_prob = calculate_gift_success_probability(row_with_proba, gift_code)

        prob_no_cancel_with_gift = prob_no_cancel + (prob_cancel * success_prob)
        prob_cancel_despite_gift = prob_cancel * (1.0 - success_prob)

        gift_expected_value = (prob_no_cancel_with_gift * rate) + \
                              (prob_cancel_despite_gift * value_if_replaced) - \
                              gift_cost

        if gift_expected_value > best_expected_value:
            best_expected_value = gift_expected_value
            best_gift = gift_code
            best_gift_cost = gift_cost

    return best_gift, best_gift_cost, best_expected_value, expected_value_no_gift

def perform_economic_analysis(results_df):
    """Realiza análisis económico de la campaña de regalos."""
    total_revenue_no_campaign = results_df['ev_no_intervention'].sum()
    total_revenue_with_campaign = results_df['final_ev_with_gift_decision'].sum()
    total_gift_costs = results_df['cost_of_selected_gift'][results_df['gift'].notna()].sum()

    gifts_offered = results_df['gift'].notna().sum()
    net_impact = total_revenue_with_campaign - total_revenue_no_campaign

    print(f"\nAnálisis Económico:")
    print(f"  Total Reservas: {len(results_df)}")
    print(f"  Regalos Ofrecidos: {gifts_offered} ({gifts_offered / len(results_df) * 100:.1f}%)")
    print(f"  Facturación Esperada SIN Campaña: {total_revenue_no_campaign:,.2f} €")
    print(f"  Facturación Esperada CON Campaña: {total_revenue_with_campaign:,.2f} €")
    print(f"  Coste Total de Regalos: {total_gift_costs:,.2f} €")
    print(f"  INCREMENTO NETO: {net_impact:,.2f} €")

    if gifts_offered > 0:
        avg_cost = total_gift_costs / gifts_offered
        roi = (net_impact / total_gift_costs) if total_gift_costs > 0 else 0
        print(f"  Coste Medio por Regalo: {avg_cost:,.2f} €")
        print(f"  ROI de Campaña: {roi:.2f}")

    print("\nDistribución de regalos:")
    gift_counts = results_df['gift'].value_counts(dropna=False)
    for gift_type, count in gift_counts.items():
        gift_name = {
            'A': 'Desayuno', 'B': 'Mejora habitación', 'C': 'Parking', 'D': 'Spa',
            np.nan: 'Sin regalo (bajo riesgo o no rentable)'
        }.get(gift_type, 'Null')

        print(f"  - {gift_name}: {count} ({count / len(results_df) * 100:.1f}%)")

        if gift_type is not np.nan:
            subset = results_df[results_df['gift'] == gift_type]
            avg_prob = subset['cancellation_probability'].mean()
            avg_cost = subset['cost_of_selected_gift'].mean()
            print(f"    * Prob. media cancelación: {avg_prob:.2f}")
            print(f"    * Coste medio: {avg_cost:.2f} €")


def export_results(results_df, output_path=None):
    """Exporta resultados en formato requerido."""
    if output_path is None:
        output_path = os.getenv("OUTPUT_PATH", "data/output_predictions.csv")

    output_df = pd.DataFrame({
        'prediction': results_df['predicted_to_cancel'],
        'probability': results_df['cancellation_probability'],
        'gift': results_df['gift']
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_df.to_csv(output_path, index=False, float_format='%.4f')
    print(f"\nArchivo guardado en: {output_path}")

    return output_df

def main():
    """Ejecuta el proceso completo de predicción y asignación de regalos."""
    try:
        print(f"Iniciando proceso... ({datetime.now()})")
        start_time = datetime.now()

        X_for_pipeline, raw_data = get_X()
        pipeline, threshold = get_pipeline()
        print(f"Umbral de predicción: {threshold:.4f}")

        cancel_probs = get_predictions(pipeline, X_for_pipeline)

        bookings = impute_booking_data_for_gifting(raw_data)
        bookings['cancellation_probability'] = cancel_probs
        bookings['predicted_to_cancel'] = (cancel_probs >= threshold).astype(int)
        bookings['loaded_threshold'] = threshold

        num_bookings = len(bookings)
        num_predicted_cancel = bookings['predicted_to_cancel'].sum()
        print(f"\nEstadísticas de Predicción:")
        print(f"  Total reservas: {num_bookings}")
        print(f"  Predicción de cancelación: {num_predicted_cancel} "
              f"({num_predicted_cancel / num_bookings * 100:.1f}%)")

        print("\nSeleccionando regalos óptimos...")
        gift_results = bookings.apply(
            lambda row: select_optimal_gift(row, threshold), axis=1, result_type='expand'
        )
        gift_results.columns = ['gift', 'cost_of_selected_gift',
                                'final_ev_with_gift_decision', 'ev_no_intervention']

        results_df = pd.concat([bookings, gift_results], axis=1)

        perform_economic_analysis(results_df)

        output_df = export_results(results_df)

        elapsed_time = (datetime.now() - start_time).total_seconds()
        print(f"\nProceso completado en {elapsed_time:.2f} segundos.")

        if 'predicted_to_cancel' in results_df.columns:
            total_cancel = results_df['predicted_to_cancel'].sum()
            total_gifts = results_df['gift'].notna().sum()

            print("\nEstadísticas adicionales:")
            print(f"  Reservas con predicción de cancelación: {total_cancel}")
            print(f"  Regalos asignados: {total_gifts}")

            if total_cancel > 0:
                coverage = total_gifts / total_cancel * 100
                print(f"  Cobertura de regalos: {coverage:.1f}%")

                print("\nDistribución por probabilidad de cancelación:")
                segments = [
                    (threshold, 0.6),
                    (0.6, 0.7),
                    (0.7, 0.8),
                    (0.8, 0.9),
                    (0.9, 1.0)
                ]

                for low, high in segments:
                    segment = results_df[(results_df['cancellation_probability'] >= low) &
                                         (results_df['cancellation_probability'] < high)]
                    if len(segment) > 0:
                        gifts = segment['gift'].notna().sum()
                        print(f"  - Prob [{low:.2f}-{high:.2f}]: {len(segment)} reservas, "
                              f"{gifts} regalos ({gifts / len(segment) * 100:.1f}%)")

        return 0

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    if exit_code == 0:
        print("Ejecución finalizada con éxito.")
    else:
        print(f"Ejecución finalizada con errores (código: {exit_code}).")
