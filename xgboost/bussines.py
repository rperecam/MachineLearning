import pandas as pd
import numpy as np

# Valores por defecto
DEFAULT_STAY_NIGHTS = 2
DEFAULT_TOTAL_GUESTS = 2
DEFAULT_RATE = 250

# Costes de los servicios
COST_BREAKFAST = 4
COST_ROOM_UPGRADE = 0.09
COST_PARKING = 7
COST_SPA = 9
CAMPAIGN_COST = 5


# --- FUNCIONES AUXILIARES ---
def impute_missing_values(df):
    """Imputes missing numerical values with the mean and categorical with the mode."""
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                # Use mode for categorical data, handle potential multiple modes
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col].fillna(mode_val.iloc[0], inplace=True)
    return df

def calculate_replacement_rate(row):
    """Calculates the replacement rate based on booking characteristics."""
    # Ensure arrival_date is datetime object
    arrival_date = pd.to_datetime(row['arrival_date'])
    stay_nights = row.get('stay_nights', DEFAULT_STAY_NIGHTS)
    avg_review = row.get('avg_review', 5)
    hotel_type = row.get('hotel_type', '') # Correctly get hotel_type

    # Factor tiempo
    # Check if weekday is Thursday (3), Friday (4), or Saturday (5) AND stay_nights is 1, 2, or 3
    if 3 <= arrival_date.weekday() <= 5 and 1 <= stay_nights <= 3:
        time_factor = 0.5
    else:
        time_factor = 0.35

    # Factor tipo de hotel (assuming 'country_y' is city based on original code comment)
    hotel_factor = 0.7 if hotel_type == 'country_y' else 0.55

    # Factor review
    review_factor = min(1.0, 0.2 * avg_review)

    return time_factor * hotel_factor * review_factor


def is_eligible_for_gift(row, gift_type):
    """Checks if a booking is eligible for a specific gift type."""
    if gift_type == 'A':
        # Eligible if board is not BB, FB, HB and hotel has a restaurant
        return row.get('board', '') not in ['BB', 'FB', 'HB'] and row.get('restaurant', 0) == 1
    elif gift_type == 'B':
        # Eligible if hotel has more than 80 rooms
        return row.get('total_rooms', 0) > 80
    elif gift_type == 'C':
        # Eligible if hotel has parking (and implicitly if client requested parking,
        # though the PDF says "Regalar el parking a aquellos clientes que ya lo hayan solicitado"
        # the code only checks if the hotel has parking. Sticking to code logic for now).
        return row.get('parking', 0) == 1
    elif gift_type == 'D':
        # Eligible if hotel has pool and spa
        return row.get('pool_and_spa', 0) == 1
    return False


def calculate_success_rate(row, gift_type):
    """Calculates the expected success rate for a given gift type and booking."""
    stay_nights = row.get('stay_nights', DEFAULT_STAY_NIGHTS)
    total_guests = row.get('total_guests', DEFAULT_TOTAL_GUESTS)
    hotel_type = row.get('hotel_type', '') # Correctly get hotel_type

    if gift_type == 'A':
        # Success rate proportional to sqrt(total_guests * stay_nights), capped at 1
        return min(1, np.sqrt(total_guests * stay_nights) / 4)
    elif gift_type == 'B':
        # Success rate depends on hotel type (assuming 'country_y' is city)
        return 0.8 if hotel_type == 'country_y' else 0.65
    elif gift_type == 'C':
        # Fixed success rate for parking
        return 0.5
    elif gift_type == 'D':
        # Success rate depends on hotel type (assuming 'country_y' is city, so not 'country_y' is resort)
        return 0.55 if hotel_type == 'country_y' else 0.7
    return 0


def calculate_gift_cost(row, gift_type):
    """Calculates the cost of offering a specific gift type for a booking."""
    stay_nights = row.get('stay_nights', DEFAULT_STAY_NIGHTS)
    total_guests = row.get('total_guests', DEFAULT_TOTAL_GUESTS)
    rate = row.get('rate', DEFAULT_RATE)

    if gift_type == 'A':
        return COST_BREAKFAST * total_guests * stay_nights
    elif gift_type == 'B':
        return COST_ROOM_UPGRADE * rate * stay_nights
    elif gift_type == 'C':
        return COST_PARKING * stay_nights
    elif gift_type == 'D':
        return COST_SPA * total_guests
    return 0


# --- ESTRATEGIAS ---

def strategy_1(row):
    """Strategy 1: Prioritizes A for short stays in country_y, then B, C, D."""
    hotel_type = row.get('hotel_type', '')
    stay_nights = row.get('stay_nights', DEFAULT_STAY_NIGHTS)

    if hotel_type == 'country_y' and stay_nights <= 3 and is_eligible_for_gift(row, 'A'):
        return 'A'
    elif is_eligible_for_gift(row, 'B'):
        return 'B'
    elif is_eligible_for_gift(row, 'C'):
        return 'C'
    elif is_eligible_for_gift(row, 'D'):
        return 'D'
    return np.nan


def strategy_2(row):
    """Strategy 2: Prioritizes B for low rates, then A, D, C."""
    rate = row.get('rate', DEFAULT_RATE)

    if rate < 200 and is_eligible_for_gift(row, 'B'):
        return 'B'
    elif is_eligible_for_gift(row, 'A'):
        return 'A'
    elif is_eligible_for_gift(row, 'D'):
        return 'D'
    elif is_eligible_for_gift(row, 'C'):
        return 'C'
    return np.nan


def strategy_3(row):
    """Strategy 3: Selects the eligible gift with the highest ROI."""
    gifts = ['A', 'B', 'C', 'D']
    best_gift = np.nan
    best_roi = -1 # Initialize with a value lower than any possible ROI

    for gift in gifts:
        if is_eligible_for_gift(row, gift):
            success_rate = calculate_success_rate(row, gift)
            cost = calculate_gift_cost(row, gift)
            # Avoid division by zero for cost
            if cost > 0:
                # Calculate ROI for this specific gift and booking
                # The PDF defines success_rate as a percentage, but the code uses it as a factor (0 to 1).
                # The ROI calculation in evaluate_strategy is (Net Benefit / Total Cost).
                # Here, to choose the best gift for a single row, we can use success_rate / cost as a proxy for potential ROI.
                roi = success_rate / cost
                if roi > best_roi:
                    best_roi = roi
                    best_gift = gift
            # Handle case where cost is 0 but success rate > 0 (very high ROI)
            elif success_rate > 0:
                 # If cost is 0 but success rate is positive, this gift is infinitely profitable.
                 # This is likely an edge case, but we should prioritize it if it occurs.
                 # Set a very high ROI to ensure it's selected.
                 roi = float('inf')
                 if roi > best_roi:
                    best_roi = roi
                    best_gift = gift


    return best_gift

def strategy_4(row):
    """
    Strategy 4: Prioritizes gifts based on highest expected success rates in specific scenarios,
    then considers others based on potential impact/eligibility.
    """
    hotel_type = row.get('hotel_type', '')
    stay_nights = row.get('stay_nights', DEFAULT_STAY_NIGHTS)
    total_guests = row.get('total_guests', DEFAULT_TOTAL_GUESTS)

    # 1. Prioritize Room Upgrade (B) for City Hotels ('country_y') with highest success rate (0.8)
    if hotel_type == 'country_y' and is_eligible_for_gift(row, 'B'):
        return 'B'
    # 2. Prioritize SPA Access (D) for Resort Hotels (not 'country_y') with highest success rate (0.7)
    elif hotel_type != 'country_y' and is_eligible_for_gift(row, 'D'):
        return 'D'
    # 3. Consider Breakfast (A) if eligible and potentially higher impact (more guests or nights)
    # The success rate of A increases with sqrt(total_guests * stay_nights)
    elif is_eligible_for_gift(row, 'A') and (stay_nights >= 2 or total_guests >= 2):
        return 'A'
    # 4. Offer Parking (C) if eligible - fixed success rate (0.5)
    elif is_eligible_for_gift(row, 'C'):
        return 'C'
    # 5. Otherwise, no gift
    return np.nan


# --- EVALUACIÓN ---

def evaluate_strategy(df, strategy_func):
    """Evaluates a given strategy on the DataFrame."""
    df_eval = df.copy()
    # Apply the strategy to determine the gift for each booking
    df_eval['gift'] = df_eval.apply(strategy_func, axis=1)

    total_revenue_without = 0
    total_revenue_with = 0
    total_cost = 0
    total_gifts = 0

    for _, row in df_eval.iterrows():
        rate = row.get('rate', DEFAULT_RATE)
        stay_nights = row.get('stay_nights', DEFAULT_STAY_NIGHTS)
        # Calculate the total potential revenue from the booking
        total_potential_revenue = rate * stay_nights
        # Calculate the expected revenue if the booking were canceled and replaced
        replacement_rate = calculate_replacement_rate(row)
        expected_without = total_potential_revenue * replacement_rate

        total_revenue_without += expected_without

        gift = row['gift']
        # If a gift is offered
        if pd.notna(gift):
            total_gifts += 1
            gift_cost = calculate_gift_cost(row, gift)
            # The campaign effect is the success rate of the gift preventing cancellation
            campaign_effect = calculate_success_rate(row, gift)
            # Expected revenue if the campaign is successful (booking is retained)
            # This is the full potential revenue * success rate of retention
            expected_with = total_potential_revenue * campaign_effect

            # Total cost of the campaign for this booking
            total_cost += gift_cost + CAMPAIGN_COST
            # Add the expected revenue with the campaign
            total_revenue_with += expected_with
        else:
            # If no gift is offered, the expected revenue is the same as without the campaign (based on replacement rate)
            total_revenue_with += expected_without

    # Calculate net benefit and ROI
    # Net benefit is the increase in revenue (revenue with campaign minus revenue without campaign) minus the total campaign cost
    net_benefit = total_revenue_with - total_revenue_without - total_cost
    # ROI is Net Benefit / Total Cost, avoid division by zero
    roi = (net_benefit / total_cost) if total_cost > 0 else 0

    # Print evaluation details
    print(f"\n--- Evaluation Details for Strategy ---")
    print(f"Total Revenue Without Campaign: {total_revenue_without:.2f}")
    print(f"Total Revenue With Campaign: {total_revenue_with:.2f}")
    print(f"Total Cost of Campaign: {total_cost:.2f}")
    print(f"Total Gifts Offered: {total_gifts}")
    print(f"Net Benefit: {net_benefit:.2f}")
    print(f"ROI: {roi:.2f}")

    return {
        'net_benefit': net_benefit,
        'ROI': roi,
        'total_cost': total_cost,
        'total_gifts': total_gifts
    }


# --- SCRIPT PRINCIPAL ---

def main():
    # Cargar datos
    try:
        bookings = pd.read_csv("data/bookings_test.csv")
        predictions = pd.read_csv("data/output_predictions.csv")
        hotels = pd.read_csv("data/hotels.csv")
    except FileNotFoundError as e:
        print(f"Error loading data file: {e}. Make sure 'data' directory and files exist.")
        return

    # Merge hotel data into bookings
    bookings = bookings.merge(hotels, on='hotel_id', how='left')

    # Imputar nulos
    bookings = impute_missing_values(bookings)

    # Ensure 'arrival_date' is in datetime format for calculate_replacement_rate
    bookings['arrival_date'] = pd.to_datetime(bookings['arrival_date'])

    # Combine predictions with bookings
    # Assuming output_predictions.csv contains a single column of predictions matching the order of bookings_test.csv
    if len(bookings) != len(predictions):
         print("Warning: Length of bookings and predictions do not match. Proceeding with caution.")
         # Optionally, handle this mismatch more robustly, e.g., by merging on a common ID if available.
         # For this script's purpose, assuming they align by index/order after initial loading.


    # The prediction values are assumed to be in the first column of output_predictions.csv
    bookings['prediction'] = predictions.iloc[:, 0].values.flatten().astype(int)

    # Filter for bookings with a positive prediction (likely to cancel)
    bookings_positive = bookings[bookings['prediction'] == 1].copy()

    print(f"Número de reservas con predicción positiva: {len(bookings_positive)}")
    print(f"Valores únicos en predictions:\n{bookings['prediction'].value_counts()}")

    # Check if there are any positive predictions to evaluate strategies on
    if bookings_positive.empty:
        print("\nNo bookings with positive prediction to evaluate strategies.")
        # Apply no gift to all bookings if no positive predictions
        bookings['final_gift'] = np.nan
    else:
        # Evaluar estrategias
        print("\nEvaluating Strategies...")
        eval_1 = evaluate_strategy(bookings_positive, strategy_1)
        eval_2 = evaluate_strategy(bookings_positive, strategy_2)
        eval_3 = evaluate_strategy(bookings_positive, strategy_3)
        eval_4 = evaluate_strategy(bookings_positive, strategy_4) # Evaluate the new strategy

        # Store evaluation results in a dictionary for easy comparison
        eval_results = {
            1: eval_1,
            2: eval_2,
            3: eval_3,
            4: eval_4
        }

        print("\nResultados de evaluación:")
        for i, eval_res in eval_results.items():
            print(f"Estrategia {i}: Beneficio neto = {eval_res['net_benefit']:.2f}€, ROI = {eval_res['ROI']:.2f}")

        # Determine the best strategy based on Net Benefit
        best_strategy_idx = max(eval_results, key=lambda k: eval_results[k]['net_benefit'])
        best_net_benefit = eval_results[best_strategy_idx]['net_benefit']

        print(f"Mejor estrategia: Estrategia {best_strategy_idx} con beneficio neto de {best_net_benefit:.2f}€")

        # Apply the best strategy to the original dataframe (only for positive predictions)
        if best_strategy_idx == 1:
            bookings['final_gift'] = bookings.apply(lambda row: strategy_1(row) if row['prediction'] == 1 else np.nan, axis=1)
        elif best_strategy_idx == 2:
            bookings['final_gift'] = bookings.apply(lambda row: strategy_2(row) if row['prediction'] == 1 else np.nan, axis=1)
        elif best_strategy_idx == 3:
            bookings['final_gift'] = bookings.apply(lambda row: strategy_3(row) if row['prediction'] == 1 else np.nan, axis=1)
        elif best_strategy_idx == 4:
             bookings['final_gift'] = bookings.apply(lambda row: strategy_4(row) if row['prediction'] == 1 else np.nan, axis=1)


    # Guardar predicciones finales
    output_filename = "data/gift_predictions.csv"
    try:
        bookings[['final_gift']].to_csv(output_filename, index=False, header=False)
        print(f"\nArchivo generado: {output_filename}")
    except IOError as e:
        print(f"Error saving output file {output_filename}: {e}")


if __name__ == "__main__":
    main()