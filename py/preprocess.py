import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(df_train, test_size=0.3, random_state=42):
    """
    Realiza el preprocesamiento inicial de los datos y los divide en train y test.
    """

    # Eliminar hotel_id y country_x y days_diff
    df_train.drop(columns=['hotel_id', 'country_x', 'days_diff'], errors='ignore', inplace=True)

    # Renombrar la columna 'country_y' a 'country'
    if 'country_y' in df_train.columns:
        df_train.rename(columns={'country_y': 'country'}, inplace=True)

    # Transformaciones de fechas y creación de variable target
    df_train['booking_date'] = pd.to_datetime(df_train['booking_date'])
    df_train['arrival_date'] = pd.to_datetime(df_train['arrival_date'])
    df_train['reservation_status_date'] = pd.to_datetime(df_train['reservation_status_date'])

    df_train["days_diff"] = (pd.to_datetime(df_train["arrival_date"]) - pd.to_datetime(df_train["reservation_status_date"])).dt.days
    df_train["cancelled_last_30_days"] = ((df_train["reservation_status"] == "Canceled") & (df_train["days_diff"] <= 30)).astype(int)

    # Cálculo de la anticipación de la reserva
    df_train['advance_reservation_days'] = (pd.to_datetime(df_train['arrival_date']) - pd.to_datetime(df_train['booking_date'])).dt.days

    # De boleano a entero
    df_train['parking'] = df_train['parking'].astype(int)
    df_train['restaurant'] = df_train['restaurant'].astype(int)
    df_train['pool_and_spa'] = df_train['pool_and_spa'].astype(int)

    # Manejo de outliers en 'rate' y 'total_guests' usando el rango intercuartil (IQR)
    q1_rate = df_train["rate"].quantile(0.25)
    q3_rate = df_train["rate"].quantile(0.75)
    iqr_rate = q3_rate - q1_rate
    lower_bound_rate = q1_rate - 1.5 * iqr_rate
    upper_bound_rate = q3_rate + 1.5 * iqr_rate
    df_train = df_train[(df_train["rate"] >= lower_bound_rate) & (df_train["rate"] <= upper_bound_rate)]

    q1_guests = df_train["total_guests"].quantile(0.25)
    q3_guests = df_train["total_guests"].quantile(0.75)
    iqr_guests = q3_guests - q1_guests
    lower_bound_guests = q1_guests - 1.5 * iqr_guests
    upper_bound_guests = q3_guests + 1.5 * iqr_guests
    df_train = df_train[(df_train["total_guests"] >= lower_bound_guests) & (df_train["total_guests"] <= upper_bound_guests)]

    # Eliminar columnas de fecha
    date_cols = ['booking_date', 'arrival_date', 'reservation_status_date']
    df_train.drop(columns=date_cols, inplace=True)

    # Imputar valores nulos específicos
    df_train['special_requests'].fillna(0, inplace=True)
    df_train['required_car_parking_spaces'].fillna(0, inplace=True)

    # Eliminar columnas redundantes
    df_train.drop(columns=['reservation_status'], inplace=True)

    # Separar la variable objetivo y las predictoras
    X = df_train.drop('cancelled_last_30_days', axis=1)
    y = df_train['cancelled_last_30_days']

    # Dividir en train y test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    return X_train, X_test, y_train, y_test