import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocesa el conjunto de datos de reservas de hotel.
    """

    # 1. Conversión de las columnas de fechas a formato datetime
    df["reservation_status_date"] = pd.to_datetime(df["reservation_status_date"])
    df["arrival_date"] = pd.to_datetime(df["arrival_date"])
    df["booking_date"] = pd.to_datetime(df["booking_date"])

    # 2. Manejo de valores nulos
    df["required_car_parking_spaces"] = np.where(df["required_car_parking_spaces"] >= 1, 1, 0)
    df["total_guests"].fillna(df["total_guests"].mean(), inplace=True)
    df["total_guests"] = df["total_guests"].apply(lambda x: int(round(x, 0)))
    df["stay_nights"].fillna(df["stay_nights"].mean(), inplace=True)
    df["stay_nights"] = df["stay_nights"].astype(int)
    df.dropna(subset=["board"], inplace=True)
    df["board"].fillna("BB", inplace=True)
    df["market_segment"].fillna(df["market_segment"].mode()[0], inplace=True)
    df["distribution_channel"].fillna(df["distribution_channel"].mode()[0], inplace=True)
    df["rate"].fillna(df["rate"].mean(), inplace=True)

    # 3. Manejo de outliers en 'rate' y 'total_guests' usando el rango intercuartil (IQR)
    q1_rate = df["rate"].quantile(0.25)
    q3_rate = df["rate"].quantile(0.75)
    iqr_rate = q3_rate - q1_rate
    lower_bound_rate = q1_rate - 1.5 * iqr_rate
    upper_bound_rate = q3_rate + 1.5 * iqr_rate
    df = df[(df["rate"] >= lower_bound_rate) & (df["rate"] <= upper_bound_rate)]

    q1_guests = df["total_guests"].quantile(0.25)
    q3_guests = df["total_guests"].quantile(0.75)
    iqr_guests = q3_guests - q1_guests
    lower_bound_guests = q1_guests - 1.5 * iqr_guests
    upper_bound_guests = q3_guests + 1.5 * iqr_guests
    df = df[(df["total_guests"] >= lower_bound_guests) & (df["total_guests"] <= upper_bound_guests)]

    # 4. Transformación de las columnas categóricas
    df["market_segment"] = df["market_segment"].replace({
        "Corporate": "Business", "Aviation": "Business", "Online TA": "Leisure",
        "Offline TA/TO": "Leisure", "Groups": "Leisure", "Complementary": "Complementary"
    })
    df["distribution_channel"] = df["distribution_channel"].replace({
        "OTA": "Agency", "Offline TA/TO": "Agency", "GDS": "GDS", "Corporate": "Corporate"
    })
    df["special_requests"] = np.where(df["special_requests"] >= 1, 1, 0)
    df.rename(columns={"country_y": "country"}, inplace=True)
    df[["parking", "restaurant", "pool_and_spa"]] = df[["parking", "restaurant", "pool_and_spa"]].astype(int)
    df["room_type"] = np.where(df["room_type"] == "A", "A", "X")
    df["room_type"] = df["room_type"].apply(lambda x: 1 if x == "A" else 0)
    df["hotel_type"] = np.where(df["hotel_type"] == "City Hotel", 1, 0)

    # 5. Creación de la variable target y conjunto de validación
    df["days_diff"] = (df["arrival_date"] - df["reservation_status_date"]).dt.days
    df["cancelled_last_30_days"] = ((df["reservation_status"] == "Canceled") & (df["days_diff"] <= 30)).astype(int)
    df_validation = df[df["reservation_status"] == "Booked"].copy()
    df.drop(df[df["reservation_status"] == "Booked"].index, inplace=True)

    # 6. Cálculo de la anticipación de la reserva
    df['advance_reservation_days'] = (df['arrival_date'] - df['booking_date']).dt.days
    df_validation['advance_reservation_days'] = (df_validation['arrival_date'] - df_validation['booking_date']).dt.days

    # 7. Codificación de variables categóricas usando One-Hot Encoding
    df = pd.get_dummies(df, columns=['board', 'distribution_channel', 'country', 'market_segment'], drop_first=True)
    df = df.applymap(lambda x: 1 if x is True else (0 if x is False else x))

    # 8. Normalización de las columnas numéricas
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    binary_cols = [col for col in numerical_cols if df[col].isin([0, 1]).all()]
    cols_to_scale = [col for col in numerical_cols if col not in binary_cols]
    scaler = StandardScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    # Filtrar las fechas según la fecha, al tratarse de una serie temporal y no provocar data leakage
    df = df[df['booking_date'] <= '2017-06-30']
    df_validation = df_validation[df_validation['booking_date'] > '2017-06-30']

    # 9. Eliminación de columnas y filas innecesarias
    df.drop(df[df["reservation_status"] == "No-Show"].index, inplace=True)
    df.drop(['booking_date','reservation_status', "days_diff", 'country_x', 'arrival_date', 'reservation_status_date'], axis=1, inplace=True)
    df_validation.drop(["reservation_status_date", "country_x", 'hotel_id'], axis=1, inplace=True)

    # Eliminar columnas con alta correlación (feature selection)
    df.drop(["stay_nights", "distribution_channel_GDS", "total_rooms", "board_Undefined", "board_HB", "distribution_channel_Undefined"], axis=1, inplace=True)
    df.drop("hotel_id", axis=1, inplace=True)

    return df, df_validation

def preprocess_data_pipeline(input_path: str, output_train_path: str, output_val_path: str) -> None:
    """
    Pipeline que carga, preprocesa los datos y guarda los DataFrames resultantes.
    """
    df = pd.read_csv(input_path)
    train_df, val_df = preprocess_data(df)

    train_df.to_csv(output_train_path, index=False)
    val_df.to_csv(output_val_path, index=False)
    print(f"Datos de entrenamiento preprocesados y guardados en: {output_train_path}")
    print(f"Datos de validación preprocesados y guardados en: {output_val_path}")

if __name__ == "__main__":
    input_file = "data/data_processed/data.csv"  # Nombre del archivo de entrada
    train_output_file = "data/data_processed/train_preprocessed.csv"  # Nombre del archivo de entrenamiento
    val_output_file = "data/data_processed/validation_preprocessed.csv"  # Nombre del archivo de validación

    preprocess_data_pipeline(input_file, train_output_file, val_output_file)