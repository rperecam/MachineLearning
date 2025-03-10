import pandas as pd

def load_and_merge_data(bookings_path: str, hotels_path: str) -> pd.DataFrame:
    """
    Carga los datos de reservas y hoteles desde archivos CSV y los une en un solo DataFrame.
    """
    df_book = pd.read_csv(bookings_path)
    df_hotel = pd.read_csv(hotels_path)
    df = pd.merge(df_book, df_hotel, on='hotel_id')
    return df

def load_data_pipeline(bookings_path: str, hotels_path: str, output_path: str) -> None:
    """
    Pipeline que carga y une los datos, y guarda el resultado en un archivo CSV.
    """
    merged_df = load_and_merge_data(bookings_path, hotels_path)
    merged_df.to_csv(output_path, index=False)
    print(f"Datos cargados y guardados en: {output_path}")

if __name__ == "__main__":
    bookings_file = "data/bookings_train.csv"  # Reemplaza con la ruta real
    hotels_file = "data/hotels.csv"    # Reemplaza con la ruta real
    output_file = "data/data_processed/data.csv"  # Nombre del archivo de salida

    load_data_pipeline(bookings_file, hotels_file, output_file)