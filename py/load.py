import pandas as pd

def merge_data(bookings_file, hotels_file, output_file):
    """
    Carga y fusiona datos de reservas y hoteles, guardando el resultado en un archivo CSV.

    Args:
        bookings_file (str): Ruta al archivo CSV de reservas.
        hotels_file (str): Ruta al archivo CSV de hoteles.
        output_file (str): Ruta al archivo CSV de salida.
    """
    df_book = pd.read_csv(bookings_file)
    df_hotel = pd.read_csv(hotels_file)
    df = pd.merge(df_book, df_hotel, on='hotel_id')
    df.to_csv(output_file, index=False)
    print(f"Datos cargados y guardados en: {output_file}")