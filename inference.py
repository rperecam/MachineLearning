import pandas as pd
import cloudpickle
import os
import numpy as np  # Importa numpy

# Configuración de variables de entorno
#os.environ['MODEL_PATH'] = 'model/model.pkl'  # Ruta al modelo entrenado
#os.environ['INPUT_BOOKINGS_PATH'] = 'data/bookings_train.csv'  # Ruta al archivo bookings.csv
#os.environ['INPUT_HOTELS_PATH'] = 'data/hotels.csv'  # Ruta al archivo hotels.csv
#os.environ['OUTPUT_PREDICTIONS_PATH'] = 'data/output_predictions.csv'  # Ruta para guardar las predicciones

# Definir las rutas de los archivos desde las variables de entorno
model_path = os.environ.get('MODEL_PATH')
input_bookings_path = os.environ.get('INPUT_BOOKINGS_PATH')
input_hotels_path = os.environ.get('INPUT_HOTELS_PATH')  # Ruta para el archivo hotels.csv
output_predictions_path = os.environ.get('OUTPUT_PREDICTIONS_PATH')

# Cargar el modelo entrenado
def load_model(filepath):
    try:
        with open(filepath, 'rb') as f:
            model = cloudpickle.load(f)
        print(f"Modelo cargado desde: {filepath}")
        return model
    except FileNotFoundError:
        print(f"Error: Archivo no encontrado en: {filepath}")
        return None
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None

# Alinear las columnas del DataFrame de entrada con las columnas esperadas por el modelo
def align_columns(df, expected_columns):
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0  # O usa un valor por defecto apropiado
    return df[expected_columns]

# Realizar predicciones y guardar en un archivo CSV
def make_predictions(model, input_bookings_path, input_hotels_path, output_predictions_path):
    try:
        # Cargar los datos de entrada
        df_bookings = pd.read_csv(input_bookings_path)
        df_hotels = pd.read_csv(input_hotels_path)

        # Fusionar los DataFrames
        df_input = pd.merge(df_bookings, df_hotels, on='hotel_id', how='left').drop('hotel_id', axis=1, errors='ignore')

        # Asegurarse de que no haya 'reservation_status' en 'Booked' o NaN
        df_input = df_input[~df_input['reservation_status'].isin(['Booked', np.nan])].copy()

        # Separar características y target (si es necesario para la alineación)
        if 'is_canceled' in df_input.columns:
            y = df_input['is_canceled']
            df_input = df_input.drop('is_canceled', axis=1)

        # Alinear las columnas del DataFrame de entrada con las columnas esperadas por el modelo
        # Obtener las columnas utilizadas durante el entrenamiento
        expected_columns = model.named_steps['preprocessor'].transformers_[0][2] + model.named_steps['preprocessor'].transformers_[1][2]

        # Alinear las columnas del DataFrame de entrada con las columnas esperadas por el modelo
        df_input_aligned = align_columns(df_input, expected_columns)

        # Realizar predicciones
        predictions = model.predict(df_input_aligned)
        probabilities = model.predict_proba(df_input_aligned)[:, 1]

        # Guardar las predicciones en un archivo CSV
        df_output = pd.DataFrame({
            'predicted_cancellation': predictions,
            'probability_cancellation': probabilities
        })
        df_output.to_csv(output_predictions_path, index=False)
        print(f"Predicciones guardadas en: {output_predictions_path}")
    except Exception as e:
        print(f"Error durante la predicción: {e}")

if __name__ == '__main__':
    model = load_model(model_path)
    if model:
        make_predictions(model, input_bookings_path, input_hotels_path, output_predictions_path)