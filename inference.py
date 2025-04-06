import pandas as pd
import cloudpickle
import os

# Definir las rutas de los archivos desde las variables de entorno
model_path = os.environ.get('MODEL_PATH')
input_data_path = os.environ.get('INPUT_DATA_PATH')
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
def make_predictions(model, input_data_path, output_predictions_path):
    try:
        # Cargar los datos de entrada
        df_input = pd.read_csv(input_data_path)

        # Alinear las columnas del DataFrame de entrada con las columnas esperadas por el modelo
        expected_columns = model.named_steps['preprocessor'].transformers_[0][2] + model.named_steps['preprocessor'].transformers_[1][2]
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
        make_predictions(model, input_data_path, output_predictions_path)