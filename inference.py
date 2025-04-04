# inference.py
import pandas as pd
import argparse
import os
import sys
from typing import Optional

# Import the pipeline class definition from the separate file
try:
    from train import HotelBookingPipeline
except ImportError:
    print("Error: Asegúrate de que 'train.py' existe y contiene la clase 'HotelBookingPipeline'.")
    sys.exit(1)

def run_inference(input_csv: str, output_csv: str, model_path: str, model_type: str = 'logistic') -> bool:
    """
    Loads the pipeline, performs inference on input_csv, and saves results to output_csv.

    Args:
        input_csv (str): Path to the input CSV file with new data.
        output_csv (str): Path to save the output CSV file with predictions.
        model_path (str): Path to the saved pipeline (.pkl) file.
        model_type (str): Type of model to use ('logistic' or 'sgd').

    Returns:
        bool: True if inference was successful, False otherwise.
    """
    print(f"--- Iniciando Inferencia ---")
    print(f"Modelo a cargar: {model_path}")
    print(f"Datos de entrada: {input_csv}")
    print(f"Archivo de salida: {output_csv}")
    print(f"Tipo de modelo especificado: {model_type}")

    # 1. Load the pipeline
    pipeline_instance: Optional[HotelBookingPipeline] = HotelBookingPipeline.load_pipeline(model_path)

    if pipeline_instance is None:
        print(f"Error fatal: No se pudo cargar el pipeline desde {model_path}")
        return False

    # Verify the correct model type is loaded/available if needed
    if model_type == 'logistic' and pipeline_instance.full_pipeline_logistic is None:
        print(f"Error: Se especificó el modelo 'logistic', pero no está cargado en el pipeline.")
        return False
    elif model_type == 'sgd' and pipeline_instance.full_pipeline_sgd is None:
        print(f"Error: Se especificó el modelo 'sgd', pero no está cargado en el pipeline.")
        return False


    # 2. Load input data
    try:
        input_df = pd.read_csv(input_csv)
        print(f"Datos de entrada cargados. Shape: {input_df.shape}")
    except FileNotFoundError:
        print(f"Error fatal: Archivo de entrada no encontrado en {input_csv}")
        return False
    except Exception as e:
        print(f"Error fatal al leer el archivo de entrada {input_csv}: {e}")
        return False

    # 3. Perform prediction
    print(f"Realizando predicciones con el modelo {model_type.upper()}...")
    try:
        predictions = pipeline_instance.predict(input_df, model_type=model_type)
        # Optional: Get probabilities as well
        # probabilities = pipeline_instance.predict_proba(input_df, model_type=model_type)
    except Exception as e:
        print(f"Error fatal durante la predicción: {e}")
        import traceback
        traceback.print_exc()
        return False

    if predictions is None:
        print("Error fatal: La función predict devolvió None.")
        return False

    print(f"Predicciones generadas ({len(predictions)} registros).")

    # 4. Prepare output data
    output_df = input_df.copy()
    output_df['prediction'] = predictions
    # Optional: Add probability score if needed (e.g., probability of class 1)
    # if probabilities is not None:
    #     output_df['prediction_probability_1'] = probabilities[:, 1] # Assuming binary classification

    # 5. Save output data
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
        output_df.to_csv(output_csv, index=False)
        print(f"Resultados guardados exitosamente en: {output_csv}")
    except Exception as e:
        print(f"Error fatal al guardar los resultados en {output_csv}: {e}")
        return False

    print(f"--- Inferencia Completada Exitosamente ---")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejecutar inferencia con el pipeline de Hotel Booking.")
    parser.add_argument("--input-csv", type=str, required=True,
                        help="Ruta al archivo CSV de entrada con los nuevos datos.")
    parser.add_argument("--output-csv", type=str, required=True,
                        help="Ruta al archivo CSV de salida donde se guardarán las predicciones.")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Ruta al archivo .pkl del pipeline entrenado.")
    parser.add_argument("--model-type", type=str, default='logistic', choices=['logistic', 'sgd'],
                        help="Tipo de modelo a usar para la predicción ('logistic' o 'sgd'). Default: logistic.")

    args = parser.parse_args()

    success = run_inference(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        model_path=args.model_path,
        model_type=args.model_type
    )

    if not success:
        sys.exit(1) # Exit with error code if inference failed