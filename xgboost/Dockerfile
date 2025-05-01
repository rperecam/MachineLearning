FROM python:3.11-slim

ENV SCRIPT_TO_RUN=train

WORKDIR /app

# Crear estructura de directorios
RUN mkdir -p /app/data /app/models

# Copiar requirements y archivos necesarios
COPY ensembled/requirements.txt /app/
COPY data/*.csv /app/data/
COPY ensembled/*.py /app/

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Configurar variables de entorno
ENV INFERENCE_DATA_PATH=/app/data/bookings_test.csv
ENV HOTELS_DATA_PATH=/app/data/hotels.csv
ENV TRAIN_DATA_PATH=/app/data/bookings_train.csv
ENV MODEL_PATH=/app/models/pipeline.cloudpkl
ENV OUTPUT_PATH=/app/data/output_predictions.csv

# Comando principal
CMD ["sh", "-c", "python -m $SCRIPT_TO_RUN"]

#------------ Instrucciones de uso desde la raíz del proyecto en el terminal---------#
# Para construir la imagen de Docker:
# docker build -t hotel-predictor -f xgboost/Dockerfile .
#
# Para entrenar el modelo:
# docker run -v "$(pwd)/models:/app/models" hotel-predictor
#
# Para ejecutar inferencia y guardar predicciones en data/:
# docker run -e SCRIPT_TO_RUN=inference -v "$(pwd)/models:/app/models" -v "$(pwd)/data:/app/data" hotel-predictor