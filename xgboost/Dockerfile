FROM python:3.11-slim

ENV SCRIPT_TO_RUN=boost_train

WORKDIR /app

# Instalar dependencias del sistema necesarias para LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Crear estructura de directorios
RUN mkdir -p /app/data /app/models

# Copiar requirements y archivos necesarios
COPY xgboost/requirements.txt /app/
COPY data/*.csv /app/data/
COPY xgboost/*.py /app/

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Configurar variables de entorno
ENV INFERENCE_DATA_PATH=/app/data/bookings_train.csv
ENV HOTELS_DATA_PATH=/app/data/hotels.csv
ENV TRAIN_DATA_PATH=/app/data/bookings_train.csv
ENV MODEL_PATH=/app/models/pipeline.cloudpkl

# Comando principal
CMD ["sh", "-c", "python -m $SCRIPT_TO_RUN"]

#------------ Instrucciones de uso desde la raíz del proyecto en el terminal---------#
# Para construir la imagen de Docker:
# docker build -t ensembled-model -f xgboost/Dockerfile .
# Para ejecutar los contenedores con diferentes scripts:
# docker run -v "$(pwd)/models:/app/models" ensembled-model
# docker run -e SCRIPT_TO_RUN=inference -v "$(pwd)/models:/app/models" -v "$(pwd)/data:/app/data" ensembled-model