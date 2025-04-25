FROM python:3.12-slim

ENV SCRIPT_TO_RUN=boost_train

WORKDIR /app

# Crear estructura de directorios
RUN mkdir -p /app/data /app/models

# Copiar archivos necesarios
COPY data/*.csv /app/data/
COPY xgboost/*.py /app/
COPY xgboost/requirements.txt /app/


# Modifica la línea para instalar las herramientas de compilación necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python
RUN pip install -r requirements.txt

# Configurar rutas de datos
ENV INFERENCE_DATA_PATH=/app/data/bookings_train.csv
ENV HOTELS_DATA_PATH=/app/data/hotels.csv
ENV TRAIN_DATA_PATH=/app/data/bookings_train.csv
ENV MODEL_PATH=/app/models/pipeline.cloudpkl

# Verificar archivos
RUN ls -la /app/ && ls -la /app/data/

# Comando principal
CMD ["sh", "-c", "python -m $SCRIPT_TO_RUN"]

# Instrucciones de uso desde la raíz del proyecto en el terminal:
# docker build -t xgboost-model -f xgboost/Dockerfile .
# docker run -v "${PWD}\models:/app/models" xgboost-model
# docker run -e SCRIPT_TO_RUN=inference -v "${PWD}\models:/app/models" -v "${PWD}\data:/app/data" xgboost-model