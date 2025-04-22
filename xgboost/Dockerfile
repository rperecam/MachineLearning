FROM python:3.12-slim

ENV SCRIPT_TO_RUN=bost_train

WORKDIR /app

# Crear primero la estructura de directorios necesaria
RUN mkdir -p /app/data /app/models

# Copiar los archivos CSV desde la ubicación correcta
COPY data/*.csv /app/data/

# Copiar los scripts Python
COPY xgboost/*.py /app/

# Copiar el archivo de requisitos e instalar dependencias
COPY xgboost/requirements.txt /app/
RUN pip install -r requirements.txt

# Configurar las rutas correctas para los archivos de datos
ENV INFERENCE_DATA_PATH=/app/data/bookings_train.csv
ENV HOTELS_DATA_PATH=/app/data/hotels.csv
ENV TRAIN_DATA_PATH=/app/data/bookings_train.csv
ENV MODEL_PATH=/app/models/pipeline.cloudpkl

# Verificar los archivos copiados
RUN ls -la /app/ && ls -la /app/data/

CMD ["sh", "-c", "python -m $SCRIPT_TO_RUN"]

# Para construir la imagen, ejecutar el siguiente comando en la terminal desde la raiz del proyecto:
# docker build -t xgboost-model -f xgboost/Dockerfile .
# docker run -v "${PWD}\models:/app/models" xgboost-model
# docker run -e SCRIPT_TO_RUN=inference -v "${PWD}\models:/app/models" xgboost-model