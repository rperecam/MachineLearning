FROM python:3.12-slim

ENV SCRIPT_TO_RUN=bost_train

WORKDIR /app

# Copiar primero solo los datos
COPY data/*.csv /app/
COPY xgboost/data/*.csv /app/ 2>/dev/null || true

# Luego copiar el código
COPY xgboost/*.py /app/
COPY *.py /app/ 2>/dev/null || true
COPY xgboost/requirements.txt /app/

RUN pip install -r requirements.txt

# NO CAMBIAR ESTAS RUTAS
ENV INFERENCE_DATA_PATH=/app/bookings_train.csv
ENV HOTELS_DATA_PATH=/app/hotels.csv
ENV TRAIN_DATA_PATH=/app/bookings_train.csv
ENV MODEL_PATH=/app/pipeline.cloudpkl

# Crear directorio para modelos
RUN mkdir -p /app/models

# Verificar los archivos copiados
RUN ls -la /app/

CMD ["sh", "-c", "python -m $SCRIPT_TO_RUN"]