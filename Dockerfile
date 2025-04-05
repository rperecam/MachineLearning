FROM python:3.12-slim

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

# Definir las rutas de los archivos como variables de entorno
ENV MODEL_PATH=/app/model/model.pkl
ENV INPUT_DATA_PATH=/app/data/bookings_train.csv
ENV OUTPUT_PREDICTIONS_PATH=/app/data/predictions.csv

CMD ["python", "inference.py"]