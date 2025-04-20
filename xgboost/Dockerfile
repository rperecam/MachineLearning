FROM python:3.12-slim

ENV SCRIPT_TO_RUN=bost_train

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

# NO CAMBIAR ESTAS RUTAS
ENV INFERENCE_DATA_PATH=/app/bookings_test.csv
ENV HOTELS_DATA_PATH=/app/hotels.csv
ENV TRAIN_DATA_PATH=/app/bookings_train.csv
ENV MODEL_PATH=/app/pipeline.cloudpkl

CMD ["sh", "-c", "python -m $SCRIPT_TO_RUN"]