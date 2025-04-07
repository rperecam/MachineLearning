FROM python:3.10-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

ENV MODEL_PATH=/app/model/model.pkl
ENV INPUT_BOOKINGS_PATH=/app/data/bookings_train.csv
ENV INPUT_HOTELS_PATH=/app/data/hotels.csv
ENV OUTPUT_PREDICTIONS_PATH=/app/data/predictions.csv

CMD ["python", "inference.py"]