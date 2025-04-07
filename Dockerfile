FROM python:3.12-slim

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

ENV MODEL_PATH=/app/model/model.pkl
ENV INPUT_BOOKINGS_PATH=/app/data/bookings_test.csv
ENV INPUT_HOTELS_PATH=/app/data/hotels.csv
ENV OUTPUT_PREDICTIONS_PATH=/app/data/predictions.csv

CMD ["python", "inference.py"]