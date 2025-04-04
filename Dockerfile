FROM python:3.12-slim

ENV SCRIPT_TO_RUN=train

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

ENV INFERENCE_DATA_PATH=/app/inference_clients.parquet
ENV TRAIN_DATA_PATH=/app/train_clients.parquet
ENV MODEL_PATH=/app/pipeline.cloudpkl

# PARA CREAR LA IMAGEN
# docker build . -t uax-churn:latest

# PARA CORRER EL CONTENEDOR DE ENTRENAMIENTO
# docker run -d -v .:/app -e SCRIPT_TO_RUN=train uax-churn:latest

# PARA CORRER EL CONTENEDOR DE INFERENCIA
# docker run -d -v .:/app -e SCRIPT_TO_RUN=inference uax-churn:latest

CMD ["sh", "-c", "python -m $SCRIPT_TO_RUN"]