FROM python:3.11-slim
RUN pip install --no-cache-dir mlflow==3.18.0
WORKDIR /mlflow
EXPOSE 5000
