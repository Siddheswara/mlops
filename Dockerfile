FROM python:3.12-slim-bookworm

ARG MLFLOW_TRACKING_URI=http://localhost:5000

ENV DEBIAN_FRONTEND=noninteractive \
    MODEL_PATH=/app/artifacts/return_model.joblib \
    MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} \
    PYTHONUNBUFFERED=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends nginx curl \
 && rm -rf /var/lib/apt/lists/* \
 && rm -f /etc/nginx/sites-enabled/default

WORKDIR /app

COPY pyproject.toml requirements.txt ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.txt

# Dataset must exist on the build machine (gitignored; pull with DVC first).
# This build step intentionally disables MLflow so the image can be created
# without depending on a running MLflow container.
COPY returns_dataset.csv ./returns_dataset.csv
RUN mkdir -p artifacts \
 && python -m returns_app.train \
      --data returns_dataset.csv \
      --output artifacts/return_model.joblib \
      --no-mlflow

# For EC2 without docker-compose, override MLFLOW_TRACKING_URI at runtime, e.g.:
#   docker run --rm --network my-net -e MLFLOW_TRACKING_URI=http://mlflow:5000 my-image \
#       python -m returns_app.train --data returns_dataset.csv --output artifacts/return_model.joblib

COPY docker/nginx.conf /etc/nginx/conf.d/returns_api.conf
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 80

# No systemd in containers: entrypoint runs uvicorn + nginx
CMD ["/entrypoint.sh"]
