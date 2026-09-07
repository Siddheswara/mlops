FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    MODEL_PATH=/app/artifacts/return_model.joblib \
    PYTHONUNBUFFERED=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends nginx curl \
 && rm -rf /var/lib/apt/lists/* \
 && rm -f /etc/nginx/sites-enabled/default

WORKDIR /app

COPY pyproject.toml requirements.txt ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.txt

# The model artifact must exist on the build machine (pull it with DVC first).
RUN mkdir -p artifacts
COPY artifacts/return_model.joblib ./artifacts/return_model.joblib

COPY docker/nginx.conf /etc/nginx/conf.d/returns_api.conf
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 80

# No systemd in containers: entrypoint runs uvicorn + nginx
CMD ["/entrypoint.sh"]
