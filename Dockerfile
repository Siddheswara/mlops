FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml requirements.txt ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.txt
COPY returns_dataset.csv ./returns_dataset.csv
RUN python -m returns_app.train --data returns_dataset.csv --output artifacts/return_model.joblib

EXPOSE 8000
CMD ["uvicorn", "returns_app.api:app", "--host", "0.0.0.0", "--port", "8000"]
