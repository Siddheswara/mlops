# Returns Prediction API

A packaged scikit-learn pipeline and FastAPI service for predicting whether an order will be returned.

## Run locally

```powershell
.\.venv\Scripts\Activate.ps1
pip install -e .
python -m returns_app.train --data returns_dataset.csv --output artifacts/return_model.joblib
uvicorn returns_app.api:app --reload
```

Open `http://127.0.0.1:8000/` for the prediction UI, or `http://127.0.0.1:8000/docs` for the interactive API docs.

## Endpoints

- `GET /` serves the prediction form (toggles for binary fields, dropdowns for categories).
- `GET /health` verifies the service and model artifact.
- `GET /metadata` lists the model, holdout metrics, and form options.
- `POST /predict` accepts one order and returns a return probability.

The model artifact is intentionally generated during deployment or setup and is ignored by Git. The notebook remains useful for exploration and reporting; production preprocessing and inference live under `src/returns_app`.
