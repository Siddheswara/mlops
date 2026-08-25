from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from returns_app.api import app


def test_index_page() -> None:
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "Predict return probability" in response.text


def test_metadata_includes_metrics_and_options() -> None:
    with TestClient(app) as client:
        response = client.get("/metadata")
        assert response.status_code == 200
        body = response.json()
        assert body["model"] == "logistic_regression"
        assert body["metrics"]["accuracy"] == 0.8558
        assert body["metrics"]["f1"] == 0.784
        assert body["metrics"]["precision"] == 0.7194
        assert body["metrics"]["recall"] == 0.8615
        assert "Fashion" in body["categorical_options"]["product_category"]
        assert body["categorical_options"]["sub_category_by_category"]["Beauty"] == [
            "Haircare",
            "Makeup",
            "Skincare",
        ]


def test_predict_endpoint() -> None:
    row = pd.read_csv(Path(__file__).parents[1] / "returns_dataset.csv").iloc[0]
    payload = row.drop(labels=["order_id", "returned"]).to_dict()
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert 0 <= response.json()["probability"] <= 1
        assert client.get("/health").json()["status"] == "ok"