from pathlib import Path

import pandas as pd

from returns_app.model import FEATURE_COLUMNS, load_model, train_model


PROJECT_ROOT = Path(__file__).parents[1]


def test_model_round_trip(tmp_path: Path) -> None:
    artifact = tmp_path / "model.joblib"
    summary = train_model(PROJECT_ROOT / "returns_dataset.csv", artifact)
    model = load_model(artifact)
    row = pd.read_csv(PROJECT_ROOT / "returns_dataset.csv").loc[:, FEATURE_COLUMNS].head(1)
    prediction = model.predict_proba(row)
    assert summary["rows"] == 12000
    assert prediction.shape == (1, 2)
    assert 0 <= prediction[0, 1] <= 1
