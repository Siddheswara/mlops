from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from .mlflow_log import configure_mlflow, log_training_run
from .model import FEATURE_COLUMNS, TARGET_COLUMN, clean_data, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the returns prediction model.")
    parser.add_argument("--data", type=Path, default=Path("returns_dataset.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/return_model.joblib"))
    parser.add_argument("--experiment", default="returns-production-new")
    parser.add_argument("--run-name", default="cli-train")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging")
    args = parser.parse_args()

    summary = train_model(args.data, args.output)
    print(summary)

    if args.no_mlflow:
        return

    configure_mlflow(experiment=args.experiment)

    data = clean_data(pd.read_csv(args.data), training=True)
    x, y = data[FEATURE_COLUMNS], data[TARGET_COLUMN].astype(int)
    _, x_test, _, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    preds = summary["pipeline"].predict(x_test)
    metrics = {"holdout_f1": float(f1_score(y_test, preds))}

    run_id = log_training_run(
        pipeline=summary["pipeline"],
        summary=summary,
        run_name=args.run_name,
        metrics=metrics,
    )
    print(f"mlflow run_id={run_id}")


if __name__ == "__main__":
    main()