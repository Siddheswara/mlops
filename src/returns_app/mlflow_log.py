from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import mlflow
from sklearn.pipeline import Pipeline

SKOPS_TRUSTED = [
    "returns_app.model.add_features",
    "numpy.dtype",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT = "returns-production-new"


def _git_commit() -> str | None:
    if sha := os.getenv("GITHUB_SHA"):
        return sha
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _git_branch() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def configure_mlflow(
    experiment: str = DEFAULT_EXPERIMENT,
    tracking_uri: str | None = None,
) -> None:
    uri = tracking_uri or os.getenv(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}",
    )
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)


def log_training_run(
    *,
    pipeline: Pipeline,
    summary: dict[str, Any],
    run_name: str = "cli-train",
    source: str = "src/returns_app/train.py",
    metrics: dict[str, float] | None = None,
) -> str:
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("mlflow.source.name", source)
        mlflow.set_tag("mlflow.source.type", "LOCAL")
        if commit := _git_commit():
            mlflow.set_tag("mlflow.source.git.commit", commit)
        if branch := _git_branch():
            mlflow.set_tag("mlflow.source.git.branch", branch)

        mlflow.log_params({k: str(v) for k, v in summary.items() if k != "pipeline"})
        if metrics:
            mlflow.log_metrics(metrics)

        mlflow.log_artifact(summary["artifact"])
        mlflow.sklearn.log_model(
            pipeline,
            name="sklearn_pipeline",
            skops_trusted_types=SKOPS_TRUSTED,
        )
        return mlflow.active_run().info.run_id