from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TARGET_COLUMN = "returned"
ID_COLUMN = "order_id"
CATEGORICAL_COLUMNS = [
    "product_category",
    "sub_category",
    "brand",
    "fulfillment_type",
    "payment_method",
]
BINARY_COLUMNS = [
    "fragile_item",
    "warranty_available",
    "delayed_delivery",
    "wishlist_before_purchase",
]
NUMERIC_COLUMNS = [
    "product_price",
    "discount_percent",
    "product_rating",
    "review_count",
    "product_return_rate",
    "category_return_rate",
    "brand_return_rate",
    "defect_rate",
    "seller_rating",
    "seller_return_rate",
    "quantity",
    "shipping_distance_km",
    "product_page_views",
    "customer_support_calls",
    "chat_interactions",
]
FEATURE_COLUMNS = CATEGORICAL_COLUMNS + BINARY_COLUMNS + NUMERIC_COLUMNS


def clean_data(data: pd.DataFrame, *, training: bool = False) -> pd.DataFrame:
    """Apply deterministic cleaning shared by training and inference."""
    result = data.copy()
    missing = sorted(set(FEATURE_COLUMNS) - set(result.columns))
    if missing:
        raise ValueError(f"Data is missing required feature columns: {missing}")

    for column in CATEGORICAL_COLUMNS:
        result[column] = result[column].astype("string").str.strip()
    result["fulfillment_type"] = (
        result["fulfillment_type"].str.replace("_", " ", regex=False).str.title()
    )
    for column in NUMERIC_COLUMNS + BINARY_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    if training:
        result = result.drop_duplicates()
        result[TARGET_COLUMN] = pd.to_numeric(result[TARGET_COLUMN], errors="coerce")
        result = result.dropna(subset=[TARGET_COLUMN])
        if not result[TARGET_COLUMN].isin([0, 1]).all():
            raise ValueError("Target column must contain only 0 and 1")
    return result


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create features available at prediction time."""
    result = data.copy()
    result["total_support_contacts"] = (
        result["customer_support_calls"] + result["chat_interactions"]
    )
    result["log_price"] = result["product_price"].clip(lower=0).map(math.log1p)
    result = result.drop(columns=["customer_support_calls", "chat_interactions"])
    return result


def build_pipeline() -> Pipeline:
    engineered_numeric = [
        column for column in NUMERIC_COLUMNS
        if column not in {"customer_support_calls", "chat_interactions"}
    ] + ["total_support_contacts", "log_price"] + BINARY_COLUMNS
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, engineered_numeric),
        ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
    ])
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("preprocess", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])


def train_model(data_path: str | Path, artifact_path: str | Path) -> dict[str, Any]:
    data = pd.read_csv(data_path)
    missing = sorted(set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    data = clean_data(data, training=True)
    pipeline = build_pipeline()
    pipeline.fit(data[FEATURE_COLUMNS], data[TARGET_COLUMN].astype(int))
    artifact = Path(artifact_path)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, artifact)
    return {"rows": len(data), "features": len(FEATURE_COLUMNS), "artifact": str(artifact), "pipeline": pipeline}


def load_model(artifact_path: str | Path) -> Pipeline:
    return joblib.load(artifact_path)
