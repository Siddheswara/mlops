from __future__ import annotations

import argparse
from pathlib import Path

from .model import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the returns prediction model.")
    parser.add_argument("--data", type=Path, default=Path("returns_dataset.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/return_model.joblib"))
    args = parser.parse_args()
    print(train_model(args.data, args.output))


if __name__ == "__main__":
    main()
