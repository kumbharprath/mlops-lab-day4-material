"""Orchestrate generation, validation, training, evaluation, and saving."""
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

from scripts.create_dataset import create_dataset
from scripts.evaluate import evaluate_model
from scripts.model import FEATURES, train_model
from scripts.validate_data import validate_dataset


def main():
    df = create_dataset(n=1000, seed=42)
    validate_dataset(df)
    X, y = df[FEATURES], df["late_delivery"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = train_model(X_train, y_train, max_depth=5)
    metrics = evaluate_model(model, X_test, y_test)
    print("Model Performance:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    output = Path(__file__).resolve().parent.parent / "models" / "delivery_model.joblib"
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)
    print(f"Model saved: {output}")


if __name__ == "__main__":
    main()
