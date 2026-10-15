"""Compute held-out classification metrics."""
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
    }
