"""Sanity tests for Decision Tree training and prediction."""
from scripts.create_dataset import create_dataset
from scripts.model import FEATURES, train_model


def test_training_returns_model():
    df = create_dataset(n=200)
    model = train_model(df[FEATURES], df["late_delivery"])
    assert model is not None


def test_prediction_count_matches_rows():
    df = create_dataset(n=200)
    model = train_model(df[FEATURES], df["late_delivery"])
    assert len(model.predict(df[FEATURES].head(10))) == 10


def test_predictions_are_binary():
    df = create_dataset(n=200)
    model = train_model(df[FEATURES], df["late_delivery"])
    assert set(model.predict(df[FEATURES])).issubset({0, 1})
