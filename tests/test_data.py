"""Automated dataset-validation examples."""
import pandas as pd
import pytest

from scripts.create_dataset import create_dataset
from scripts.validate_data import validate_dataset


def test_generated_dataset_is_valid():
    assert validate_dataset(create_dataset(n=100)) is True


def test_missing_required_column_is_rejected():
    df = create_dataset(n=100).drop(columns=["raining"])
    with pytest.raises(ValueError, match="Missing columns"):
        validate_dataset(df)


def test_negative_distance_is_rejected():
    df = create_dataset(n=100)
    df.loc[0, "distance_km"] = -1.0
    with pytest.raises(ValueError, match="Negative distance"):
        validate_dataset(df)


def test_non_binary_traffic_is_rejected():
    df = create_dataset(n=100)
    df.loc[0, "heavy_traffic"] = 3
    with pytest.raises(ValueError, match="Invalid values"):
        validate_dataset(df)


def test_missing_value_is_rejected():
    df = create_dataset(n=100)
    df.loc[0, "raining"] = None
    with pytest.raises(ValueError, match="missing values"):
        validate_dataset(df)
