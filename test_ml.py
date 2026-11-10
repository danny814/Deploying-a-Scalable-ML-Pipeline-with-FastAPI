import os

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import compute_model_metrics, inference, train_model

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "census.csv")


@pytest.fixture(scope="module")
def data():
    """Load the census data once for all tests."""
    return pd.read_csv(DATA_PATH)


@pytest.fixture(scope="module")
def split_data(data):
    """Split the data the same way train_model.py does."""
    train, test = train_test_split(
        data, test_size=0.20, random_state=42, stratify=data["salary"]
    )
    return train, test


@pytest.fixture(scope="module")
def processed_data(split_data):
    """Process a small sample of the training data so the tests run quickly."""
    train, _ = split_data
    sample = train.sample(n=1000, random_state=42)
    X, y, encoder, lb = process_data(
        sample, categorical_features=CAT_FEATURES, label="salary", training=True
    )
    return X, y, encoder, lb


def test_train_model_uses_random_forest(processed_data):
    """
    train_model should return a fitted RandomForestClassifier.
    """
    X, y, _, _ = processed_data
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "estimators_")


def test_inference_returns_expected_type(processed_data):
    """
    inference should return a numpy array with one 0/1 prediction per row.
    """
    X, y, _, _ = processed_data
    model = train_model(X, y)
    preds = inference(model, X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X.shape[0]
    assert set(np.unique(preds)).issubset({0, 1})


def test_compute_model_metrics_expected_values():
    """
    compute_model_metrics should return the correct precision, recall and F1
    for a small hand-checked example.
    """
    y = np.array([1, 1, 0, 0])
    preds = np.array([1, 0, 1, 0])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == pytest.approx(0.5)
    assert recall == pytest.approx(0.5)
    assert fbeta == pytest.approx(0.5)


def test_train_test_split_size(data, split_data):
    """
    The train and test datasets should be an 80/20 split of the full dataset
    and keep every column.
    """
    train, test = split_data
    assert len(train) + len(test) == len(data)
    assert len(test) == pytest.approx(len(data) * 0.20, abs=1)
    assert list(train.columns) == list(data.columns)
    assert isinstance(train, pd.DataFrame)
