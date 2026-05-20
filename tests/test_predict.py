"""Unit tests for the inference module.

The trained pipeline contains a cuML KMeans estimator that requires a GPU,
so these tests mock the pipeline to keep them CPU-only and CI-friendly.
"""

import numpy as np
import pandas as pd
import pytest

from src.inference import predict


class _StubPipeline:
    """Minimal stand-in for the trained sklearn Pipeline used in inference."""

    def __init__(self, return_value=2):
        self._return_value = return_value
        self.calls = []

    def predict(self, X):
        self.calls.append(X)
        return np.array([self._return_value])


@pytest.fixture(autouse=True)
def reset_pipeline_cache():
    """Reset the lazy-loaded singleton between tests."""
    predict._pipeline = None
    yield
    predict._pipeline = None


def test_predict_returns_int(monkeypatch):
    stub = _StubPipeline(return_value=2)
    monkeypatch.setattr(predict, "_pipeline", stub)

    result = predict.predict_customer_segment(recency=30, frequency=5, monetary=250.0)

    assert isinstance(result, int)
    assert result == 2


def test_predict_passes_correct_dataframe(monkeypatch):
    stub = _StubPipeline()
    monkeypatch.setattr(predict, "_pipeline", stub)

    predict.predict_customer_segment(recency=10, frequency=3, monetary=99.5)

    assert len(stub.calls) == 1
    passed = stub.calls[0]
    assert isinstance(passed, pd.DataFrame)
    assert list(passed.columns) == ["Recency", "Frequency", "Monetary"]
    assert passed.iloc[0].tolist() == [10, 3, 99.5]


@pytest.mark.parametrize(
    "recency, frequency, monetary",
    [
        (-1, 5, 100.0),
        (10, -1, 100.0),
        (10, 5, -0.01),
        (-1, -1, -1),
    ],
)
def test_predict_rejects_negative_inputs(recency, frequency, monetary, monkeypatch):
    monkeypatch.setattr(predict, "_pipeline", _StubPipeline())

    with pytest.raises(ValueError, match="non-negative"):
        predict.predict_customer_segment(recency, frequency, monetary)


def test_predict_raises_if_model_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(predict, "MODEL_PATH", tmp_path / "does_not_exist.pkl")
    monkeypatch.setattr(predict, "_pipeline", None)

    with pytest.raises(FileNotFoundError, match="Trained pipeline not found"):
        predict.predict_customer_segment(10, 5, 100.0)
