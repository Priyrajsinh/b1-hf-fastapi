"""Tests for src/api/app.py.

Patches SentimentClassifier so no model weights are needed.
Uses FastAPI TestClient (synchronous ASGI transport via httpx).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROBA = {
    "joy": 0.80,
    "sadness": 0.05,
    "anger": 0.03,
    "fear": 0.03,
    "surprise": 0.03,
    "disgust": 0.03,
    "neutral": 0.03,
}


def _make_mock_classifier() -> MagicMock:
    clf = MagicMock()
    clf.predict_proba.return_value = [_PROBA]
    return clf


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    """Return a TestClient with the model loading patched out."""
    mock_clf = _make_mock_classifier()

    with patch("src.api.app.SentimentClassifier", return_value=mock_clf):
        # Import app after patch so startup uses the mock
        import src.api.app as api_module

        api_module.classifier = mock_clf
        api_module.model_loaded = True

        with TestClient(api_module.app, raise_server_exceptions=False) as c:
            yield c


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health_ok(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_loaded"] is True
    assert body["status"] == "ok"
    assert "uptime_seconds" in body
    assert "memory_mb" in body


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------


def test_predict_returns_sentiment_output(client):
    resp = client.post("/api/v1/predict", json={"text": "I am so happy!"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] == "joy"
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["probabilities"].keys()) == set(_PROBA.keys())
    assert "trace_id" in body


def test_predict_empty_text_rejected(client):
    resp = client.post("/api/v1/predict", json={"text": "   "})
    assert resp.status_code == 422


def test_predict_missing_text_field_rejected(client):
    resp = client.post("/api/v1/predict", json={})
    assert resp.status_code == 422


def test_predict_trace_id_unique(client):
    r1 = client.post("/api/v1/predict", json={"text": "hello"})
    r2 = client.post("/api/v1/predict", json={"text": "hello"})
    assert r1.json()["trace_id"] != r2.json()["trace_id"]


# ---------------------------------------------------------------------------
# Model info
# ---------------------------------------------------------------------------


def test_model_info_keys(client):
    resp = client.get("/api/v1/model_info")
    assert resp.status_code == 200
    body = resp.json()
    assert "model" in body
    assert "num_labels" in body
    assert body["num_labels"] == 7


# ---------------------------------------------------------------------------
# Docs / metrics
# ---------------------------------------------------------------------------


def test_docs_available(client):
    assert client.get("/docs").status_code == 200


def test_metrics_available(client):
    assert client.get("/metrics").status_code == 200
