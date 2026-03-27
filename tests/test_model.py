"""Tests for SentimentClassifier in src/models/model.py.

Uses unittest.mock to patch AutoTokenizer and
AutoModelForSequenceClassification so no network calls are made and the
test suite runs in under 5 seconds.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.exceptions import ConfigError, ModelNotFoundError, PredictionError
from src.models.model import SentimentClassifier, _load_config

_NUM_LABELS = 7


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_mock_tokenizer() -> MagicMock:
    """Return a mock tokenizer that produces dummy input tensors."""
    tok = MagicMock()
    tok.return_value = {
        "input_ids": torch.zeros(1, 10, dtype=torch.long),
        "attention_mask": torch.ones(1, 10, dtype=torch.long),
    }
    return tok


def _make_mock_model(num_labels: int = _NUM_LABELS) -> MagicMock:
    """Return a mock model whose forward pass returns deterministic logits."""
    model = MagicMock()
    model.to.return_value = model  # support .to(device) chaining

    logits = torch.zeros(1, num_labels)
    logits[0, 0] = 5.0  # class 0 (joy) wins softmax
    output = MagicMock()
    output.logits = logits
    model.return_value = output
    return model


@pytest.fixture()
def classifier() -> SentimentClassifier:
    """SentimentClassifier with fully mocked HuggingFace model + tokenizer."""
    mock_tok = _make_mock_tokenizer()
    mock_model = _make_mock_model()
    with (
        patch(
            "src.models.model.AutoTokenizer.from_pretrained",
            return_value=mock_tok,
        ),
        patch(
            "src.models.model.AutoModelForSequenceClassification.from_pretrained",
            return_value=mock_model,
        ),
    ):
        clf = SentimentClassifier()
    return clf


# ---------------------------------------------------------------------------
# _load_config
# ---------------------------------------------------------------------------


def test_load_config_raises_on_missing_file(monkeypatch):
    """_load_config raises ConfigError when config file does not exist."""
    import src.models.model as mm

    monkeypatch.setattr(mm, "_CONFIG_PATH", Path("nonexistent_dir/config.yaml"))
    with pytest.raises(ConfigError, match="Config not found"):
        mm._load_config()


def test_load_config_returns_dict():
    """_load_config returns a dict when config/config.yaml is present."""
    cfg = _load_config()
    assert isinstance(cfg, dict)
    assert "model" in cfg
    assert "training" in cfg
    assert "emotion_labels" in cfg


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


def test_classifier_has_label_names(classifier):
    """SentimentClassifier.label_names is a 7-element list of strings."""
    assert len(classifier.label_names) == _NUM_LABELS
    assert all(isinstance(name, str) for name in classifier.label_names)


def test_classifier_label_names_order(classifier):
    """label_names are ordered by config key (joy=0, neutral=6)."""
    assert classifier.label_names[0] == "joy"
    assert classifier.label_names[-1] == "neutral"


# ---------------------------------------------------------------------------
# _safe_inputs
# ---------------------------------------------------------------------------


def test_safe_inputs_empty_list_raises(classifier):
    with pytest.raises(PredictionError, match="non-empty"):
        classifier._safe_inputs([])


def test_safe_inputs_whitespace_raises(classifier):
    with pytest.raises(PredictionError, match="Empty or NaN-like"):
        classifier._safe_inputs(["   "])


def test_safe_inputs_nan_string_raises(classifier):
    with pytest.raises(PredictionError, match="Empty or NaN-like"):
        classifier._safe_inputs(["nan"])


def test_safe_inputs_nan_case_insensitive(classifier):
    with pytest.raises(PredictionError, match="Empty or NaN-like"):
        classifier._safe_inputs(["NaN"])


def test_safe_inputs_non_string_raises(classifier):
    with pytest.raises(PredictionError, match="must be strings"):
        classifier._safe_inputs([123])  # type: ignore[list-item]


def test_safe_inputs_valid_passes_through(classifier):
    texts = ["I love this!", "what a sad day"]
    result = classifier._safe_inputs(texts)
    assert result == texts


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


def test_predict_returns_list_of_strings(classifier):
    result = classifier.predict(["I am very happy today!"])
    assert isinstance(result, list)
    assert all(isinstance(lbl, str) for lbl in result)


def test_predict_returns_correct_label(classifier):
    """Mock logits set class 0 (joy) highest — predict must return 'joy'."""
    result = classifier.predict(["great news!"])
    assert result == ["joy"]


def test_predict_empty_input_raises(classifier):
    with pytest.raises(PredictionError):
        classifier.predict([])


def test_predict_nan_input_raises(classifier):
    with pytest.raises(PredictionError):
        classifier.predict(["nan"])


# ---------------------------------------------------------------------------
# predict_proba
# ---------------------------------------------------------------------------


def test_predict_proba_returns_list_of_dicts(classifier):
    result = classifier.predict_proba(["hello world"])
    assert isinstance(result, list)
    assert isinstance(result[0], dict)


def test_predict_proba_all_labels_present(classifier):
    expected = {"joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"}
    result = classifier.predict_proba(["test input"])
    assert set(result[0].keys()) == expected


def test_predict_proba_sums_to_one(classifier):
    result = classifier.predict_proba(["test input"])
    assert abs(sum(result[0].values()) - 1.0) < 1e-5


def test_predict_proba_empty_raises(classifier):
    with pytest.raises(PredictionError):
        classifier.predict_proba([])


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------


def test_fit_raises_not_implemented(classifier):
    with pytest.raises(NotImplementedError):
        classifier.fit(None, None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------


def test_save_creates_directory(classifier, tmp_path):
    dest = tmp_path / "output_model"
    classifier.save(dest)
    assert dest.exists()


def test_save_calls_save_pretrained(classifier, tmp_path):
    dest = tmp_path / "output_model"
    classifier.save(dest)
    classifier.model.save_pretrained.assert_called_once_with(str(dest))
    classifier.tokenizer.save_pretrained.assert_called_once_with(str(dest))


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------


def test_load_raises_if_path_missing(classifier, tmp_path):
    with pytest.raises(ModelNotFoundError, match="not found"):
        classifier.load(tmp_path / "does_not_exist")


def test_load_returns_self(classifier, tmp_path):
    model_dir = tmp_path / "saved_model"
    model_dir.mkdir()
    mock_tok = _make_mock_tokenizer()
    mock_model = _make_mock_model()
    with (
        patch(
            "src.models.model.AutoTokenizer.from_pretrained",
            return_value=mock_tok,
        ),
        patch(
            "src.models.model.AutoModelForSequenceClassification.from_pretrained",
            return_value=mock_model,
        ),
    ):
        result = classifier.load(model_dir)
    assert result is classifier


def test_load_sets_eval_mode(classifier, tmp_path):
    """load() must call .eval() on the loaded model."""
    model_dir = tmp_path / "saved_model"
    model_dir.mkdir()
    mock_tok = _make_mock_tokenizer()
    mock_model = _make_mock_model()
    with (
        patch(
            "src.models.model.AutoTokenizer.from_pretrained",
            return_value=mock_tok,
        ),
        patch(
            "src.models.model.AutoModelForSequenceClassification.from_pretrained",
            return_value=mock_model,
        ),
    ):
        classifier.load(model_dir)
    mock_model.eval.assert_called_once()
