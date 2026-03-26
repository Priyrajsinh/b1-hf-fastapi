"""Tests for src/data/dataset.py.

All tests mock load_dataset to avoid network calls.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers — fake HuggingFace Dataset object
# ---------------------------------------------------------------------------


class _FakeDataset:
    """Minimal stand-in for a HuggingFace Dataset slice."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def filter(self, fn):  # noqa: D102
        return _FakeDataset([r for r in self._rows if fn(r)])

    def __iter__(self):  # noqa: D105
        return iter(self._rows)

    def __len__(self) -> int:  # noqa: D105
        return len(self._rows)


_FAKE_ROWS = [
    {"text": "I am so happy today", "labels": [14]},  # joy -> 0
    {"text": "This makes me very sad", "labels": [25]},  # sadness -> 1
    {"text": "I am furious right now", "labels": [3]},  # anger -> 2
    {"text": "That scared me badly", "labels": [13]},  # fear -> 3
    {"text": "What a big surprise", "labels": [24]},  # surprise -> 4
    {"text": "That is truly disgusting", "labels": [12]},  # disgust -> 5
    {"text": "Nothing special happened", "labels": [27]},  # neutral -> 6
    {"text": "Multi label row", "labels": [14, 25]},  # filtered out
]

_FAKE_DS = {"train": _FakeDataset(_FAKE_ROWS), "test": _FakeDataset(_FAKE_ROWS[:3])}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@patch("src.data.dataset.load_dataset", return_value=_FAKE_DS)
def test_load_goemotions_columns(mock_ld, tmp_path, monkeypatch):
    """Returned DataFrame must have exactly ['text', 'label'] columns."""
    import src.data.dataset as _mod

    monkeypatch.setattr(_mod, "_CHECKSUM_PATH", tmp_path / "checksums.json")
    monkeypatch.setattr(_mod, "_CONFIG_PATH", _mod._CONFIG_PATH)  # keep real config

    from src.data.dataset import load_goemotions

    df = load_goemotions()
    assert list(df.columns) == ["text", "label"]


@patch("src.data.dataset.load_dataset", return_value=_FAKE_DS)
def test_load_goemotions_label_range(mock_ld, tmp_path, monkeypatch):
    """All labels must be integers in {0, 1, 2, 3, 4, 5, 6}."""
    import src.data.dataset as _mod

    monkeypatch.setattr(_mod, "_CHECKSUM_PATH", tmp_path / "checksums.json")

    from src.data.dataset import load_goemotions

    df = load_goemotions()
    assert df["label"].between(0, 6).all()
    assert df["label"].dtype == int or str(df["label"].dtype).startswith("int")


@patch("src.data.dataset.load_dataset", return_value=_FAKE_DS)
def test_load_goemotions_filters_multilabel(mock_ld, tmp_path, monkeypatch):
    """Multi-label rows must be excluded from the result."""
    import src.data.dataset as _mod

    monkeypatch.setattr(_mod, "_CHECKSUM_PATH", tmp_path / "checksums.json")

    from src.data.dataset import load_goemotions

    df = load_goemotions()
    # The multi-label row has text "Multi label row" — must not appear
    assert "Multi label row" not in df["text"].values


@patch("src.data.dataset.load_dataset", return_value=_FAKE_DS)
def test_load_goemotions_saves_checksum(mock_ld, tmp_path, monkeypatch):
    """load_goemotions must write a 'goemotions_full' key to checksums.json."""
    import src.data.dataset as _mod

    checksum_path = tmp_path / "checksums.json"
    monkeypatch.setattr(_mod, "_CHECKSUM_PATH", checksum_path)

    from src.data.dataset import load_goemotions

    load_goemotions()
    assert checksum_path.exists()
    data = json.loads(checksum_path.read_text())
    assert "goemotions_full" in data
    assert len(data["goemotions_full"]) == 64  # SHA-256 hex digest length


@patch("src.data.dataset.load_dataset", return_value=_FAKE_DS)
def test_load_goemotions_raises_on_schema_failure(mock_ld, tmp_path, monkeypatch):
    """DataLoadError must be raised when EMOTION_SCHEMA validation fails."""
    from pandera.errors import SchemaError

    import src.data.dataset as _mod
    from src.exceptions import DataLoadError

    monkeypatch.setattr(_mod, "_CHECKSUM_PATH", tmp_path / "checksums.json")

    with patch("src.data.dataset.EMOTION_SCHEMA") as mock_schema:
        mock_schema.validate.side_effect = SchemaError(
            MagicMock(), MagicMock(), MagicMock()
        )
        with pytest.raises(DataLoadError):
            from src.data.dataset import load_goemotions as _fn

            _fn()


@patch("src.data.dataset.load_dataset", return_value=_FAKE_DS)
def test_load_goemotions_no_empty_text(mock_ld, tmp_path, monkeypatch):
    """No row should have an empty or whitespace-only text value."""
    import src.data.dataset as _mod

    monkeypatch.setattr(_mod, "_CHECKSUM_PATH", tmp_path / "checksums.json")

    from src.data.dataset import load_goemotions

    df = load_goemotions()
    assert df["text"].str.strip().str.len().gt(0).all()
