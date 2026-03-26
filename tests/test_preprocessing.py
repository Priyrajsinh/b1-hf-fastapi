"""Tests for src/data/preprocessing.py.

stratified_split uses synthetic data; tokenize_dataset uses a mock tokenizer.
"""

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Shared fixture — synthetic DataFrame with all 7 classes
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_df() -> pd.DataFrame:
    """700-row DataFrame with 100 samples per class (perfectly balanced)."""
    rng = np.random.default_rng(42)
    labels = np.repeat(np.arange(7), 100)
    rng.shuffle(labels)
    texts = [f"sample text number {i}" for i in range(len(labels))]
    return pd.DataFrame({"text": texts, "label": labels})


# ---------------------------------------------------------------------------
# stratified_split tests
# ---------------------------------------------------------------------------


def test_split_sizes(synthetic_df, tmp_path, monkeypatch):
    """train + val + test must equal total rows."""
    import src.data.preprocessing as _mod

    monkeypatch.setattr(_mod, "_PROCESSED_DIR", tmp_path)

    from src.data.preprocessing import stratified_split

    train, val, test = stratified_split(synthetic_df)
    assert len(train) + len(val) + len(test) == len(synthetic_df)


def test_split_no_overlap(synthetic_df, tmp_path, monkeypatch):
    """Zero rows may appear in more than one split."""
    import src.data.preprocessing as _mod

    monkeypatch.setattr(_mod, "_PROCESSED_DIR", tmp_path)

    from src.data.preprocessing import stratified_split

    train, val, test = stratified_split(synthetic_df)
    train_texts = set(train["text"])
    val_texts = set(val["text"])
    test_texts = set(test["text"])

    assert len(train_texts & val_texts) == 0
    assert len(train_texts & test_texts) == 0
    assert len(val_texts & test_texts) == 0


def test_split_approximate_proportions(synthetic_df, tmp_path, monkeypatch):
    """Each split size should be within 5% of the target fraction."""
    import src.data.preprocessing as _mod

    monkeypatch.setattr(_mod, "_PROCESSED_DIR", tmp_path)

    from src.data.preprocessing import stratified_split

    total = len(synthetic_df)
    train, val, test = stratified_split(synthetic_df, train=0.80, val=0.10, test=0.10)

    assert abs(len(train) / total - 0.80) < 0.05
    assert abs(len(val) / total - 0.10) < 0.05
    assert abs(len(test) / total - 0.10) < 0.05


def test_split_saves_csvs(synthetic_df, tmp_path, monkeypatch):
    """train.csv, val.csv, test.csv must be written to the processed dir."""
    import src.data.preprocessing as _mod

    monkeypatch.setattr(_mod, "_PROCESSED_DIR", tmp_path)

    from src.data.preprocessing import stratified_split

    stratified_split(synthetic_df)

    assert (tmp_path / "train.csv").exists()
    assert (tmp_path / "val.csv").exists()
    assert (tmp_path / "test.csv").exists()


def test_split_csv_has_correct_columns(synthetic_df, tmp_path, monkeypatch):
    """Saved CSVs must contain exactly 'text' and 'label' columns."""
    import src.data.preprocessing as _mod

    monkeypatch.setattr(_mod, "_PROCESSED_DIR", tmp_path)

    from src.data.preprocessing import stratified_split

    stratified_split(synthetic_df)

    for fname in ("train.csv", "val.csv", "test.csv"):
        saved = pd.read_csv(tmp_path / fname)
        assert list(saved.columns) == ["text", "label"]


def test_split_stratification(synthetic_df, tmp_path, monkeypatch):
    """Each split must contain all 7 classes."""
    import src.data.preprocessing as _mod

    monkeypatch.setattr(_mod, "_PROCESSED_DIR", tmp_path)

    from src.data.preprocessing import stratified_split

    train, val, test = stratified_split(synthetic_df)

    for df in (train, val, test):
        assert set(df["label"].unique()) == set(range(7))


def test_split_fractions_must_sum_to_one(synthetic_df, tmp_path, monkeypatch):
    """Passing fractions that don't sum to 1.0 must raise AssertionError."""
    import src.data.preprocessing as _mod

    monkeypatch.setattr(_mod, "_PROCESSED_DIR", tmp_path)

    from src.data.preprocessing import stratified_split

    with pytest.raises(AssertionError):
        stratified_split(synthetic_df, train=0.70, val=0.10, test=0.10)


# ---------------------------------------------------------------------------
# tokenize_dataset tests
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Minimal mock tokenizer that returns fixed-length token sequences."""

    def __call__(
        self,
        texts: list[str],
        padding: str,
        truncation: bool,
        max_length: int,
    ) -> dict:
        n = len(texts)
        return {
            "input_ids": [[101] + [0] * (max_length - 1)] * n,
            "attention_mask": [[1] * max_length] * n,
            "token_type_ids": [[0] * max_length] * n,  # excluded by set_format
        }


@pytest.fixture()
def small_df() -> pd.DataFrame:
    """Tiny 6-row DataFrame for tokenization tests."""
    return pd.DataFrame(
        {
            "text": [f"sentence {i}" for i in range(6)],
            "label": [0, 1, 2, 3, 4, 5],
        }
    )


def test_tokenize_returns_dataset(small_df):
    """tokenize_dataset must return a datasets.Dataset instance."""
    import datasets

    from src.data.preprocessing import tokenize_dataset

    result = tokenize_dataset(small_df, _FakeTokenizer(), max_len=16)
    assert isinstance(result, datasets.Dataset)


def test_tokenize_row_count(small_df):
    """Output dataset must have the same number of rows as the input DataFrame."""
    from src.data.preprocessing import tokenize_dataset

    result = tokenize_dataset(small_df, _FakeTokenizer(), max_len=16)
    assert len(result) == len(small_df)


def test_tokenize_expected_columns(small_df):
    """Dataset must expose input_ids, attention_mask, and label after set_format."""
    from src.data.preprocessing import tokenize_dataset

    result = tokenize_dataset(small_df, _FakeTokenizer(), max_len=16)
    assert "input_ids" in result.column_names
    assert "attention_mask" in result.column_names
    assert "label" in result.column_names


def test_tokenize_token_type_ids_excluded(small_df):
    """token_type_ids must not appear in the formatted output columns."""
    from src.data.preprocessing import tokenize_dataset

    result = tokenize_dataset(small_df, _FakeTokenizer(), max_len=16)
    # set_format restricts which columns are returned as tensors
    assert result.format["columns"] == ["input_ids", "attention_mask", "label"]
