"""Stratified splitting and HuggingFace tokenization utilities.

Public API:
    stratified_split(df, train, val, test, seed) -> tuple[pd.DataFrame, ...]
    tokenize_dataset(df, tokenizer, max_len) -> datasets.Dataset
"""

from pathlib import Path
from typing import Any

import datasets
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

from src.data.load_raw import MACRO_LABEL_NAMES
from src.exceptions import ConfigError
from src.logger import get_logger

logger = get_logger(__name__)

_CONFIG_PATH = Path("config/config.yaml")
_PROCESSED_DIR = Path("data/processed")


def _load_config() -> dict:
    """Read config/config.yaml and return the parsed dict.

    Raises:
        ConfigError: If the file is missing or cannot be parsed.
    """
    try:
        with _CONFIG_PATH.open() as fh:
            return yaml.safe_load(fh)
    except FileNotFoundError as exc:
        raise ConfigError(f"Config not found: {_CONFIG_PATH}") from exc


def stratified_split(
    df: pd.DataFrame,
    train: float = 0.80,
    val: float = 0.10,
    test: float = 0.10,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split df into stratified train/val/test subsets.

    Uses two StratifiedShuffleSplit passes to preserve class ratios in all
    three splits.  Asserts zero index overlap before saving CSVs.

    Args:
        df: Full DataFrame with 'text' and 'label' columns.
        train: Fraction for training set (default 0.80).
        val: Fraction for validation set (default 0.10).
        test: Fraction for test set (default 0.10).
        seed: Random seed for reproducibility (default 42).

    Returns:
        Tuple (train_df, val_df, test_df), each with reset index.
    """
    assert abs(train + val + test - 1.0) < 1e-9, "Fractions must sum to 1.0"

    # Pass 1: carve out test set
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test, random_state=seed)
    remainder_pos, test_pos = next(sss1.split(df, df["label"]))
    test_df = df.iloc[test_pos]
    remainder_df = df.iloc[
        remainder_pos
    ]  # original index preserved — needed for overlap check

    # Pass 2: carve val from the train+val remainder
    val_of_remainder = val / (1.0 - test)
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=val_of_remainder, random_state=seed
    )
    train_pos, val_pos = next(sss2.split(remainder_df, remainder_df["label"]))
    train_df = remainder_df.iloc[train_pos]
    val_df = remainder_df.iloc[val_pos]

    # Zero-overlap assertion across all three splits
    assert len(set(train_df.index) & set(val_df.index)) == 0, "Train/val overlap"
    assert len(set(train_df.index) & set(test_df.index)) == 0, "Train/test overlap"
    assert len(set(val_df.index) & set(test_df.index)) == 0, "Val/test overlap"

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    logger.info(
        "Split sizes — train: %d  val: %d  test: %d  (total %d)",
        len(train_df),
        len(val_df),
        len(test_df),
        len(df),
    )
    _log_class_weights(train_df)
    _save_splits(train_df, val_df, test_df)
    return train_df, val_df, test_df


def _log_class_weights(train_df: pd.DataFrame) -> None:
    """Log balanced class weights derived from the training split."""
    classes = np.arange(7)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_df["label"].to_numpy(),
    )
    for cls_idx, weight in zip(classes, weights):
        logger.info(
            "Class weight [%d] %s: %.4f",
            cls_idx,
            MACRO_LABEL_NAMES[int(cls_idx)],
            weight,
        )


def _save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Persist split CSVs to data/processed/."""
    _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(_PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(_PROCESSED_DIR / "val.csv", index=False)
    test_df.to_csv(_PROCESSED_DIR / "test.csv", index=False)
    logger.info("Splits saved to %s", _PROCESSED_DIR)


def tokenize_dataset(
    df: pd.DataFrame,
    tokenizer: Any,
    max_len: int = 128,
) -> datasets.Dataset:
    """Tokenize a DataFrame and return a HuggingFace Dataset in torch format.

    Args:
        df: DataFrame with 'text' and 'label' columns.
        tokenizer: HuggingFace tokenizer (e.g. from AutoTokenizer.from_pretrained).
        max_len: Maximum token sequence length; pads and truncates to this value.

    Returns:
        datasets.Dataset with torch tensors for input_ids, attention_mask, label.
    """
    hf_dataset = datasets.Dataset.from_pandas(df, preserve_index=False)

    def _tokenize(batch: dict) -> dict:
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_len,
        )

    hf_dataset = hf_dataset.map(_tokenize, batched=True)
    hf_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    logger.info("Tokenized dataset: %d samples  max_len=%d", len(hf_dataset), max_len)
    return hf_dataset
