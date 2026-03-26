"""Load GoEmotions and return a validated DataFrame with 7 macro-labels.

Public API:
    load_goemotions() -> pd.DataFrame  # columns: ['text', 'label']
"""

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml
from datasets import load_dataset
from pandera.errors import SchemaError

from src.data.load_raw import GOEMOTION_TO_MACRO, MACRO_LABEL_NAMES
from src.data.validation import EMOTION_SCHEMA
from src.exceptions import ConfigError, DataLoadError
from src.logger import get_logger

logger = get_logger(__name__)

_CONFIG_PATH = Path("config/config.yaml")
_CHECKSUM_PATH = Path("data/raw/checksums.json")


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


def load_goemotions() -> pd.DataFrame:
    """Load GoEmotions simplified split and return a validated DataFrame.

    Combines all HuggingFace splits, filters to single-label rows, maps
    the 27 fine-grained labels to 7 macro-categories, computes a SHA-256
    checksum, validates against EMOTION_SCHEMA, and logs statistics.

    Returns:
        pd.DataFrame with columns ['text', 'label'] where label in {0..6}.

    Raises:
        DataLoadError: If EMOTION_SCHEMA validation fails.
    """
    _load_config()  # fail fast if config is broken

    logger.info("Loading google-research-datasets/go_emotions (simplified)")
    ds = load_dataset(  # nosec B615
        "google-research-datasets/go_emotions", "simplified"
    )

    frames: list[pd.DataFrame] = []
    for split_name, split_data in ds.items():
        single = split_data.filter(lambda row: len(row["labels"]) == 1)
        logger.info(
            "Split '%s': %d / %d samples kept (single-label filter)",
            split_name,
            len(single),
            len(split_data),
        )
        frames.append(
            pd.DataFrame(
                {
                    "text": [r["text"] for r in single],
                    "label": [GOEMOTION_TO_MACRO[r["labels"][0]] for r in single],
                }
            )
        )

    df = pd.concat(frames, ignore_index=True)

    _save_checksum(df)
    _log_distribution(df)
    _log_text_stats(df)

    try:
        EMOTION_SCHEMA.validate(df)
    except SchemaError as exc:
        raise DataLoadError(f"EMOTION_SCHEMA validation failed: {exc}") from exc

    return df


def _save_checksum(df: pd.DataFrame) -> None:
    """Compute SHA-256 of the full DataFrame CSV and persist to JSON."""
    _CHECKSUM_PATH.parent.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(df.to_csv().encode()).hexdigest()
    checksums: dict = {}
    if _CHECKSUM_PATH.exists():
        with _CHECKSUM_PATH.open() as fh:
            checksums = json.load(fh)
    checksums["goemotions_full"] = sha
    with _CHECKSUM_PATH.open("w") as fh:
        json.dump(checksums, fh, indent=2)
    logger.info("SHA-256 saved to %s  —  %s", _CHECKSUM_PATH, sha)


def _log_distribution(df: pd.DataFrame) -> None:
    """Log sample count and percentage per macro-class."""
    total = len(df)
    counts = df["label"].value_counts().sort_index()
    for idx, count in counts.items():
        logger.info(
            "Class %s (%d): %d samples (%.1f%%)",
            MACRO_LABEL_NAMES[int(idx)],
            idx,
            count,
            100.0 * count / total,
        )


def _log_text_stats(df: pd.DataFrame) -> None:
    """Log descriptive statistics for raw text character lengths."""
    lengths = df["text"].str.len()
    logger.info(
        "Text length — mean: %.1f  std: %.1f  min: %d  max: %d  p95: %d",
        lengths.mean(),
        lengths.std(),
        lengths.min(),
        lengths.max(),
        int(lengths.quantile(0.95)),
    )
