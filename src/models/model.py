"""DistilBERT-based 7-class emotion classifier.

Implements SentimentClassifier, a concrete subclass of BaseMLModel that
wraps HuggingFace AutoTokenizer and AutoModelForSequenceClassification.
All hyperparameters are read from config/config.yaml at construction time.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.exceptions import ConfigError, ModelNotFoundError, PredictionError
from src.logger import get_logger
from src.models.base import BaseMLModel

logger = get_logger(__name__)

_CONFIG_PATH = Path("config/config.yaml")


def _load_config() -> dict:
    """Read config/config.yaml and return the parsed dict.

    Raises:
        ConfigError: If the file is missing or cannot be parsed as YAML.
    """
    try:
        with _CONFIG_PATH.open() as fh:
            return yaml.safe_load(fh)
    except FileNotFoundError as exc:
        raise ConfigError(f"Config not found: {_CONFIG_PATH}") from exc


class SentimentClassifier(BaseMLModel):
    """DistilBERT-based emotion classifier for 7 macro-emotion classes.

    Wraps AutoTokenizer and AutoModelForSequenceClassification. Hyperparameters
    (base_model, num_labels, max_seq_len, emotion_labels) come exclusively
    from config/config.yaml — nothing is hardcoded.

    Attributes:
        config: Full parsed config/config.yaml dict.
        label_names: Ordered list of emotion label strings, indexed by class id.
        max_len: Maximum tokenizer sequence length from training config.
        device: torch.device — cuda if available, else cpu.
        tokenizer: HuggingFace AutoTokenizer instance.
        model: HuggingFace AutoModelForSequenceClassification instance.
    """

    def __init__(self) -> None:
        """Initialise tokenizer and model from config/config.yaml.

        Reads base_model, num_labels, and max_seq_len from config, then
        instantiates AutoTokenizer and AutoModelForSequenceClassification.
        Moves the model to the best available device.

        Raises:
            ConfigError: If config/config.yaml is missing or malformed.
        """
        self.config = _load_config()
        model_cfg: dict = self.config["model"]
        training_cfg: dict = self.config["training"]
        label_map: dict = self.config["emotion_labels"]

        self.label_names: list[str] = [label_map[i] for i in sorted(label_map)]
        self.max_len: int = training_cfg["max_seq_len"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        base_model: str = model_cfg["base_model"]
        num_labels: int = model_cfg["num_labels"]

        logger.info("Loading tokenizer: '%s'", base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)  # nosec B615

        logger.info(
            "Loading model '%s'  num_labels=%d  device=%s",
            base_model,
            num_labels,
            self.device,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(  # nosec B615
            base_model, num_labels=num_labels
        ).to(self.device)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _safe_inputs(self, texts: list[str]) -> list[str]:
        """Validate inputs before tokenization.

        Rejects empty lists, non-string items, whitespace-only strings,
        and strings whose stripped form evaluates to NaN as a float.

        Args:
            texts: Raw list of input strings from the caller.

        Returns:
            The same list if all inputs pass validation.

        Raises:
            PredictionError: If the list is empty, any element is not a
                string, whitespace-only, or is a NaN-like value.
        """
        if not texts:
            raise PredictionError("texts must be a non-empty list")

        for t in texts:
            if not isinstance(t, str):
                raise PredictionError(
                    f"All inputs must be strings; got {type(t).__name__}"
                )
            stripped = t.strip()
            if stripped == "" or stripped.lower() == "nan":
                raise PredictionError(f"Empty or NaN-like string in texts: {t!r}")
            try:
                if math.isnan(float(stripped)):
                    raise PredictionError(f"NaN value in texts: {t!r}")
            except ValueError:
                pass  # not float-parseable — fine

        return texts

    def _forward(self, texts: list[str]) -> torch.Tensor:
        """Tokenize inputs and return raw logits from the model.

        Args:
            texts: Validated list of non-empty strings.

        Returns:
            Float tensor of shape (len(texts), num_labels) with raw logits.
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        return outputs.logits  # shape: (batch_size, num_labels)

    # ------------------------------------------------------------------
    # BaseMLModel interface
    # ------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        """Not implemented — training is handled by src/training/train.py.

        Args:
            train_df: Unused training DataFrame.
            val_df: Unused validation DataFrame.

        Raises:
            NotImplementedError: Always. Use ``make train`` to fine-tune.
        """
        raise NotImplementedError(
            "SentimentClassifier.fit() is not used. "
            "Run 'make train' to fine-tune the model."
        )

    def predict(self, texts: list[str]) -> list[str]:  # type: ignore[override]
        """Return the predicted emotion label name for each input text.

        Args:
            texts: Non-empty list of raw input strings. Must not contain
                   NaN-like or whitespace-only entries.

        Returns:
            List of emotion label strings (e.g. ['joy', 'anger']) aligned
            position-for-position with the input list.

        Raises:
            PredictionError: If texts is empty, contains invalid entries,
                             or the model forward pass raises an exception.
        """
        cleaned = self._safe_inputs(texts)
        try:
            logits = self._forward(cleaned)
            indices = logits.argmax(dim=-1).tolist()
            return [self.label_names[i] for i in indices]
        except PredictionError:
            raise
        except Exception as exc:
            raise PredictionError(f"Inference failed: {exc}") from exc

    def predict_proba(  # type: ignore[override]
        self, texts: list[str]
    ) -> list[dict[str, float]]:
        """Return per-class softmax probabilities for each input text.

        Args:
            texts: Non-empty list of raw input strings. Must not contain
                   NaN-like or whitespace-only entries.

        Returns:
            List of dicts mapping each emotion label to its probability.
            Each dict sums to 1.0. One dict per input text.

        Raises:
            PredictionError: If texts is empty, contains invalid entries,
                             or the model forward pass raises an exception.
        """
        cleaned = self._safe_inputs(texts)
        try:
            logits = self._forward(cleaned)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            return [
                {label: float(p) for label, p in zip(self.label_names, row)}
                for row in probs
            ]
        except PredictionError:
            raise
        except Exception as exc:
            raise PredictionError(f"Probability estimation failed: {exc}") from exc

    def save(self, path: Path) -> None:
        """Persist model weights and tokenizer to disk.

        Calls model.save_pretrained and tokenizer.save_pretrained, writing
        config.json, model weights, and tokenizer vocab files under path.

        Args:
            path: Destination directory. Created if it does not exist.
        """
        dest = Path(path)
        dest.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(dest))
        self.tokenizer.save_pretrained(str(dest))
        logger.info("Model + tokenizer saved to '%s'", dest)

    def load(self, path: Path) -> "SentimentClassifier":
        """Load model weights and tokenizer from a previously saved directory.

        Replaces the current tokenizer and model in-place, moves to device,
        and switches the model to eval mode so dropout is disabled.

        Args:
            path: Directory written by a previous call to :meth:`save`.

        Returns:
            self — the mutated instance ready for inference.

        Raises:
            ModelNotFoundError: If path does not exist on disk.
        """
        src_path = Path(path)
        if not src_path.exists():
            raise ModelNotFoundError(f"Model artefacts not found at '{src_path}'")

        logger.info("Loading model + tokenizer from '%s'", src_path)
        self.tokenizer = AutoTokenizer.from_pretrained(str(src_path))  # nosec B615
        num_labels: int = self.config["model"]["num_labels"]
        self.model = AutoModelForSequenceClassification.from_pretrained(  # nosec B615
            str(src_path), num_labels=num_labels
        ).to(self.device)
        self.model.eval()
        logger.info("Model loaded — eval mode — device: %s", self.device)
        return self
