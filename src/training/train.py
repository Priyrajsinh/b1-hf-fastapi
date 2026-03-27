"""Fine-tune DistilBERT on GoEmotions 7-class macro-emotion splits.

Loads processed train/val CSVs, validates them with EMOTION_SCHEMA, tokenizes
with the classifier's own tokenizer, applies balanced class weights via a
custom WeightedTrainer, logs every epoch to MLflow, and saves the best
checkpoint plus a training_stats.json artefact.

Usage:
    python src/training/train.py --config config/config.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from pandera.errors import SchemaError
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    EvalPrediction,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    set_seed,
)

from src.data.preprocessing import tokenize_dataset
from src.data.validation import EMOTION_SCHEMA
from src.exceptions import ConfigError, DataLoadError
from src.logger import get_logger
from src.models.model import SentimentClassifier

logger = get_logger(__name__)

_PROCESSED_DIR = Path("data/processed")
_MODEL_OUTPUT_DIR = Path("models/sentiment_model")
_STATS_PATH = Path("models/training_stats.json")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace with a ``config`` Path attribute.
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT on GoEmotions macro-emotion splits"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to config YAML (default: config/config.yaml)",
    )
    return parser.parse_args()


def _load_config(config_path: Path) -> dict:
    """Read the YAML config file and return it as a dict.

    Args:
        config_path: Path to config/config.yaml.

    Returns:
        Parsed config dict.

    Raises:
        ConfigError: If the file is missing or cannot be parsed.
    """
    try:
        with config_path.open() as fh:
            return yaml.safe_load(fh)
    except FileNotFoundError as exc:
        raise ConfigError(f"Config not found: {config_path}") from exc


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _load_split(name: str) -> pd.DataFrame:
    """Load and validate a processed CSV split.

    Args:
        name: Split name — ``'train'``, ``'val'``, or ``'test'``.

    Returns:
        Validated DataFrame with ``text`` (str) and ``label`` (int 0-6) columns.

    Raises:
        DataLoadError: If the CSV file is missing or fails EMOTION_SCHEMA.
    """
    path = _PROCESSED_DIR / f"{name}.csv"
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise DataLoadError(
            f"Processed split not found: {path}. Run the data pipeline first."
        ) from exc

    try:
        EMOTION_SCHEMA.validate(df)
    except SchemaError as exc:
        raise DataLoadError(f"Schema validation failed for '{name}': {exc}") from exc

    logger.info("Loaded '%s' split: %d samples", name, len(df))
    return df


def _save_training_stats(train_df: pd.DataFrame) -> None:
    """Compute and save text-length statistics for the training split.

    Saves ``{mean, std, min, max}`` of character lengths to
    ``models/training_stats.json``.

    Args:
        train_df: Training DataFrame with a ``text`` column.
    """
    lengths = train_df["text"].str.len()
    stats = {
        "text_length_mean": float(lengths.mean()),
        "text_length_std": float(lengths.std()),
        "text_length_min": int(lengths.min()),
        "text_length_max": int(lengths.max()),
    }
    _STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _STATS_PATH.open("w") as fh:
        json.dump(stats, fh, indent=2)
    logger.info("Training stats saved to '%s': %s", _STATS_PATH, stats)


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------


def _compute_weights(train_df: pd.DataFrame, num_labels: int) -> torch.Tensor:
    """Compute balanced class weights from training labels using sklearn.

    Args:
        train_df: Training DataFrame with a ``label`` column.
        num_labels: Total number of classes (7 for GoEmotions macro).

    Returns:
        Float tensor of shape ``(num_labels,)`` with per-class weights.
    """
    classes = np.arange(num_labels)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_df["label"].to_numpy(),
    )
    for idx, w in zip(classes, weights):
        logger.info("Class weight [%d]: %.4f", idx, float(w))
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    """Compute weighted F1 score from Trainer eval predictions.

    Args:
        eval_pred: Named tuple with ``predictions`` (logits) and ``label_ids``.

    Returns:
        Dict ``{"f1": <weighted_f1>}`` — Trainer prepends ``eval_`` prefix.
    """
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    preds = np.argmax(logits, axis=-1)
    weighted_f1 = f1_score(labels, preds, average="weighted")
    return {"f1": float(weighted_f1)}


# ---------------------------------------------------------------------------
# MLflow callback
# ---------------------------------------------------------------------------


class MLflowEpochCallback(TrainerCallback):
    """Log per-epoch evaluation metrics to the active MLflow run.

    Hooks into ``on_evaluate`` which fires after every evaluation pass,
    so metrics appear in MLflow at the end of each epoch when
    ``eval_strategy='epoch'`` is set.
    """

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Log scalar metrics to MLflow after each evaluation.

        Args:
            args: TrainingArguments for the current run.
            state: TrainerState holding epoch and global_step.
            control: TrainerControl (unused).
            metrics: Dict of metric names to scalar values from the evaluator.
            **kwargs: Forwarded from Trainer internals (ignored).
        """
        if not metrics:
            return
        step = int(state.global_step)
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, float(value), step=step)
        logger.info(
            "MLflow — epoch %.0f  step %d  metrics: %s",
            state.epoch,
            step,
            {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
        )


# ---------------------------------------------------------------------------
# Weighted Trainer
# ---------------------------------------------------------------------------


class WeightedTrainer(Trainer):
    """Trainer subclass that applies balanced class weights to the CE loss.

    Inherits all HuggingFace Trainer behaviour and only overrides
    ``compute_loss`` to inject the per-class weight tensor.

    Attributes:
        class_weights: Float tensor of shape ``(num_labels,)`` on CPU;
                       moved to the correct device inside ``compute_loss``.
    """

    def __init__(self, class_weights: torch.Tensor, **kwargs: Any) -> None:
        """Initialise with a pre-computed class weight tensor.

        Args:
            class_weights: Balanced weight tensor from
                :func:`_compute_weights`. Shape ``(num_labels,)``.
            **kwargs: Forwarded verbatim to :class:`transformers.Trainer`.
        """
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Any = None,
    ) -> Any:
        """Compute weighted cross-entropy loss.

        Pops ``labels`` from inputs, runs a forward pass, and applies
        CrossEntropyLoss with the balanced class weight tensor.

        Args:
            model: The model being trained.
            inputs: Batch dict (input_ids, attention_mask, labels).
            return_outputs: If True, return ``(loss, outputs)`` tuple.
            num_items_in_batch: Present for API compatibility; unused.

        Returns:
            loss tensor if ``return_outputs`` is False, else ``(loss, outputs)``.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weights = self.class_weights.to(logits.device)
        loss = torch.nn.CrossEntropyLoss(weight=weights)(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------


def train(config_path: Path) -> None:
    """Run the full fine-tuning pipeline.

    Steps:
        1. Load config and set random seed.
        2. Load + validate train/val splits.
        3. Save text-length statistics.
        4. Instantiate SentimentClassifier and tokenize splits.
        5. Compute balanced class weights.
        6. Build TrainingArguments from config.
        7. Open an MLflow run, log params, attach MLflowEpochCallback.
        8. Train with WeightedTrainer (load best model at end).
        9. Save model and tokenizer to models/sentiment_model.

    Args:
        config_path: Path to config/config.yaml.
    """
    config = _load_config(config_path)
    training_cfg: dict = config["training"]
    model_cfg: dict = config["model"]

    set_seed(training_cfg["seed"])
    logger.info("Seed set to %d", training_cfg["seed"])

    train_df = _load_split("train")
    val_df = _load_split("val")

    _save_training_stats(train_df)

    classifier = SentimentClassifier()

    logger.info("Tokenizing train split (%d samples)...", len(train_df))
    train_dataset = tokenize_dataset(
        train_df, classifier.tokenizer, training_cfg["max_seq_len"]
    )
    logger.info("Tokenizing val split (%d samples)...", len(val_df))
    val_dataset = tokenize_dataset(
        val_df, classifier.tokenizer, training_cfg["max_seq_len"]
    )

    class_weights = _compute_weights(train_df, model_cfg["num_labels"])

    training_args = TrainingArguments(
        output_dir="models/checkpoints",
        num_train_epochs=training_cfg["epochs"],
        per_device_train_batch_size=training_cfg["batch_size"],
        per_device_eval_batch_size=training_cfg["batch_size"],
        learning_rate=training_cfg["learning_rate"],
        warmup_steps=training_cfg["warmup_steps"],
        weight_decay=training_cfg["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        logging_dir="logs",
        logging_steps=50,
        seed=training_cfg["seed"],
        report_to="none",  # MLflow handled by MLflowEpochCallback
    )

    mlflow.set_experiment("b1-sentiment")
    with mlflow.start_run():
        mlflow.log_params(
            {
                "base_model": model_cfg["base_model"],
                "num_labels": model_cfg["num_labels"],
                "epochs": training_cfg["epochs"],
                "learning_rate": training_cfg["learning_rate"],
                "warmup_steps": training_cfg["warmup_steps"],
                "weight_decay": training_cfg["weight_decay"],
                "batch_size": training_cfg["batch_size"],
                "max_seq_len": training_cfg["max_seq_len"],
                "seed": training_cfg["seed"],
            }
        )
        logger.info("MLflow run started — experiment: b1-sentiment")

        trainer = WeightedTrainer(
            class_weights=class_weights,
            model=classifier.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=_compute_metrics,
            callbacks=[MLflowEpochCallback()],
        )

        logger.info("Starting training (%d epochs)...", training_cfg["epochs"])
        trainer.train()
        logger.info("Training complete.")

        classifier.save(_MODEL_OUTPUT_DIR)
        logger.info("Artefacts saved — run 'make test' to verify.")


if __name__ == "__main__":
    args = _parse_args()
    train(args.config)
