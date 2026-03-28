"""Day 3 evaluation: fine-tuned vs. zero-shot emotion classifier comparison.

Computes accuracy, weighted F1, and a per-class classification report for the
fine-tuned model, then benchmarks against
bhadresh-savani/distilbert-base-uncased-emotion on a 200-sample subset.
All results are written to reports/results.json and logged to MLflow.

NOTE: The training data was built with an incorrect GOEMOTION_TO_MACRO mapping
in src/data/load_raw.py (GoEmotions simplified label indices were wrong).
As a result, the class indices in train/test CSVs do not correspond to the
emotion names expected by config.yaml.  The fine-tuned metrics below are
internally self-consistent (model trained and tested on the same broken scheme)
but the absolute emotion label names should not be trusted.  The zero-shot
comparison scores poorly because the zero-shot model uses correct emotion labels
while our test CSV has the broken mapping.  Fixing GOEMOTION_TO_MACRO and
retraining is tracked as the next action.

Usage:
    python src/evaluation/evaluate.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import pipeline

from src.logger import get_logger
from src.models.model import SentimentClassifier

logger = get_logger(__name__)

_CONFIG_PATH = Path("config/config.yaml")
_TEST_CSV = Path("data/processed/test.csv")
_REPORTS_DIR = Path("reports")
_FIGURES_DIR = _REPORTS_DIR / "figures"
_RESULTS_PATH = _REPORTS_DIR / "results.json"
_ZERO_SHOT_MODEL = "bhadresh-savani/distilbert-base-uncased-emotion"
_ZERO_SHOT_N = 200


def _load_config() -> dict:
    with _CONFIG_PATH.open() as fh:
        return yaml.safe_load(fh)


def _batch_predict(
    classifier: SentimentClassifier,
    texts: list[str],
    batch_size: int = 64,
) -> list[str]:
    """Run inference in fixed-size batches to avoid OOM on CPU."""
    results: list[str] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        results.extend(classifier.predict(batch))
        if (start // batch_size) % 10 == 0:
            logger.info(
                "Inference progress: %d / %d",
                min(start + batch_size, len(texts)),
                len(texts),
            )
    return results


def evaluate_finetuned(
    classifier: SentimentClassifier,
    test_df: pd.DataFrame,
    emotion_labels: list[str],
) -> dict:
    """Evaluate the fine-tuned model on the full test split.

    Args:
        classifier: Loaded SentimentClassifier instance.
        test_df: DataFrame with 'text' (str) and 'label' (int) columns.
        emotion_labels: Ordered list of emotion name strings (index = class id).

    Returns:
        Dict with keys 'accuracy', 'weighted_f1', 'report'.
    """
    logger.info("Running fine-tuned evaluation on %d samples", len(test_df))

    preds = _batch_predict(classifier, test_df["text"].tolist())
    id2name = {i: lbl for i, lbl in enumerate(emotion_labels)}
    true = test_df["label"].map(id2name).tolist()

    accuracy = accuracy_score(true, preds)
    f1 = f1_score(true, preds, average="weighted", zero_division=0)
    report = classification_report(
        true, preds, target_names=emotion_labels, zero_division=0
    )

    logger.info("Fine-tuned accuracy: %.4f  weighted_f1: %.4f", accuracy, f1)
    logger.info("Classification report:\n%s", report)

    _save_confusion_matrix(true, preds, emotion_labels)

    return {"accuracy": accuracy, "weighted_f1": f1, "report": report}


def _save_confusion_matrix(
    true: list[str],
    preds: list[str],
    emotion_labels: list[str],
) -> None:
    """Render and save a seaborn confusion-matrix heatmap."""
    plt.switch_backend("Agg")  # non-interactive backend for file-only output
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(true, preds, labels=emotion_labels)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=emotion_labels,
        yticklabels=emotion_labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Fine-tuned model — confusion matrix (test set)")
    fig.tight_layout()

    dest = _FIGURES_DIR / "confusion_matrix_finetuned.png"
    fig.savefig(str(dest), dpi=120)
    plt.close(fig)
    logger.info("Confusion matrix saved to '%s'", dest)


def evaluate_zero_shot(
    test_df: pd.DataFrame,
    emotion_labels: list[str],
) -> dict:
    """Benchmark bhadresh-savani zero-shot model on 200 non-neutral samples.

    The zero-shot model does not have a 'neutral' class, so samples with
    ground-truth label 'neutral' are excluded before sampling.

    Args:
        test_df: Full test DataFrame with 'text' and integer 'label' columns.
        emotion_labels: Ordered list of emotion name strings (index = class id).

    Returns:
        Dict with 'accuracy', 'weighted_f1', 'model_used', 'n_samples'.
    """
    logger.info("Loading zero-shot model: %s", _ZERO_SHOT_MODEL)
    pipe = pipeline(  # nosec B615
        "text-classification",
        model=_ZERO_SHOT_MODEL,
    )

    id2name = {i: lbl for i, lbl in enumerate(emotion_labels)}
    neutral_idx = emotion_labels.index("neutral")
    non_neutral = test_df[test_df["label"] != neutral_idx].copy()

    sample = non_neutral.sample(n=min(_ZERO_SHOT_N, len(non_neutral)), random_state=42)
    logger.info("Zero-shot evaluation: %d samples (neutral excluded)", len(sample))

    raw_preds = pipe(sample["text"].tolist(), truncation=True, max_length=512)
    preds = [r["label"] for r in raw_preds]
    true = sample["label"].map(id2name).tolist()

    six_labels = [lbl for lbl in emotion_labels if lbl != "neutral"]
    accuracy = accuracy_score(true, preds)
    f1 = f1_score(true, preds, average="weighted", zero_division=0, labels=six_labels)

    logger.info("Zero-shot accuracy: %.4f  weighted_f1: %.4f", accuracy, f1)

    return {
        "accuracy": accuracy,
        "weighted_f1": f1,
        "model_used": _ZERO_SHOT_MODEL,
        "n_samples": len(sample),
    }


def main() -> None:
    """Orchestrate evaluation, JSON export, and MLflow logging."""
    config = _load_config()
    label_map: dict = config["emotion_labels"]
    emotion_labels: list[str] = [label_map[i] for i in sorted(label_map)]

    test_df = pd.read_csv(_TEST_CSV)
    logger.info("Test set loaded: %d rows", len(test_df))

    classifier = SentimentClassifier()
    classifier.load(Path("models/sentiment_model"))

    finetuned = evaluate_finetuned(classifier, test_df, emotion_labels)
    zero_shot = evaluate_zero_shot(test_df, emotion_labels)

    delta_f1 = finetuned["weighted_f1"] - zero_shot["weighted_f1"]
    logger.info("delta_f1 (finetuned - zero_shot): %.4f", delta_f1)

    results = {
        "finetuned": finetuned,
        "zero_shot": zero_shot,
        "delta_f1": delta_f1,
    }

    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with _RESULTS_PATH.open("w") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Results saved to '%s'", _RESULTS_PATH)

    with mlflow.start_run(run_name="day3-evaluation"):
        mlflow.log_metrics(
            {
                "test_accuracy": finetuned["accuracy"],
                "test_f1_weighted": finetuned["weighted_f1"],
                "zero_shot_f1": zero_shot["weighted_f1"],
                "delta_f1": delta_f1,
            }
        )
        mlflow.log_artifact(str(_FIGURES_DIR / "confusion_matrix_finetuned.png"))
        mlflow.log_artifact(str(_RESULTS_PATH))
    logger.info("MLflow run 'day3-evaluation' complete")


if __name__ == "__main__":
    main()
