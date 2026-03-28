"""Load GoEmotions dataset and map 27 labels to 7 macro-categories.

Saves the label mapping to data/raw/label_map.json and logs class
distribution via get_logger.  Only single-label samples are retained.
"""

import json
from pathlib import Path

from datasets import load_dataset

from src.logger import get_logger

logger = get_logger(__name__)

# Mapping from GoEmotions simplified-split label indices to 7 macro-categories.
# joy=0, sadness=1, anger=2, fear=3, surprise=4, disgust=5, neutral=6
#
# Verified GoEmotions simplified indices (2026-03-28):
#   0:admiration  1:amusement   2:anger       3:annoyance   4:approval
#   5:caring      6:confusion   7:curiosity   8:desire      9:disappointment
#  10:disapproval 11:disgust    12:embarrassment 13:excitement 14:fear
#  15:gratitude  16:grief      17:joy         18:love       19:nervousness
#  20:optimism   21:pride      22:realization 23:relief     24:remorse
#  25:sadness    26:surprise   27:neutral
GOEMOTION_TO_MACRO: dict[int, int] = {
    # Joy cluster
    0: 0,  # admiration -> joy-adjacent
    1: 0,  # amusement
    4: 0,  # approval -> positive/joy
    13: 0,  # excitement
    15: 0,  # gratitude -> joy
    17: 0,  # joy
    18: 0,  # love
    20: 0,  # optimism
    23: 0,  # relief
    # Sadness cluster
    9: 1,  # disappointment
    16: 1,  # grief
    24: 1,  # remorse
    25: 1,  # sadness
    # Anger cluster
    2: 2,  # anger
    3: 2,  # annoyance
    10: 2,  # disapproval
    21: 2,  # pride -> disapproval-adjacent
    # Fear cluster
    5: 3,  # caring -> concern/fear-adjacent
    12: 3,  # embarrassment
    14: 3,  # fear
    19: 3,  # nervousness
    # Surprise cluster
    7: 4,  # curiosity
    22: 4,  # realization
    26: 4,  # surprise
    # Disgust cluster
    6: 5,  # confusion -> disgust-adjacent
    8: 5,  # desire -> mapped to disgust (closest remaining)
    11: 5,  # disgust
    # Neutral
    27: 6,  # neutral
}

MACRO_LABEL_NAMES: dict[int, str] = {
    0: "joy",
    1: "sadness",
    2: "anger",
    3: "fear",
    4: "surprise",
    5: "disgust",
    6: "neutral",
}


def load_and_map() -> None:
    """Load GoEmotions simplified split, map labels, and save artefacts.

    Uses the 'simplified' config which already provides single-label
    samples.  Logs class distribution for each split and writes the
    macro-category label map to data/raw/label_map.json.
    """
    logger.info("Loading google-research-datasets/go_emotions (simplified)")
    ds = load_dataset(  # nosec B615
        "google-research-datasets/go_emotions", "simplified"
    )

    label_map_path = Path("data/raw/label_map.json")
    label_map_path.parent.mkdir(parents=True, exist_ok=True)

    with label_map_path.open("w") as fh:
        json.dump(MACRO_LABEL_NAMES, fh, indent=2)
    logger.info("Saved label map to %s", label_map_path)

    for split_name, split_data in ds.items():
        logger.info(
            "Split '%s' — %d samples before filtering",
            split_name,
            len(split_data),
        )

        # simplified split stores labels as a list; keep single-label rows
        single = split_data.filter(lambda row: len(row["labels"]) == 1)
        logger.info(
            "Split '%s' — %d samples after single-label filter",
            split_name,
            len(single),
        )

        # Count macro-category distribution
        distribution: dict[int, int] = {k: 0 for k in range(7)}
        for row in single:
            original_label = row["labels"][0]
            macro = GOEMOTION_TO_MACRO.get(original_label, 6)
            distribution[macro] += 1

        for macro_idx, count in distribution.items():
            logger.info(
                "Split '%s' | %s (%d) = %d samples",
                split_name,
                MACRO_LABEL_NAMES[macro_idx],
                macro_idx,
                count,
            )


if __name__ == "__main__":
    load_and_map()
