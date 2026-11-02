"""Shared constants and paths used by all scripts and notebooks."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

CLASS_NAMES = ["person", "bicycle", "car", "van", "truck", "tricycle", "bus", "motor"]
SEED = 42
N_FOLDS = 3

# Generated datasets and training outputs (both ignored by git)
WORK_DIR = REPO_ROOT / "data"
RUNS_DIR = REPO_ROOT / "runs"

# Possible locations of the competition data, the first existing one is used
_DATA_CANDIDATES = [
    os.environ.get("VISDRONE_ROOT", ""),
    REPO_ROOT / "data" / "raw",
    REPO_ROOT / "context",
    "/kaggle/input/competitions/multiple-object-video-detection-hw-2-ucu-ai-2026",
]


def find_data_root():
    for candidate in _DATA_CANDIDATES:
        if candidate and (Path(candidate) / "train" / "annotations_coco.json").is_file():
            return Path(candidate)
    raise FileNotFoundError(
        "Competition data not found. Set VISDRONE_ROOT to the folder with train/, val/, test/."
    )
