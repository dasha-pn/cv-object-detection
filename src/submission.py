"""Kaggle submission format: one row per test image, `image_id,PredictionString`.

PredictionString is a space-separated list of `class score x y w h` groups with 0-based
class ids and the top-left corner in pixels of the original image. Only the 500 most
confident detections per image are scored.
"""

from collections import defaultdict

import pandas as pd

from .common import find_data_root

MAX_DETECTIONS = 500


def test_images():
    """Test frames as (image_id, path) pairs in the order of test_images.csv."""
    root = find_data_root()
    table = pd.read_csv(root / "test_images.csv")
    return [(row.image_id, root / row.file_name) for row in table.itertuples()]


def prediction_string(detections):
    best = sorted(detections, key=lambda d: -d["score"])[:MAX_DETECTIONS]
    groups = []
    for det in best:
        x, y, w, h = det["bbox"]
        groups.append(f"{det['category_id'] - 1} {det['score']:.4f} {x:.1f} {y:.1f} {w:.1f} {h:.1f}")
    return " ".join(groups)


def write_submission(detections, image_ids, path):
    """Write COCO-style detections (1-based category ids) as a submission CSV."""
    by_image = defaultdict(list)
    for det in detections:
        by_image[det["image_id"]].append(det)

    # Every test image gets a row, an empty string means no detections
    rows = [(image_id, prediction_string(by_image[image_id])) for image_id in image_ids]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["image_id", "PredictionString"]).to_csv(path, index=False)


def read_submission(path):
    """Parse a submission CSV back into COCO-style detections (used to check the format)."""
    table = pd.read_csv(path, keep_default_na=False)
    detections = []
    for image_id, text in zip(table["image_id"], table["PredictionString"]):
        values = text.split()
        for k in range(0, len(values), 6):
            cls, score, x, y, w, h = values[k:k + 6]
            detections.append({
                "image_id": image_id,
                "category_id": int(cls) + 1,
                "score": float(score),
                "bbox": [float(x), float(y), float(w), float(h)],
            })
    return detections
