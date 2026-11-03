"""Running a trained Ultralytics detector on evaluation frames and scoring it."""

import json

import cv2
import numpy as np

from .common import WORK_DIR, find_data_root
from .evaluate import coco_evaluate

# Evaluation sets: name -> (source split, COCO ground truth built by prepare_data.py)
EVAL_SETS = {
    "fold0": ("train", WORK_DIR / "eval" / "fold0_holdout.json"),
    "val": ("val", WORK_DIR / "eval" / "val.json"),
}

# Low threshold and many boxes per image, as needed for a full precision-recall curve
PREDICT_ARGS = {"conf": 0.001, "iou": 0.7, "max_det": 500, "quantize": 16, "verbose": False}
WARMUP_IMAGES = 10


def eval_images(eval_set):
    split, gt_path = EVAL_SETS[eval_set]
    gt = json.loads(gt_path.read_text())
    root = find_data_root() / split
    return [(img["id"], root / img["file_name"]) for img in gt["images"]]


def to_coco(result, image_id):
    """Ultralytics result -> COCO detections (category ids are 1-based)."""
    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    scores = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)

    detections = []
    for (x1, y1, x2, y2), score, cls in zip(xyxy, scores, classes):
        detections.append({
            "image_id": image_id,
            "category_id": int(cls) + 1,
            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            "score": float(score),
        })
    return detections


def predict_images(model, images, imgsz, **overrides):
    """Predict (image_id, path) pairs one by one.

    Returns COCO detections and the mean time per image in ms (batch 1, FP16):
    preprocessing + network + NMS as timed by Ultralytics, without JPEG decoding.
    The first frames are a warm-up and are not timed.
    """
    args = {**PREDICT_ARGS, **overrides}
    detections, times = [], []

    for i, (image_id, path) in enumerate(images):
        image = cv2.imread(str(path))
        result = model.predict(image, imgsz=imgsz, **args)[0]
        if i >= WARMUP_IMAGES:
            times.append(sum(result.speed.values()))
        detections.extend(to_coco(result, image_id))

    return detections, float(np.mean(times)) if times else float("nan")


def predict_set(model, eval_set, imgsz, **overrides):
    return predict_images(model, eval_images(eval_set), imgsz, **overrides)


def evaluate_model(model, imgsz, eval_sets=("fold0", "val"), save_dir=None, **overrides):
    """Score a model on the evaluation sets, optionally saving metrics and confident predictions."""
    report = {}
    for eval_set in eval_sets:
        detections, ms_per_image = predict_set(model, eval_set, imgsz, **overrides)
        metrics = coco_evaluate(EVAL_SETS[eval_set][1], detections)
        metrics["ms_per_image"] = ms_per_image
        metrics["fps"] = 1000.0 / ms_per_image
        report[eval_set] = metrics

        if save_dir is not None:
            # Predictions above 0.05 are enough for error analysis and keep files small
            confident = [d for d in detections if d["score"] >= 0.05]
            (save_dir / f"predictions_{eval_set}.json").write_text(json.dumps(confident))

    if save_dir is not None:
        (save_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    return report
