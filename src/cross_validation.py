"""Out-of-fold evaluation: every fold holdout is scored with the run that did not train on it.

Run k of an experiment trains on all folds except k (prepare_data.py --fold k, train.py --fold k)
and predicts the holdout of fold k. The out-of-fold score pools these predictions over all folds,
so every annotated training clip is scored exactly once by a model that never saw it.
"""

import json

import numpy as np

from .common import N_FOLDS, WORK_DIR
from .evaluate import coco_evaluate
from .predict import FOLD_SETS

OOF_GT_PATH = WORK_DIR / "eval" / "oof_holdout.json"
CV_METRICS = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"]


def oof_ground_truth():
    """COCO ground truth of the holdouts of all folds together, written once."""
    if not OOF_GT_PATH.exists():
        parts = [json.loads(FOLD_SETS[f"fold{k}"][1].read_text()) for k in range(N_FOLDS)]
        images = [img for part in parts for img in part["images"]]

        # Annotation ids restart in every holdout file, so they are renumbered
        annotations = [{**ann, "id": i} for i, ann in
                       enumerate((ann for part in parts for ann in part["annotations"]), start=1)]
        OOF_GT_PATH.write_text(json.dumps({**parts[0], "images": images, "annotations": annotations}))
    return OOF_GT_PATH


def cross_validate(detections_by_fold):
    """Score {fold: COCO detections of that fold's holdout} per fold and out of fold.

    Returns the metrics of every fold, their mean and standard deviation over the folds, and the
    out-of-fold metrics of all predictions pooled.
    """
    per_fold = {k: coco_evaluate(FOLD_SETS[f"fold{k}"][1], detections)
                for k, detections in sorted(detections_by_fold.items())}
    pooled = [det for detections in detections_by_fold.values() for det in detections]

    values = {m: [metrics[m] for metrics in per_fold.values()] for m in CV_METRICS}
    return {
        "folds": per_fold,
        "mean": {m: float(np.mean(v)) for m, v in values.items()},
        "std": {m: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for m, v in values.items()},
        "oof": coco_evaluate(oof_ground_truth(), pooled) if len(per_fold) == N_FOLDS else None,
    }
