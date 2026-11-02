"""COCO-style evaluation shared by all detectors.

Ignored regions are iscrowd=1 annotations, so detections inside them are not
counted as false positives. Up to 500 detections per image are scored, as on Kaggle.
"""

from faster_coco_eval import COCO, COCOeval_faster

from .common import CLASS_NAMES

MAX_DETS = [1, 100, 500]


def coco_evaluate(gt_path, detections):
    """Evaluate COCO detections: image_id, category_id (1-based), bbox xywh, score."""
    gt = COCO(str(gt_path))
    if not detections:
        return {"AP": 0.0}
    dt = gt.loadRes(detections)

    evaluator = COCOeval_faster(gt, dt, iouType="bbox", print_function=lambda *a: None)
    evaluator.params.maxDets = MAX_DETS
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    stats = evaluator.stats_as_dict
    metrics = {
        "AP": stats["AP_all"],
        "AP50": stats["AP_50"],
        "AP75": stats["AP_75"],
        "AP_small": stats["AP_small"],
        "AP_medium": stats["AP_medium"],
        "AP_large": stats["AP_large"],
        f"AR_{MAX_DETS[-1]}": stats[f"AR_{MAX_DETS[-1]}"],
    }

    # precision has shape [iou, recall, class, area, max_dets]
    # area index 0 is "all", the last max_dets index is 500
    precision = evaluator.eval["precision"][:, :, :, 0, -1]
    for k, name in enumerate(CLASS_NAMES):
        values = precision[:, :, k]
        values = values[values > -1]
        metrics[f"AP_{name}"] = float(values.mean()) if values.size else float("nan")

    return {k: float(v) for k, v in metrics.items()}


def format_metrics(metrics, keys=("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large")):
    return "  ".join(f"{k}={metrics[k]:.4f}" for k in keys if k in metrics)
