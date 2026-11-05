"""Error analysis at a fixed confidence threshold, shared by all detectors.

Every detection gets one type, checked in this order (a simplified version of TIDE,
Bolya et al., 2020, arXiv:2008.08115):
    tp          IoU >= 0.5 with an unmatched ground-truth box of the same class
    duplicate   IoU >= 0.5 with a same-class box that is already matched
    class       IoU >= 0.5 with a box of another class
    location    0.1 <= IoU < 0.5 with a same-class box
    ignored     mostly inside an ignored region (not an error)
    background  everything else
Every ground-truth box is either matched or missed.
"""

import json
from collections import defaultdict

import numpy as np
import pandas as pd

from .common import CLASS_NAMES
from .evaluate import coco_evaluate

IOU_MATCH = 0.5
IOU_LOCATION = 0.1
IGNORE_COVERAGE = 0.5
SIZE_BINS = [0, 32 ** 2, 96 ** 2, np.inf]
SIZE_NAMES = ["small", "medium", "large"]


def xywh_to_xyxy(boxes):
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return np.concatenate([boxes[:, :2], boxes[:, :2] + boxes[:, 2:]], axis=1)


def pairwise_iou(a, b):
    """IoU matrix between xyxy boxes a (n, 4) and b (m, 4)."""
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def coverage(a, regions):
    """Fraction of each box in a covered by its best overlapping region."""
    if len(regions) == 0:
        return np.zeros(len(a))
    tl = np.maximum(a[:, None, :2], regions[None, :, :2])
    br = np.minimum(a[:, None, 2:], regions[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    return (inter / (area_a[:, None] + 1e-9)).max(axis=1)


def load_ground_truth(gt_path):
    """Per image: target boxes, their classes and the ignored regions (all xyxy)."""
    gt = json.loads(gt_path.read_text())
    per_image = {img["id"]: {"boxes": [], "classes": [], "ignored": set()} for img in gt["images"]}
    for ann in gt["annotations"]:
        item = per_image[ann["image_id"]]
        if ann["iscrowd"]:
            item["ignored"].add(tuple(ann["bbox"]))
        else:
            item["boxes"].append(ann["bbox"])
            item["classes"].append(ann["category_id"] - 1)

    for item in per_image.values():
        item["boxes"] = xywh_to_xyxy(item["boxes"])
        item["classes"] = np.array(item["classes"], dtype=int)
        item["ignored"] = xywh_to_xyxy(sorted(item["ignored"]))
    return per_image


def classify_errors(gt_path, detections, score_thr=0.3):
    """Return (detections table with an error type, ground-truth table with a matched flag)."""
    ground_truth = load_ground_truth(gt_path)
    dets_by_image = defaultdict(list)
    for det in detections:
        if det["score"] >= score_thr:
            dets_by_image[det["image_id"]].append(det)

    det_rows, gt_rows = [], []
    for image_id, gt in ground_truth.items():
        dets = sorted(dets_by_image.get(image_id, []), key=lambda d: -d["score"])
        det_boxes = xywh_to_xyxy([d["bbox"] for d in dets])
        det_classes = np.array([d["category_id"] - 1 for d in dets], dtype=int)

        matched = np.zeros(len(gt["boxes"]), dtype=bool)
        ious = pairwise_iou(det_boxes, gt["boxes"]) if len(dets) and len(gt["boxes"]) else None
        ignored_cov = coverage(det_boxes, gt["ignored"]) if len(dets) else []

        for i, det in enumerate(dets):
            kind = "background"
            if ious is not None:
                same = gt["classes"] == det_classes[i]
                iou_same = np.where(same, ious[i], 0.0)
                iou_free = np.where(same & ~matched, ious[i], 0.0)
                if iou_free.max() >= IOU_MATCH:
                    kind = "tp"
                    matched[iou_free.argmax()] = True
                elif iou_same.max() >= IOU_MATCH:
                    kind = "duplicate"
                elif np.where(~same, ious[i], 0.0).max() >= IOU_MATCH:
                    kind = "class"
                elif iou_same.max() >= IOU_LOCATION:
                    kind = "location"
            if kind == "background" and ignored_cov[i] >= IGNORE_COVERAGE:
                kind = "ignored"
            det_rows.append((image_id, det_classes[i], det["score"], kind))

        areas = np.prod(gt["boxes"][:, 2:] - gt["boxes"][:, :2], axis=1)
        for k in range(len(gt["boxes"])):
            gt_rows.append((image_id, gt["classes"][k], areas[k], matched[k]))

    det_table = pd.DataFrame(det_rows, columns=["image_id", "cls", "score", "type"])
    gt_table = pd.DataFrame(gt_rows, columns=["image_id", "cls", "area", "matched"])
    gt_table["size"] = pd.cut(gt_table["area"], SIZE_BINS, labels=SIZE_NAMES, right=False)
    det_table["class_name"] = [CLASS_NAMES[c] for c in det_table["cls"]]
    gt_table["class_name"] = [CLASS_NAMES[c] for c in gt_table["cls"]]
    return det_table, gt_table


def per_clip_ap(gt_path, detections):
    """COCO AP of every clip separately, for paired comparisons between runs."""
    clips = defaultdict(list)
    for img in json.loads(gt_path.read_text())["images"]:
        clips[img["video"]].append(img["id"])
    return {clip: coco_evaluate(gt_path, detections, image_ids=ids)["AP"]
            for clip, ids in sorted(clips.items())}


def paired_comparison(clip_ap, reference):
    """Per-clip AP difference of every run against a reference run.

    Clips are independent videos, so the spread of the differences across clips tells
    whether a change in the overall AP is larger than the clip-to-clip noise.
    """
    table = pd.DataFrame(clip_ap)
    rows = {}
    for run in table.columns:
        diff = table[run] - table[reference]
        rows[run] = {
            "mean_diff": diff.mean(),
            "std_err": diff.std(ddof=1) / np.sqrt(len(diff)),
            "clips_better": int((diff > 0).sum()),
            "clips_worse": int((diff < 0).sum()),
        }
    return pd.DataFrame(rows).T


def error_summary(det_table, gt_table):
    """One row of counts: detections per type and recall per object size."""
    counts = det_table["type"].value_counts()
    row = {kind: int(counts.get(kind, 0))
           for kind in ["tp", "duplicate", "class", "location", "background", "ignored"]}
    row["missed"] = int((~gt_table["matched"]).sum())
    row["recall"] = gt_table["matched"].mean()
    for size in SIZE_NAMES:
        row[f"recall_{size}"] = gt_table.loc[gt_table["size"] == size, "matched"].mean()
    return row
