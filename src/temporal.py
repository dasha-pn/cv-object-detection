"""Temporal rescoring: detections are rescored with the predictions of neighbouring frames.

The frames of a clip are consecutive, so a real object found in frame t is usually found again
in frames t-1 and t+1 at almost the same place, while many false positives appear in a single
frame only. Every detection is matched to the best overlapping detection of the same class in
each neighbouring frame, and its new score is the weighted mean of its own score and the
scores of these matches, where a missing match counts as 0. This is the idea of Seq-NMS
(Han et al., 2016, arXiv:1602.08465) reduced to a local window around every frame.

Predictions are kept per image as (boxes, scores, classes), as in src/inference.py.
"""

import numpy as np
import pandas as pd

from .common import find_data_root
from .data import load_coco
from .inference import EMPTY, top_k


# ---------------------------------------------------------------- frames and neighbourhoods

def frames_table(split):
    """One row per frame: image_id, clip, frame number and path, for train, val or test."""
    root = find_data_root()
    if split == "test":
        table = pd.read_csv(root / "test_images.csv")
        return pd.DataFrame({"image_id": table["image_id"], "clip": table["clip"], "frame": table["frame"],
                             "path": [root / name for name in table["file_name"]]})
    images = load_coco(split)["images"]
    return pd.DataFrame({"image_id": [img["id"] for img in images], "clip": [img["video"] for img in images],
                         "frame": [img["frame_id"] for img in images],
                         "path": [root / split / img["file_name"] for img in images]})


def neighbours(frames, centre_ids, radius):
    """Map every centre image_id to (image_id, offset) of the frames t-radius..t+radius of its clip."""
    index = {(row.clip, row.frame): row.image_id for row in frames.itertuples()}
    centres = frames[frames["image_id"].isin(set(centre_ids))]
    window = {}
    for row in centres.itertuples():
        found = [(index.get((row.clip, row.frame + d)), d) for d in range(-radius, radius + 1) if d != 0]
        window[row.image_id] = [(image_id, d) for image_id, d in found if image_id is not None]
    return window


# ---------------------------------------------------------------- rescoring

def pairwise_iou(a, b):
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def best_match_scores(pred, other, iou_thr):
    """Score of the matching same-class box of `other` for each box of `pred`, 0 if none.

    Every box of `other` supports only the box of `pred` it overlaps most. Without this, a weak
    near-duplicate next to a real object would also match the object in the neighbouring frame
    and inherit its high score.
    """
    boxes, _, classes = pred
    other_boxes, other_scores, other_classes = other
    matched = np.zeros(len(boxes))
    for cls in np.unique(classes):
        mine = np.flatnonzero(classes == cls)
        theirs = np.flatnonzero(other_classes == cls)
        if len(theirs) == 0:
            continue
        iou = pairwise_iou(boxes[mine], other_boxes[theirs])
        owner = iou.argmax(axis=0)
        support = np.where(iou[owner, np.arange(len(theirs))] >= iou_thr, other_scores[theirs], 0.0)
        best = np.zeros(len(mine))
        np.maximum.at(best, owner, support)
        matched[mine] = best
    return matched


def rescore(preds, window, radius, iou_thr=0.5, own_weight=1.0, mode="mean"):
    """New predictions of the centre frames in `window`.

    preds holds the predictions of all frames that appear in the window. Only neighbours up to
    `radius` frames away are used, so one cached window can be evaluated at several radii.
    mode "mean" counts a neighbour without a match as score 0, so boxes that flicker lose score;
    mode "smooth" averages only over the neighbours with a match, so it smooths the scores of
    consistent boxes without punishing intermittent ones.
    """
    out = {}
    for centre, neighbour_ids in window.items():
        pred = preds.get(centre, EMPTY)
        used = [image_id for image_id, offset in neighbour_ids if abs(offset) <= radius and image_id in preds]
        total = own_weight * pred[1]
        weight = np.full(len(pred[1]), own_weight)
        for neighbour in used:
            matched = best_match_scores(pred, preds[neighbour], iou_thr)
            total = total + matched
            weight = weight + (1.0 if mode == "mean" else (matched > 0))
        out[centre] = top_k((pred[0], total / weight, pred[2]))
    return out
