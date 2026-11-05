"""Inference strategies on top of a trained detector: flip TTA, tiling and box fusion.

Inside this module the predictions of one image are a tuple (boxes, scores, classes):
xyxy pixel boxes (n, 4), scores (n,) and 0-based classes (n,).
"""

import time
from collections import defaultdict

import cv2
import numpy as np
import torch
from torchvision.ops import batched_nms

from .predict import PREDICT_ARGS, WARMUP_IMAGES

MAX_DETECTIONS = 500
EMPTY = (np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int))


# ---------------------------------------------------------------- single passes

def predict_pass(model, image, imgsz):
    result = model.predict(image, imgsz=imgsz, **PREDICT_ARGS)[0]
    boxes = result.boxes
    return (boxes.xyxy.cpu().numpy().astype(np.float64), boxes.conf.cpu().numpy().astype(np.float64),
            boxes.cls.cpu().numpy().astype(int))


def predict_flipped(model, image, imgsz):
    """Predict the mirrored frame and mirror the boxes back."""
    boxes, scores, classes = predict_pass(model, np.ascontiguousarray(image[:, ::-1]), imgsz)
    width = image.shape[1]
    boxes = np.stack([width - boxes[:, 2], boxes[:, 1], width - boxes[:, 0], boxes[:, 3]], axis=1)
    return boxes, scores, classes


def tile_windows(width, height, grid=2, overlap=0.2):
    """grid x grid tiles as (x1, y1, x2, y2); neighbours share `overlap` of a tile."""
    tile_w = int(np.ceil(width / (grid - (grid - 1) * overlap)))
    tile_h = int(np.ceil(height / (grid - (grid - 1) * overlap)))
    xs = np.linspace(0, width - tile_w, grid).round().astype(int)
    ys = np.linspace(0, height - tile_h, grid).round().astype(int)
    return [(x, y, x + tile_w, y + tile_h) for y in ys for x in xs]


def predict_tiles(model, image, imgsz, grid=2, overlap=0.2, edge=2):
    """Predict every tile at the full inference size, i.e. zoomed in (SAHI, arXiv:2202.06934).

    Boxes touching an inner tile border are cut objects and are dropped: the object is whole
    in the full frame or in the neighbouring tile.
    """
    height, width = image.shape[:2]
    parts = []
    for x1, y1, x2, y2 in tile_windows(width, height, grid, overlap):
        boxes, scores, classes = predict_pass(model, image[y1:y2, x1:x2], imgsz)
        cut = (((boxes[:, 0] < edge) & (x1 > 0)) | ((boxes[:, 2] > x2 - x1 - edge) & (x2 < width))
               | ((boxes[:, 1] < edge) & (y1 > 0)) | ((boxes[:, 3] > y2 - y1 - edge) & (y2 < height)))
        keep = ~cut
        parts.append((boxes[keep] + [x1, y1, x1, y1], scores[keep], classes[keep]))
    return parts


# ---------------------------------------------------------------- merging several predictions

def concat(parts):
    parts = [p for p in parts if len(p[1])] or [EMPTY]
    return tuple(np.concatenate([p[k] for p in parts]) for k in range(3))


def top_k(pred, k=MAX_DETECTIONS):
    order = np.argsort(-pred[1], kind="stable")[:k]
    return tuple(a[order] for a in pred)


def nms_merge(parts, iou_thr=0.6):
    """Class-wise NMS over the union of all predictions."""
    boxes, scores, classes = concat(parts)
    keep = batched_nms(torch.from_numpy(boxes), torch.from_numpy(scores),
                       torch.from_numpy(classes), iou_thr).numpy()
    return top_k((boxes[keep], scores[keep], classes[keep]))


def iou_one_to_many(box, boxes):
    tl = np.maximum(box[:2], boxes[:, :2])
    br = np.minimum(box[2:], boxes[:, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=1)
    area = np.prod(box[2:] - box[:2])
    areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
    return inter / (area + areas - inter + 1e-9)


def weighted_boxes_fusion(parts, weights=None, iou_thr=0.55):
    """Weighted Boxes Fusion (Solovyev et al., 2021, arXiv:1910.13302).

    parts are predictions of the same image by different models or passes. Boxes of one class
    are visited by decreasing score; a box joins the first cluster whose fused box overlaps it
    with IoU > iou_thr, otherwise it starts a new cluster. The fused box is the score-weighted
    mean of the cluster boxes. Its score is the mean cluster score, scaled down when fewer
    sources than len(parts) found the object, so boxes that all sources agree on rank higher.
    """
    weights = np.ones(len(parts)) if weights is None else np.asarray(weights, dtype=np.float64)
    weighted = [(b, s * w, c) for (b, s, c), w in zip(parts, weights)]
    boxes, scores, classes = concat(weighted)
    total_weight = weights.sum()

    out = []
    for cls in np.unique(classes):
        idx = np.flatnonzero(classes == cls)
        idx = idx[np.argsort(-scores[idx], kind="stable")]

        fused = np.zeros((len(idx), 4))
        box_sum = np.zeros((len(idx), 4))
        score_sum = np.zeros(len(idx))
        weight_sum = np.zeros(len(idx))
        count = np.zeros(len(idx))
        n = 0
        for i in idx:
            j = -1
            if n:
                iou = iou_one_to_many(boxes[i], fused[:n])
                best = int(iou.argmax())
                if iou[best] > iou_thr:
                    j = best
            if j < 0:
                j, n = n, n + 1
            box_sum[j] += boxes[i] * scores[i]
            score_sum[j] += scores[i]
            count[j] += 1
            fused[j] = box_sum[j] / score_sum[j]

        # Mean score of the cluster, times the share of sources that voted for it
        fused_scores = score_sum[:n] / count[:n] * np.minimum(count[:n], total_weight) / total_weight
        out.append((fused[:n], fused_scores, np.full(n, cls)))
    return top_k(concat(out))


# ---------------------------------------------------------------- strategies

def strategy(name, imgsz, merge="nms", grid=2, overlap=0.2, min_tile_width=0):
    """Return predict(model, image) -> prediction for one of the inference strategies.

    plain   one pass on the full frame
    nms     one pass + class-wise NMS, for detectors that are trained without NMS (RT-DETR)
    flip    full frame + mirrored frame
    tiles   full frame + grid x grid zoomed tiles
    tiles_flip  full frame + mirrored frame + grid x grid zoomed tiles

    Frames narrower than min_tile_width get the plain pass in the tile strategies: tiles help by
    recovering pixels the full-frame pass loses to downscaling, and a small frame loses none.
    """
    def merge_parts(parts):
        return weighted_boxes_fusion(parts) if merge == "wbf" else nms_merge(parts)

    def predict(model, image):
        full = predict_pass(model, image, imgsz)
        if name == "plain" or (name.startswith("tiles") and image.shape[1] < min_tile_width):
            return top_k(full)
        if name == "nms":
            return nms_merge([full], iou_thr=0.7)
        if name == "flip":
            return merge_parts([full, predict_flipped(model, image, imgsz)])
        if name == "tiles":
            return merge_parts([full] + predict_tiles(model, image, imgsz, grid, overlap))
        if name == "tiles_flip":
            flipped = predict_flipped(model, image, imgsz)
            return merge_parts([full, flipped] + predict_tiles(model, image, imgsz, grid, overlap))
        raise ValueError(f"unknown strategy {name}")

    return predict


def to_coco(pred, image_id):
    boxes, scores, classes = pred
    return [{"image_id": image_id, "category_id": int(c) + 1,
             "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(s)}
            for (x1, y1, x2, y2), s, c in zip(boxes, scores, classes)]


def from_coco(detections):
    """COCO detections -> {image_id: prediction}."""
    grouped = defaultdict(list)
    for det in detections:
        grouped[det["image_id"]].append(det)
    preds = {}
    for image_id, dets in grouped.items():
        xywh = np.array([d["bbox"] for d in dets], dtype=np.float64)
        boxes = np.concatenate([xywh[:, :2], xywh[:, :2] + xywh[:, 2:]], axis=1)
        preds[image_id] = (boxes, np.array([d["score"] for d in dets]),
                           np.array([d["category_id"] - 1 for d in dets]))
    return preds


def run_strategy(model, images, predict):
    """Apply a strategy to (image_id, path) pairs.

    Returns COCO detections and the mean wall time per image in ms (all passes and the
    merging, without JPEG decoding); the first frames are a warm-up and are not timed.
    """
    detections, times = [], []
    for i, (image_id, path) in enumerate(images):
        image = cv2.imread(str(path))
        start = time.perf_counter()
        pred = predict(model, image)
        if i >= WARMUP_IMAGES:
            times.append((time.perf_counter() - start) * 1000)
        detections.extend(to_coco(pred, image_id))
    return detections, float(np.mean(times)) if times else float("nan")


def ensemble(sources, image_ids, weights=None, iou_thr=0.55):
    """Fuse the COCO detections of several runs image by image with WBF."""
    per_source = [from_coco(dets) for dets in sources]
    detections = []
    for image_id in image_ids:
        parts = [preds.get(image_id, EMPTY) for preds in per_source]
        detections.extend(to_coco(weighted_boxes_fusion(parts, weights, iou_thr), image_id))
    return detections
