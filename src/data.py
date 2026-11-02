"""Loading annotations, sequence folds, frame sampling and Ultralytics dataset export."""

import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .common import CLASS_NAMES, N_FOLDS, find_data_root

# Gray value Ultralytics uses for letterbox padding, so masked areas look like padding
MASK_VALUE = 114


@lru_cache(maxsize=None)
def load_coco(split):
    with open(find_data_root() / split / "annotations_coco.json", encoding="utf-8") as f:
        return json.load(f)


def images_table(split):
    """One row per frame with its clip, frame index and size."""
    df = pd.DataFrame(load_coco(split)["images"])
    df = df.rename(columns={"id": "image_id"})
    df["split"] = split
    return df.sort_values(["video", "frame_id"]).reset_index(drop=True)


def boxes_by_image(split):
    """Map image_id -> (targets, ignored) where targets are [cls, x1, y1, x2, y2] in pixels."""
    targets, ignored = {}, {}
    for ann in load_coco(split)["annotations"]:
        x, y, w, h = ann["bbox"]
        box = (x, y, x + w, y + h)
        if ann["iscrowd"]:
            # Ignored regions are stored once per class, keep a single copy
            ignored.setdefault(ann["image_id"], set()).add(box)
        else:
            targets.setdefault(ann["image_id"], []).append((ann["category_id"] - 1, *box))

    targets = {k: np.array(v, dtype=np.float32) for k, v in targets.items()}
    ignored = {k: np.array(sorted(v), dtype=np.float32) for k, v in ignored.items()}
    return targets, ignored


# ---------------------------------------------------------------- folds and sampling

def make_folds(train_images):
    """Sequence-level folds: all frames of one clip share a fold.

    GroupKFold is deterministic, so this reproduces the team fold manifest
    (held-out sizes 8080 / 8082 / 8039 frames).
    """
    folds = np.full(len(train_images), -1)
    splitter = GroupKFold(n_splits=N_FOLDS)
    splits = splitter.split(train_images, groups=train_images["video"])
    for fold, (_, holdout_idx) in enumerate(splits):
        folds[holdout_idx] = fold
    return train_images.assign(fold=folds)


def sample_frames(images, step):
    """Keep every step-th frame of every clip, always starting from the first frame."""
    position = images.groupby("video").cumcount()
    return images[position % step == 0].reset_index(drop=True)


# ---------------------------------------------------------------- masking

def mask_ignored(image, ignored, targets):
    """Fill ignored regions with gray, then restore pixels of target boxes on top.

    Restoring keeps every labelled object intact even if it overlaps an ignored region.
    """
    if len(ignored) == 0:
        return image

    masked = image.copy()
    for x1, y1, x2, y2 in ignored.astype(int):
        masked[max(y1, 0):y2, max(x1, 0):x2] = MASK_VALUE

    if len(targets):
        for x1, y1, x2, y2 in np.floor(targets[:, 1:]).astype(int):
            rows = slice(max(y1, 0), y2 + 1)
            cols = slice(max(x1, 0), x2 + 1)
            masked[rows, cols] = image[rows, cols]
    return masked


def yolo_label_lines(targets, width, height):
    lines = []
    for cls, x1, y1, x2, y2 in targets:
        cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
        w, h = (x2 - x1) / width, (y2 - y1) / height
        lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


# ---------------------------------------------------------------- Ultralytics export

def _link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _export_frame(row, out_images, out_labels, targets, ignored, mask):
    src = find_data_root() / row.split / row.file_name
    stem = f"{row.video}_{row.frame_id:07d}"
    dst = out_images / f"{stem}.jpg"

    frame_targets = targets.get(row.image_id, np.zeros((0, 5), np.float32))
    frame_ignored = ignored.get(row.image_id, np.zeros((0, 4), np.float32))

    if not dst.exists():
        if mask and len(frame_ignored):
            image = cv2.imread(str(src))
            cv2.imwrite(str(dst), mask_ignored(image, frame_ignored, frame_targets),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            _link_or_copy(src, dst)

    lines = yolo_label_lines(frame_targets, row.width, row.height)
    (out_labels / f"{stem}.txt").write_text("\n".join(lines))


def export_yolo_dataset(out_dir, parts, mask=True, workers=8):
    """Write an Ultralytics dataset.

    parts maps a subset name ("train", "val") to a frames table from images_table().
    Labels are rebuilt from the COCO file, ignored regions are masked when mask=True.
    """
    out_dir = Path(out_dir)
    splits = {split for frames in parts.values() for split in frames["split"].unique()}
    boxes = {split: boxes_by_image(split) for split in splits}

    for subset, frames in parts.items():
        out_images = out_dir / "images" / subset
        out_labels = out_dir / "labels" / subset
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        with ThreadPoolExecutor(workers) as pool:
            jobs = [
                pool.submit(_export_frame, row, out_images, out_labels, *boxes[row.split], mask)
                for row in frames.itertuples()
            ]
            for job in jobs:
                job.result()

    yaml_lines = [f"path: {out_dir.resolve().as_posix()}"]
    yaml_lines += [f"{subset}: images/{subset}" for subset in parts]
    yaml_lines += [f"nc: {len(CLASS_NAMES)}", f"names: {CLASS_NAMES}"]
    (out_dir / "data.yaml").write_text("\n".join(yaml_lines) + "\n")
    return out_dir / "data.yaml"


# ---------------------------------------------------------------- evaluation ground truth

def export_coco_subset(out_path, frames):
    """COCO ground truth for the given frames of one split, ignored regions kept as iscrowd=1."""
    split = frames["split"].iloc[0]
    coco = load_coco(split)
    keep = set(frames["image_id"])

    subset = {
        "info": coco["info"],
        "categories": coco["categories"],
        "images": [img for img in coco["images"] if img["id"] in keep],
        "annotations": [ann for ann in coco["annotations"] if ann["image_id"] in keep],
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(subset))
    return out_path
