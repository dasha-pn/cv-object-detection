"""Build Ultralytics datasets and COCO evaluation subsets.

Examples:
    python scripts/prepare_data.py --fold 0 --train-step 10
    python scripts/prepare_data.py --full --train-step 5
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import WORK_DIR  # noqa: E402
from src.data import (  # noqa: E402
    export_coco_subset, export_yolo_dataset, images_table, make_folds, sample_frames,
)

# Frame steps of the evaluation subsets, fixed for all experiments
HOLDOUT_EVAL_STEP = 10
VAL_EVAL_STEP = 5
# Frame step of the validation images Ultralytics scores after every epoch
CURVE_STEP = 20


def summary(name, frames):
    print(f"{name:<28} clips={frames['video'].nunique():>3}  frames={len(frames):>6}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0, help="held-out fold for tuning experiments")
    parser.add_argument("--train-step", type=int, default=10, help="keep every n-th training frame")
    parser.add_argument("--full", action="store_true", help="train on all train+val clips")
    parser.add_argument("--no-mask", action="store_true", help="keep ignored regions unmasked")
    args = parser.parse_args()

    train = make_folds(images_table("train"))
    val = images_table("val")

    folds_path = WORK_DIR / "folds" / "train_folds.csv"
    folds_path.parent.mkdir(parents=True, exist_ok=True)
    train[["image_id", "file_name", "video", "frame_id", "fold"]].to_csv(folds_path, index=False)

    mask_tag = "_nomask" if args.no_mask else ""

    # Fixed evaluation ground truth, built once on original (unmasked) frames
    holdout = sample_frames(train[train["fold"] == args.fold], HOLDOUT_EVAL_STEP)
    val_eval = sample_frames(val, VAL_EVAL_STEP)
    export_coco_subset(WORK_DIR / "eval" / f"fold{args.fold}_holdout.json", holdout)
    export_coco_subset(WORK_DIR / "eval" / "val.json", val_eval)
    summary(f"eval fold{args.fold} holdout", holdout)
    summary("eval val", val_eval)

    if args.full:
        name = f"full_s{args.train_step}{mask_tag}"
        fit = sample_frames(pd.concat([train, val], ignore_index=True), args.train_step)
        curve = sample_frames(val, CURVE_STEP)
    else:
        name = f"fold{args.fold}_s{args.train_step}{mask_tag}"
        fit = sample_frames(train[train["fold"] != args.fold], args.train_step)
        curve = sample_frames(train[train["fold"] == args.fold], CURVE_STEP)

    summary(f"{name} train", fit)
    summary(f"{name} val (curves)", curve)

    yaml_path = export_yolo_dataset(
        WORK_DIR / "yolo" / name, {"train": fit, "val": curve}, mask=not args.no_mask,
    )
    print("dataset config:", yaml_path)


if __name__ == "__main__":
    main()
