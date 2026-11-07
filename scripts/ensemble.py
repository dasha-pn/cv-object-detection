"""Fuse the saved detections of several runs with WBF and write a submission.

The sources are result names of scripts/infer.py in runs/inference/ or runs/inference_cv/. Boxes
below --min-score are dropped when a source is loaded: they cost a lot of memory and time and
hardly change the fused ranking. val is scored for every set of sources that has it, and
with --test the test frames are fused, rescored with the neighbouring frames and saved.

Examples:
    python scripts/ensemble.py qb3_bblr_nobiaswarm_1152_plain_1152 p3_rtdetr_s5_short_plain_1152
    python scripts/ensemble.py f1_rtdetr_full_s5_plain_1152 p3_rtdetr_s5_short_plain_1152 --test --name ens2
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import REPO_ROOT, RUNS_DIR  # noqa: E402
from src.evaluate import coco_evaluate, format_metrics  # noqa: E402
from src.inference import EMPTY, from_coco, to_coco, weighted_boxes_fusion  # noqa: E402
from src.predict import ALL_SETS  # noqa: E402
from src.submission import test_images, write_submission  # noqa: E402
from src.temporal import frames_table, neighbours, rescore  # noqa: E402

RESULT_DIRS = [RUNS_DIR / "inference", RUNS_DIR / "inference_cv"]


def source_path(name, image_set):
    """Saved detections of one result, or None when it has no such set."""
    paths = [folder / name / f"detections_{image_set}.json" for folder in RESULT_DIRS]
    return next((path for path in paths if path.exists()), None)


def load_source(path, min_score):
    detections = json.loads(path.read_text())
    return from_coco([det for det in detections if det["score"] >= min_score])


def fuse(names, image_set, image_ids, weights, iou_thr, min_score):
    """Fused prediction of every image, as {image_id: prediction}."""
    sources = [load_source(source_path(name, image_set), min_score) for name in names]
    return {image_id: weighted_boxes_fusion([source.get(image_id, EMPTY) for source in sources], weights, iou_thr)
            for image_id in image_ids}


def to_detections(preds):
    return [det for image_id, pred in preds.items() for det in to_coco(pred, image_id)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", nargs="+", help="result names of scripts/infer.py")
    parser.add_argument("--weights", nargs="+", type=float, help="ensemble weight of every source")
    parser.add_argument("--iou", type=float, default=0.65, help="IoU that joins a box to a WBF cluster")
    parser.add_argument("--min-score", type=float, default=0.05, help="boxes below this score are dropped")
    parser.add_argument("--test", action="store_true", help="fuse the test frames and write a submission")
    parser.add_argument("--name", default="ensemble", help="name of the submission file")
    parser.add_argument("--radius", type=int, default=2, help="temporal rescoring: frames on each side")
    parser.add_argument("--temporal-iou", type=float, default=0.3, help="temporal rescoring: IoU of a match")
    parser.add_argument("--no-temporal", action="store_true", help="skip temporal rescoring")
    args = parser.parse_args()

    # Models trained on the evaluation clips have no saved val detections, so val needs all sources
    if all(source_path(name, "val") for name in args.sources):
        gt_path = ALL_SETS["val"][1]
        image_ids = [image["id"] for image in json.loads(gt_path.read_text())["images"]]
        preds = fuse(args.sources, "val", image_ids, args.weights, args.iou, args.min_score)
        print(f"{args.name} [val] {format_metrics(coco_evaluate(gt_path, to_detections(preds)))}")

    if args.test:
        image_ids = [image_id for image_id, _ in test_images()]
        preds = fuse(args.sources, "test", image_ids, args.weights, args.iou, args.min_score)
        if not args.no_temporal:
            frames = frames_table("test")
            window = neighbours(frames, frames["image_id"], args.radius)
            preds = rescore(preds, window, args.radius, args.temporal_iou, mode="smooth")

        out_path = REPO_ROOT / "submissions" / f"{args.name}.csv"
        write_submission(to_detections(preds), image_ids, out_path)
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
