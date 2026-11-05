"""Temporal rescoring with a trained model.

Predicts every evaluation frame together with its neighbouring frames (cached in
runs/temporal/<name>/), scores a grid of rescoring settings on fold 0 and val, and with
--test rescores the consecutive test frames and writes a submission.

Examples:
    python scripts/temporal.py configs/query_based/qb0_rtdetr_l_1152.yaml
    python scripts/temporal.py configs/query_based/qb0_rtdetr_l_1152.yaml --test --radius 2 --iou 0.5
    python scripts/temporal.py configs/query_based/qb3_bblr_nobiaswarm_1152.yaml --strategy tiles --fold 1
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import RTDETR, YOLO  # noqa: E402

from src.common import REPO_ROOT, RUNS_DIR  # noqa: E402
from src.evaluate import coco_evaluate, format_metrics  # noqa: E402
from src.inference import from_coco, run_strategy, strategy, to_coco  # noqa: E402
from src.predict import ALL_SETS  # noqa: E402
from src.submission import write_submission  # noqa: E402
from src.temporal import frames_table, neighbours, rescore  # noqa: E402
from src.train_tools import checkpoint, eval_sets, fold_config, load_config  # noqa: E402

TEMPORAL_DIR = RUNS_DIR / "temporal"
CV_TEMPORAL_DIR = RUNS_DIR / "temporal_cv"
MAX_RADIUS = 3
GRID = {"radius": [1, 2, 3], "iou_thr": [0.3, 0.5], "own_weight": [1.0], "mode": ["mean", "smooth"]}


def cached_predictions(model, predict, frames, image_ids, path):
    """Predictions of the given frames as {image_id: prediction}, computed once and cached."""
    if path.exists():
        return from_coco(json.loads(path.read_text()))
    images = [(row.image_id, row.path) for row in frames[frames["image_id"].isin(set(image_ids))].itertuples()]
    detections, _ = run_strategy(model, images, predict)
    path.write_text(json.dumps(detections))
    return from_coco(detections)


def to_detections(preds):
    return [det for image_id, pred in preds.items() for det in to_coco(pred, image_id)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="experiment config of the trained model")
    parser.add_argument("--strategy", default="plain", help="per-frame inference strategy")
    parser.add_argument("--imgsz", type=int, help="inference size, defaults to the training size")
    parser.add_argument("--tile-min-width", type=int, default=0,
                        help="frames narrower than this get the plain pass in the tile strategies")
    parser.add_argument("--test", action="store_true", help="rescore the test frames and write a submission")
    parser.add_argument("--radius", type=int, default=2, help="test: frames on each side")
    parser.add_argument("--iou", type=float, default=0.5, help="test: IoU to match boxes of two frames")
    parser.add_argument("--own-weight", type=float, default=1.0, help="test: weight of the frame itself")
    parser.add_argument("--mode", default="mean", choices=["mean", "smooth"], help="test: rescoring mode")
    parser.add_argument("--test-only", action="store_true",
                        help="skip the grid, for a model trained on the evaluation clips")
    parser.add_argument("--fold", type=int, default=0, help="cross-validation fold of the trained run")
    args = parser.parse_args()

    cfg = fold_config(load_config(args.config), args.fold)
    imgsz = args.imgsz or cfg["imgsz"]
    min_width = f"_minw{args.tile_min_width}" if args.tile_min_width else ""
    name = f"{cfg['name']}_{args.strategy}{min_width}_{imgsz}"
    out_dir = (TEMPORAL_DIR if args.fold == 0 else CV_TEMPORAL_DIR) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    model_cls = RTDETR if cfg.get("arch", "yolo") == "rtdetr" else YOLO
    model = model_cls(str(checkpoint(cfg)))
    predict = strategy(args.strategy, imgsz, min_tile_width=args.tile_min_width)

    # Grid of settings on the evaluation sets; radius 0 is the model without rescoring
    report = []
    grid_sets = {} if args.test_only else {set_name: ALL_SETS[set_name] for set_name in eval_sets(cfg)}
    for eval_set, (split, gt_path) in grid_sets.items():
        frames = frames_table(split)
        centre_ids = [img["id"] for img in json.loads(gt_path.read_text())["images"]]
        window = neighbours(frames, centre_ids, MAX_RADIUS)
        needed = set(centre_ids) | {image_id for ids in window.values() for image_id, _ in ids}
        preds = cached_predictions(model, predict, frames, needed, out_dir / f"detections_{eval_set}.json")

        settings = [{"radius": 0, "iou_thr": None, "own_weight": None, "mode": None}]
        settings += [dict(zip(GRID, values)) for values in itertools.product(*GRID.values())]
        for setting in settings:
            if setting["radius"] == 0:
                rescored = {image_id: preds.get(image_id) for image_id in centre_ids if image_id in preds}
            else:
                rescored = rescore(preds, window, **setting)
            metrics = coco_evaluate(gt_path, to_detections(rescored))
            report.append({"eval_set": eval_set, **setting, **metrics})
            print(f"{eval_set} {setting} {format_metrics(metrics)}")
    if report:
        (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))

    if args.test or args.test_only:
        frames = frames_table("test")
        window = neighbours(frames, frames["image_id"], args.radius)
        preds = cached_predictions(model, predict, frames, set(frames["image_id"]), out_dir / "detections_test.json")
        rescored = rescore(preds, window, args.radius, args.iou, args.own_weight, args.mode)
        out_path = REPO_ROOT / "submissions" / f"{name}_temporal_{args.mode}_r{args.radius}_iou{args.iou}.csv"
        write_submission(to_detections(rescored), list(frames["image_id"]), out_path)
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
