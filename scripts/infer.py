"""Score an inference strategy or an ensemble on the evaluation sets, optionally predict the test set.

Every result is saved in runs/inference/<name>/ (all detections and metrics.json), so that
ensembles can reuse the detections without running the models again.

Examples:
    python scripts/infer.py configs/query_based/qb0_rtdetr_l_1152.yaml --strategy plain --imgsz 1344
    python scripts/infer.py configs/query_based/qb0_rtdetr_l_1152.yaml --strategy tiles --merge wbf
    python scripts/infer.py --ensemble qb0_rtdetr_l_1152_plain_1152 af4_topk5_1152_plain_1152 --weights 2 1
    python scripts/infer.py configs/query_based/qb3_bblr_nobiaswarm_1152.yaml --strategy tiles --fold 1
    add --test to also predict the test frames and write submissions/<name>.csv
    (--test-only skips the evaluation sets)
    --fold k uses the run trained without fold k and saves to runs/inference_cv/
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import REPO_ROOT, RUNS_DIR  # noqa: E402
from src.evaluate import coco_evaluate, format_metrics  # noqa: E402
from src.inference import ensemble, run_strategy, strategy  # noqa: E402
from src.predict import ALL_SETS, EVAL_SETS, eval_images  # noqa: E402
from src.submission import test_images, write_submission  # noqa: E402
from src.train_tools import eval_sets, fold_config, load_config, load_detector  # noqa: E402

INFERENCE_DIR = RUNS_DIR / "inference"
CV_INFERENCE_DIR = RUNS_DIR / "inference_cv"


def strategy_name(cfg, args, imgsz):
    merge = f"_{args.merge}" if args.strategy not in ("plain", "nms") else ""
    grid = f"_grid{args.grid}" if args.strategy.startswith("tiles") and args.grid != 2 else ""
    tiled = args.strategy.startswith("tiles")
    min_width = f"_minw{args.tile_min_width}" if tiled and args.tile_min_width else ""
    max_width = f"_maxw{args.tile_max_width}" if tiled and args.tile_max_width else ""
    return f"{cfg['name']}_{args.strategy}{merge}{grid}{min_width}{max_width}_{imgsz}"


def load_detections(name, image_set):
    return json.loads((INFERENCE_DIR / name / f"detections_{image_set}.json").read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", help="experiment config of the trained model")
    parser.add_argument("--strategy", default="plain", choices=["plain", "nms", "flip", "tiles", "tiles_flip"])
    parser.add_argument("--merge", default="nms", choices=["nms", "wbf"],
                        help="how flip or tile predictions are merged")
    parser.add_argument("--grid", type=int, default=2, help="tiles per side for the tile strategies")
    parser.add_argument("--tile-min-width", type=int, default=0,
                        help="frames narrower than this get the plain pass in the tile strategies")
    parser.add_argument("--tile-max-width", type=int, default=0,
                        help="frames at least this wide get the plain pass in the tile strategies")
    parser.add_argument("--imgsz", type=int, help="inference size, defaults to the training size")
    parser.add_argument("--ensemble", nargs="+", help="names of saved results to fuse with WBF")
    parser.add_argument("--weights", nargs="+", type=float, help="ensemble weight of every source")
    parser.add_argument("--name", help="result name, generated when omitted")
    parser.add_argument("--test", action="store_true", help="also predict the test set")
    parser.add_argument("--test-only", action="store_true",
                        help="only predict the test set, for a model trained on the evaluation clips")
    parser.add_argument("--fold", type=int, default=0, help="cross-validation fold of the trained run")
    args = parser.parse_args()

    set_names = EVAL_SETS
    if args.ensemble:
        name = args.name or "ens_" + "+".join(args.ensemble)
        model = None
    else:
        cfg = fold_config(load_config(args.config), args.fold)
        set_names = eval_sets(cfg)
        imgsz = args.imgsz or cfg["imgsz"]
        name = args.name or strategy_name(cfg, args, imgsz)
        model = load_detector(cfg)
        predict = strategy(args.strategy, imgsz, merge=args.merge, grid=args.grid,
                           min_tile_width=args.tile_min_width, max_tile_width=args.tile_max_width)

    image_sets = {} if args.test_only else {set_name: eval_images(set_name) for set_name in set_names}
    if args.test or args.test_only:
        image_sets["test"] = test_images()

    out_dir = (INFERENCE_DIR if args.fold == 0 else CV_INFERENCE_DIR) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"sources": args.ensemble, "weights": args.weights} if args.ensemble else {
        "config": args.config, "strategy": args.strategy, "merge": args.merge, "grid": args.grid,
        "tile_min_width": args.tile_min_width, "tile_max_width": args.tile_max_width, "imgsz": imgsz}

    for image_set, images in image_sets.items():
        if model is None:
            ids = [image_id for image_id, _ in images]
            sources = [load_detections(source, image_set) for source in args.ensemble]
            detections, ms_per_image = ensemble(sources, ids, args.weights), None
        else:
            detections, ms_per_image = run_strategy(model, images, predict)
        (out_dir / f"detections_{image_set}.json").write_text(json.dumps(detections))

        if image_set == "test":
            out_path = REPO_ROOT / "submissions" / f"{name}.csv"
            write_submission(detections, [image_id for image_id, _ in images], out_path)
            print(f"saved {out_path}")
            continue

        metrics = coco_evaluate(ALL_SETS[image_set][1], detections)
        metrics["ms_per_image"] = ms_per_image
        report[image_set] = metrics
        speed = f"  {ms_per_image:.1f} ms/img" if ms_per_image else ""
        print(f"{name} [{image_set}] {format_metrics(metrics)}{speed}")

    if not args.test_only:
        (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
