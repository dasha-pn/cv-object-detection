"""Train one experiment and score its checkpoint (best.pt unless the config says last) on the fixed evaluation sets.

Examples:
    python scripts/train.py configs/anchor_free/af0_baseline_640.yaml
    python scripts/train.py configs/anchor_free/af0_baseline_640.yaml --eval-only
    python scripts/train.py configs/anchor_free/af2_p2_1152.yaml --resume
    python scripts/train.py configs/query_based/qb3_bblr_nobiaswarm_1152.yaml --fold 1
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import RTDETR, YOLO  # noqa: E402

from src.evaluate import format_metrics  # noqa: E402
from src.predict import evaluate_model  # noqa: E402
from src.train_tools import checkpoint, eval_sets, fold_config, load_config, run_dir, train_experiment  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="+", help="one or more experiment configs, run in order")
    parser.add_argument("--eval-only", action="store_true",
                        help="skip training, score the existing checkpoint")
    parser.add_argument("--resume", action="store_true",
                        help="continue an interrupted run from its last.pt")
    parser.add_argument("--fold", type=int, default=0,
                        help="held-out fold for cross-validation (needs prepare_data.py --fold)")
    args = parser.parse_args()

    for config_path in args.config:
        cfg = fold_config(load_config(config_path), args.fold)
        out_dir = run_dir(cfg)
        print(f"\n=== {cfg['name']} ===")

        if not args.eval_only:
            train_experiment(cfg, resume=args.resume)

        # A run trained on the evaluation clips cannot be scored on them
        if not cfg.get("evaluate", True):
            continue

        model_cls = RTDETR if cfg.get("arch", "yolo") == "rtdetr" else YOLO
        model = model_cls(str(checkpoint(cfg)))
        report = evaluate_model(model, cfg["imgsz"], eval_sets=eval_sets(cfg), save_dir=out_dir)

        for eval_set, metrics in report.items():
            speed = f"{metrics['ms_per_image']:.1f} ms/img"
            print(f"{cfg['name']} [{eval_set}] {format_metrics(metrics)}  {speed}")


if __name__ == "__main__":
    main()
