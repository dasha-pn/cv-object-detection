"""Train one experiment and score its best checkpoint on the fixed evaluation sets.

Examples:
    python scripts/train.py configs/anchor_free/af0_baseline_640.yaml
    python scripts/train.py configs/anchor_free/af0_baseline_640.yaml --eval-only
    python scripts/train.py configs/anchor_free/af2_p2_1152.yaml --resume
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import RTDETR, YOLO  # noqa: E402

from src.evaluate import format_metrics  # noqa: E402
from src.predict import evaluate_model  # noqa: E402
from src.train_tools import load_config, run_dir, train_experiment  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="+", help="one or more experiment configs, run in order")
    parser.add_argument("--eval-only", action="store_true",
                        help="skip training, score the existing best.pt")
    parser.add_argument("--resume", action="store_true",
                        help="continue an interrupted run from its last.pt")
    args = parser.parse_args()

    for config_path in args.config:
        cfg = load_config(config_path)
        out_dir = run_dir(cfg)
        print(f"\n=== {cfg['name']} ===")

        if not args.eval_only:
            train_experiment(cfg, resume=args.resume)

        model_cls = RTDETR if cfg.get("arch", "yolo") == "rtdetr" else YOLO
        model = model_cls(str(out_dir / "weights" / "best.pt"))
        report = evaluate_model(model, cfg["imgsz"], save_dir=out_dir)

        for eval_set, metrics in report.items():
            speed = f"{metrics['ms_per_image']:.1f} ms/img"
            print(f"{cfg['name']} [{eval_set}] {format_metrics(metrics)}  {speed}")


if __name__ == "__main__":
    main()
