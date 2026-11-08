"""Predict the test frames with a trained experiment and write a Kaggle submission CSV.

Examples:
    python scripts/make_submission.py configs/anchor_free/af1_imgsz1152.yaml
    python scripts/make_submission.py configs/anchor_free/af1_imgsz1152.yaml --imgsz 1280
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import REPO_ROOT  # noqa: E402
from src.predict import predict_images  # noqa: E402
from src.submission import test_images, write_submission  # noqa: E402
from src.train_tools import load_config, load_detector  # noqa: E402

SUBMISSIONS_DIR = REPO_ROOT / "submissions"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="experiment config of the trained model")
    parser.add_argument("--imgsz", type=int, help="inference size, defaults to the training size")
    args = parser.parse_args()

    cfg = load_config(args.config)
    imgsz = args.imgsz or cfg["imgsz"]
    model = load_detector(cfg)

    images = test_images()
    detections, ms_per_image = predict_images(model, images, imgsz)

    out_path = SUBMISSIONS_DIR / f"{cfg['name']}_{imgsz}.csv"
    write_submission(detections, [image_id for image_id, _ in images], out_path)
    print(f"{len(images)} images, {len(detections)} detections, {ms_per_image:.1f} ms/img")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
