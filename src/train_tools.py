"""Training of Ultralytics detectors from experiment configs, with optional loss changes.

An experiment config (configs/<family>/<name>.yaml) looks like:

    name: af1_imgsz1152
    family: anchor_free
    arch: yolo                  # yolo or rtdetr
    model: yolo11s.pt           # .pt checkpoint or model .yaml
    weights: yolo11s.pt         # optional, pretrained weights loaded into a .yaml model
    data: data/yolo/fold0_s10/data.yaml
    imgsz: 1152
    train: {epochs: 30, batch: 8}   # passed to Ultralytics train() as is
    loss: {tal_topk: 10, nwd_ratio: 0.0, nwd_constant: 12.8}   # optional, YOLO only
"""

import json

import torch
import yaml
from ultralytics import RTDETR, YOLO
from ultralytics.utils.loss import BboxLoss, v8DetectionLoss
from ultralytics.utils.torch_utils import unwrap_model

from .common import REPO_ROOT, RUNS_DIR, SEED


# ---------------------------------------------------------------- NWD box loss

def normalized_wasserstein(pred, target, constant, eps=1e-7):
    """Normalized Wasserstein Distance between xyxy boxes (Wang et al., 2021, arXiv:2110.13389).

    Each box is modelled as a 2D Gaussian with mean at the box centre and std w/2, h/2.
    The squared 2-Wasserstein distance of two such Gaussians is the squared L2 distance
    of (cx, cy, w/2, h/2). Unlike IoU it changes smoothly for tiny boxes that barely overlap.
    """
    pred_c = (pred[:, :2] + pred[:, 2:]) / 2
    target_c = (target[:, :2] + target[:, 2:]) / 2
    pred_half = (pred[:, 2:] - pred[:, :2]) / 2
    target_half = (target[:, 2:] - target[:, :2]) / 2

    w2 = ((pred_c - target_c) ** 2).sum(-1) + ((pred_half - target_half) ** 2).sum(-1)
    return torch.exp(-torch.sqrt(w2 + eps) / constant)


class NWDBboxLoss(BboxLoss):
    """Box loss = (1 - ratio) * CIoU loss + ratio * NWD loss, the DFL term is unchanged."""

    def __init__(self, reg_max, ratio, constant):
        super().__init__(reg_max)
        self.ratio = ratio
        self.constant = constant

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, imgsz, stride):
        loss_iou, loss_dfl = super().forward(
            pred_dist, pred_bboxes, anchor_points, target_bboxes,
            target_scores, target_scores_sum, fg_mask, imgsz, stride,
        )

        idx = fg_mask.nonzero(as_tuple=True)
        weight = target_scores[idx].sum(-1, keepdim=True)

        # Boxes arrive in feature-grid units, NWD needs pixels because its constant is in pixels
        scale = stride[idx[1]]
        pred_px, target_px = pred_bboxes[idx] * scale, target_bboxes[idx] * scale
        nwd = normalized_wasserstein(pred_px, target_px, self.constant)
        loss_nwd = ((1.0 - nwd).unsqueeze(-1) * weight).sum() / target_scores_sum

        return (1.0 - self.ratio) * loss_iou + self.ratio * loss_nwd, loss_dfl


def loss_tweak_callback(tal_topk=10, nwd_ratio=0.0, nwd_constant=12.8):
    """Replace the YOLO criterion right before the first training batch.

    tal_topk is the number of candidate locations per object in the task-aligned assigner
    (Ultralytics default 10). Only the training model gets the new criterion, so saved
    checkpoints stay plain Ultralytics models.
    """
    def on_train_start(trainer):
        model = unwrap_model(trainer.model)
        criterion = v8DetectionLoss(model, tal_topk=tal_topk)
        if nwd_ratio > 0:
            reg_max = model.model[-1].reg_max
            criterion.bbox_loss = NWDBboxLoss(reg_max, nwd_ratio, nwd_constant).to(criterion.device)
        model.criterion = criterion

    return on_train_start


# ---------------------------------------------------------------- experiment runner

def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(cfg):
    model_cls = RTDETR if cfg.get("arch", "yolo") == "rtdetr" else YOLO
    model_path = str(REPO_ROOT / cfg["model"]) if cfg["model"].endswith(".yaml") else cfg["model"]
    model = model_cls(model_path)
    if cfg.get("weights"):
        model.load(cfg["weights"])
    return model


def run_dir(cfg):
    return RUNS_DIR / cfg["family"] / cfg["name"]


def train_experiment(cfg, resume=False):
    out_dir = run_dir(cfg)
    if resume:
        # Continue an interrupted run; Ultralytics keeps the saved settings except batch and workers
        model_cls = RTDETR if cfg.get("arch", "yolo") == "rtdetr" else YOLO
        model = model_cls(str(out_dir / "weights" / "last.pt"))
    else:
        model = build_model(cfg)
    if cfg.get("loss"):
        model.add_callback("on_train_start", loss_tweak_callback(**cfg["loss"]))

    if resume:
        model.train(resume=True, **cfg.get("train", {}))
        return out_dir / "weights" / "best.pt"

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "experiment.json").write_text(json.dumps(cfg, indent=2))

    model.train(
        data=str(REPO_ROOT / cfg["data"]),
        imgsz=cfg["imgsz"],
        project=str(out_dir.parent),
        name=out_dir.name,
        exist_ok=True,
        seed=SEED,
        deterministic=True,
        **cfg.get("train", {}),
    )
    return out_dir / "weights" / "best.pt"
