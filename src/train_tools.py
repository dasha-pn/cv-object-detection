"""Training of Ultralytics detectors from experiment configs, with optional loss changes.

An experiment config (configs/<family>/<name>.yaml) looks like:

    name: af1_imgsz1152
    family: anchor_free
    arch: yolo                  # yolo, rtdetr or dfine (see src/dfine.py)
    model: yolo11s.pt           # .pt checkpoint or model .yaml
    weights: yolo11s.pt         # optional, pretrained weights loaded into a .yaml model
    data: data/yolo/fold0_s10/data.yaml
    imgsz: 1152
    train: {epochs: 30, batch: 8}   # passed to Ultralytics train() as is
    loss: {tal_topk: 10, nwd_ratio: 0.0, nwd_constant: 12.8}   # optional, YOLO only
    backbone_lr_scale: 0.1      # optional, backbone lr relative to the rest
    extra_levels: 1             # optional, RT-DETR decoder reads this many new finer levels first
    silent_start: false         # optional, new levels start with default instead of zero BatchNorm scale
    checkpoint: last            # optional, score last.pt instead of best.pt
    evaluate: false             # optional, do not score (trained on the evaluation clips)
"""

import functools
import json

import torch
import yaml
from ultralytics import RTDETR, YOLO
from ultralytics.nn.modules.head import RTDETRDecoder
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils.loss import BboxLoss, v8DetectionLoss
from ultralytics.utils.torch_utils import intersect_dicts, unwrap_model

from .common import REPO_ROOT, RUNS_DIR, SEED
from .dfine import DFineDetector, train_dfine


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


# ---------------------------------------------------------------- separate backbone learning rate

def backbone_lr_callback(scale):
    """Train the backbone with `scale` times the learning rate of the rest of the model.

    RT-DETR is trained this way in the original paper (backbone lr 0.1x): the ImageNet/COCO
    backbone only needs small adjustments, while the encoder and decoder adapt to the new task.
    Every Ultralytics parameter group (weights, norm weights, biases) is split into a backbone
    part and a rest part right when the optimizer is built. The scheduler is created afterwards
    and keeps the per-group rates, and a resumed run gets the same groups as its checkpoint.
    """
    def on_pretrain_routine_start(trainer):
        build_optimizer = trainer.build_optimizer

        def build_split_optimizer(**kwargs):
            optimizer = build_optimizer(**kwargs)
            model = unwrap_model(kwargs["model"])
            n_backbone = len(model.yaml["backbone"])
            backbone = {id(p) for layer in model.model[:n_backbone] for p in layer.parameters()}

            groups = []
            for group in optimizer.param_groups:
                for in_backbone in (False, True):
                    params = [p for p in group["params"] if (id(p) in backbone) == in_backbone]
                    if params:
                        factor = scale if in_backbone else 1.0
                        groups.append({**group, "params": params, "lr": group["lr"] * factor})
            optimizer.param_groups = groups
            return optimizer

        trainer.build_optimizer = build_split_optimizer

    return on_pretrain_routine_start


# ---------------------------------------------------------------- RT-DETR with extra feature levels

def add_levels(value, n_heads, n_points, n_new, kind):
    """Extend a deformable-attention tensor by n_new levels in front of the existing ones.

    Rows are ordered (head, level, point[, xy]). The new levels are copies of the old finest
    level. Their attention logits start 2 lower, so the randomly initialised new branch gets
    about a seventh of the weight of P3 at first and does not disturb the pretrained decoder.
    """
    tail = value.shape[1:]
    per_level = n_points * (2 if kind == "offsets" else 1)
    grouped = value.reshape(n_heads, -1, per_level, *tail)
    new = grouped[:, :1].repeat(1, n_new, *[1] * (grouped.dim() - 2))
    if kind == "attention_bias":
        new = new - 2.0
    return torch.cat([new, grouped], 1).reshape(-1, *tail)


def load_with_extra_levels(model, weights, silent=True):
    """Load pretrained RT-DETR weights into a model whose decoder reads extra, finer levels first.

    The new layers are appended after the pretrained ones, so all other layers load by name. Only
    the decoder changes: it moves to a new layer index, its input projections shift by the number
    of new levels, and its cross-attention gets offsets and weights for the new levels. With
    silent=False the projections of the new levels keep the default initialisation.
    """
    det = model.model
    src, _ = load_checkpoint(weights)
    src_prefix, dst_prefix = f"model.{len(src.model) - 1}.", f"model.{len(det.model) - 1}."
    head = det.model[-1]
    n_new = head.nl - src.model[-1].nl
    attn = head.decoder.layers[0].cross_attn

    csd = {}
    for key, value in src.float().state_dict().items():
        if not key.startswith(src_prefix):
            csd[key] = value
            continue
        rest = key[len(src_prefix):]
        if rest.startswith("input_proj."):
            level, tail = rest[len("input_proj."):].split(".", 1)
            rest = f"input_proj.{int(level) + n_new}.{tail}"
        elif "cross_attn.sampling_offsets" in rest:
            value = add_levels(value, attn.n_heads, attn.n_points, n_new, "offsets")
        elif "cross_attn.attention_weights" in rest:
            kind = "attention_bias" if rest.endswith("bias") else "attention_weight"
            value = add_levels(value, attn.n_heads, attn.n_points, n_new, kind)
        csd[dst_prefix + rest] = value

    n_classes = det._remap_cls_by_names(csd, src, verbose=False)
    matched = intersect_dicts(csd, det.state_dict())
    det.load_state_dict(matched, strict=False)
    print(f"transferred {len(matched) + n_classes}/{len(det.state_dict())} tensors from {weights}")

    # The new levels start silent: a zero BatchNorm scale makes their projected features zero,
    # so the pretrained decoder works as before, and the scale grows as the new branch learns
    if silent:
        for level in range(n_new):
            torch.nn.init.zeros_(head.input_proj[level][1].weight)

    # Marks the model as pretrained, so that train() starts from these weights
    model.load(det)
    return model


def anchor_fix_callback(n_new):
    """Keep the anchor priors of the pretrained levels in a decoder with n_new extra levels in front.

    The decoder gives the initial boxes of level i the size 0.05 * 2^i of the image, so a new
    level 0 would double the priors of P3-P5. Halving the base for every new level gives P2 0.025
    and keeps P3-P5 at 0.05, 0.1 and 0.2. The setting is an attribute of the decoder, so it is
    saved with every checkpoint.
    """
    anchors = functools.partial(RTDETRDecoder._generate_anchors, grid_size=0.05 / 2 ** n_new)

    def on_pretrain_routine_end(trainer):
        for model in (trainer.model, trainer.ema.ema):
            unwrap_model(model).model[-1]._generate_anchors = anchors
    return on_pretrain_routine_end


# ---------------------------------------------------------------- experiment runner

def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(cfg):
    model_cls = RTDETR if cfg.get("arch", "yolo") == "rtdetr" else YOLO
    model_path = str(REPO_ROOT / cfg["model"]) if cfg["model"].endswith(".yaml") else cfg["model"]
    model = model_cls(model_path)
    if cfg.get("weights"):
        # With the dataset class names set, Ultralytics copies the pretrained classifier rows of
        # classes with the same name (person, bicycle, car, truck, bus) instead of dropping them
        with open(REPO_ROOT / cfg["data"], encoding="utf-8") as f:
            model.model.names = dict(enumerate(yaml.safe_load(f)["names"]))
        if cfg.get("extra_levels"):
            load_with_extra_levels(model, cfg["weights"], silent=cfg.get("silent_start", True))
        else:
            model.load(cfg["weights"])
    return model


def fold_config(cfg, fold):
    """The same experiment with fold `fold` held out instead of fold 0, for cross-validation.

    Data, name and holdout change, everything else stays. Fold 0 is the original run.
    """
    if fold == 0:
        return cfg
    return {**cfg, "name": f"{cfg['name']}_fold{fold}", "holdout": f"fold{fold}",
            "data": cfg["data"].replace("fold0_", f"fold{fold}_")}


def eval_sets(cfg):
    """Evaluation sets of a run: its fold holdout and val."""
    return (cfg.get("holdout", "fold0"), "val")


def run_dir(cfg):
    return RUNS_DIR / cfg["family"] / cfg["name"]


def checkpoint(cfg):
    """Weights a run is scored with: best.pt chosen on the curve set, or last.pt (`checkpoint: last`)
    for runs whose curve set is part of the training data."""
    return run_dir(cfg) / "weights" / f"{cfg.get('checkpoint', 'best')}.pt"


def load_detector(cfg):
    """Trained model of a run, every architecture has the predict() call of Ultralytics."""
    arch = cfg.get("arch", "yolo")
    if arch == "dfine":
        return DFineDetector(cfg["model"], checkpoint(cfg), **cfg.get("model_config", {}))
    return (RTDETR if arch == "rtdetr" else YOLO)(str(checkpoint(cfg)))


def train_experiment(cfg, resume=False):
    if cfg.get("arch") == "dfine":
        return train_dfine(cfg)

    out_dir = run_dir(cfg)
    if resume:
        # Continue an interrupted run; Ultralytics keeps the saved settings except batch and workers
        last = out_dir / "weights" / "last.pt"
        # A finished run has no optimizer state, and Ultralytics would silently start a demo training
        if torch.load(last, map_location="cpu", weights_only=False).get("optimizer") is None:
            raise RuntimeError(f"{last} belongs to a finished run, there is nothing to resume")
        model_cls = RTDETR if cfg.get("arch", "yolo") == "rtdetr" else YOLO
        model = model_cls(str(last))
    else:
        model = build_model(cfg)
    if cfg.get("loss"):
        model.add_callback("on_train_start", loss_tweak_callback(**cfg["loss"]))
    if cfg.get("backbone_lr_scale"):
        model.add_callback("on_pretrain_routine_start", backbone_lr_callback(cfg["backbone_lr_scale"]))
    if cfg.get("extra_levels"):
        model.add_callback("on_pretrain_routine_end", anchor_fix_callback(cfg["extra_levels"]))

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
