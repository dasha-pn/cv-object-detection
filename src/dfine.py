"""D-FINE detector (Peng et al., 2024, arXiv:2410.13842) through Hugging Face Transformers.

Ultralytics does not support this model, so it has its own data loading and training loop.
DFineDetector.predict() returns results with the fields of an Ultralytics result, so the
inference strategies, the scoring and the ensembles work on it unchanged.
"""

import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import yaml
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from transformers import DFineForObjectDetection

from .common import CLASS_NAMES, REPO_ROOT, RUNS_DIR, SEED

# Share of its area that a box must keep inside a crop to stay a training target
MIN_VISIBLE = 0.5


def load_pretrained(name, **config):
    """Pretrained model with a new classifier for our classes."""
    return DFineForObjectDetection.from_pretrained(name, num_labels=len(CLASS_NAMES),
                                                   ignore_mismatched_sizes=True, **config)


def to_input(image, imgsz):
    """RGB uint8 frame -> float tensor (3, imgsz, imgsz); the model expects values in 0-1 without normalisation."""
    resized = cv2.resize(image, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(resized).permute(2, 0, 1).float() / 255


# ---------------------------------------------------------------- training data

class FrameDataset(Dataset):
    """Training frames of a prepared YOLO dataset with a random flip, zoom and colour change.

    The zoom cuts a random window with the shape of the frame, so small objects are seen larger,
    as in the tiles at inference time.
    """

    def __init__(self, data_yaml, imgsz, min_crop):
        with open(REPO_ROOT / data_yaml, encoding="utf-8") as f:
            root = Path(yaml.safe_load(f)["path"])
        self.paths = sorted((root / "images" / "train").iterdir())
        self.labels = root / "labels" / "train"
        self.imgsz = imgsz
        self.min_crop = min_crop
        self.colour = v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)

    def __len__(self):
        return len(self.paths)

    def read_labels(self, path):
        """Classes and xyxy boxes relative to the frame."""
        label_path = self.labels / f"{path.stem}.txt"
        rows = np.loadtxt(label_path, ndmin=2) if label_path.stat().st_size else np.zeros((0, 5))
        centre, size = rows[:, 1:3], rows[:, 3:5]
        return rows[:, 0].astype(int), np.concatenate([centre - size / 2, centre + size / 2], axis=1)

    def random_crop(self, image, classes, boxes):
        scale = np.random.uniform(self.min_crop, 1.0)
        x1, y1 = np.random.uniform(0, 1 - scale, 2)
        height, width = image.shape[:2]
        window = image[int(y1 * height):int((y1 + scale) * height), int(x1 * width):int((x1 + scale) * width)]

        # Boxes move to the coordinates of the window, those mostly outside it are dropped
        moved = (boxes - [x1, y1, x1, y1]) / scale
        clipped = moved.clip(0, 1)
        area = np.prod(moved[:, 2:] - moved[:, :2], axis=1)
        visible = np.prod(clipped[:, 2:] - clipped[:, :2], axis=1)
        keep = visible >= MIN_VISIBLE * area
        return window, classes[keep], clipped[keep]

    def __getitem__(self, index):
        path = self.paths[index]
        image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        classes, boxes = self.read_labels(path)

        if np.random.rand() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            boxes = np.stack([1 - boxes[:, 2], boxes[:, 1], 1 - boxes[:, 0], boxes[:, 3]], axis=1)
        image, classes, boxes = self.random_crop(image, classes, boxes)

        size = boxes[:, 2:] - boxes[:, :2]
        centred = np.concatenate([boxes[:, :2] + size / 2, size], axis=1)
        target = {"class_labels": torch.as_tensor(classes, dtype=torch.long),
                  "boxes": torch.as_tensor(centred, dtype=torch.float32)}
        return self.colour(to_input(image, self.imgsz)), target


def collate(batch):
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)


# ---------------------------------------------------------------- training loop

def build_optimizer(model, lr, backbone_lr_scale, weight_decay):
    backbone, rest = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            (backbone if name.startswith("model.backbone") else rest).append(param)
    groups = [{"params": rest, "lr": lr}, {"params": backbone, "lr": lr * backbone_lr_scale}]
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def warmup_cosine(warmup_steps, total_steps, final_ratio=0.05):
    """Learning rate factor: linear warm-up, then a cosine down to final_ratio."""
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return final_ratio + (1 - final_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return factor


def train_dfine(cfg):
    """Train with AdamW and gradient accumulation, last.pt holds the moving average of the weights."""
    train = cfg["train"]
    out_dir = RUNS_DIR / cfg["family"] / cfg["name"]
    (out_dir / "weights").mkdir(parents=True, exist_ok=True)
    (out_dir / "experiment.json").write_text(json.dumps(cfg, indent=2))
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    dataset = FrameDataset(cfg["data"], cfg["imgsz"], train.get("min_crop", 0.5))
    loader = DataLoader(dataset, batch_size=train["batch"], shuffle=True, drop_last=True, collate_fn=collate,
                        num_workers=train.get("workers", 1), persistent_workers=train.get("workers", 1) > 0)

    model = load_pretrained(cfg["model"], **cfg.get("model_config", {})).cuda().train()
    average = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(train.get("ema", 0.999)), use_buffers=True)
    optimizer = build_optimizer(model, train["lr"], train.get("backbone_lr_scale", 0.1), train["weight_decay"])

    accumulate = train.get("accumulate", 1)
    total_steps = train["epochs"] * (len(loader) // accumulate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine(train["warmup_steps"], total_steps))

    log = []
    for epoch in range(train["epochs"]):
        start, losses = time.time(), []
        for i, (images, targets) in enumerate(loader):
            targets = [{key: value.cuda() for key, value in target.items()} for target in targets]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(pixel_values=images.cuda(), labels=targets).loss
            (loss / accumulate).backward()
            losses.append(loss.item())

            if (i + 1) % accumulate == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train.get("clip_norm", 0.1))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                average.update_parameters(model)

            if (i + 1) % 200 == 0:
                print(f"epoch {epoch + 1} batch {i + 1}/{len(loader)} loss {np.mean(losses[-200:]):.3f} "
                      f"memory {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GB", flush=True)

        log.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "minutes": (time.time() - start) / 60})
        print(log[-1], flush=True)
        (out_dir / "results.json").write_text(json.dumps(log, indent=2))
        torch.save(average.module.state_dict(), out_dir / "weights" / "last.pt")


# ---------------------------------------------------------------- inference

class DFineDetector:
    """A trained D-FINE model with the predict() call of an Ultralytics model."""

    def __init__(self, name, weights, **config):
        self.model = load_pretrained(name, **config)
        self.model.load_state_dict(torch.load(weights, map_location="cpu"))
        self.model.cuda().eval()

    @torch.inference_mode()
    def predict(self, image, imgsz, conf=0.001, max_det=500, **_):
        """Detect on one BGR frame. A query may give several classes, the best max_det pairs are kept."""
        height, width = image.shape[:2]
        start = time.perf_counter()
        pixels = to_input(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), imgsz).unsqueeze(0).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(pixel_values=pixels)

        probs = outputs.logits[0].float().sigmoid()
        scores, flat = probs.flatten().topk(min(max_det, probs.numel()))
        keep = scores >= conf
        scores, flat = scores[keep], flat[keep]
        queries, classes = flat // probs.shape[1], flat % probs.shape[1]

        centre, size = outputs.pred_boxes[0, queries].float().split(2, dim=1)
        relative = torch.cat([centre - size / 2, centre + size / 2], dim=1).clamp(0, 1)
        xyxy = relative * pixels.new_tensor([width, height] * 2)
        torch.cuda.synchronize()

        boxes = SimpleNamespace(xyxy=xyxy, conf=scores, cls=classes)
        return [SimpleNamespace(boxes=boxes, speed={"total": (time.perf_counter() - start) * 1000})]
