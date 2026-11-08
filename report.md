---
title: Homework 2 -- Detect objects across videos
authors: Daryna Shevchuk and Roman Prokhorov
course: Artificial Intelligence · UCU
date: 2026-10-12
---

## Introduction

In this homework, we focused on multi-class object detection in crowded video scenes, aiming to develop an accurate detection pipeline and systematically compare different architectures, preprocessing strategies, and training configurations.

Our dataset contains **24,201 training frames from 56 videos** and **2,846 validation frames from 7 videos**, with eight target classes. After excluding ignored regions, we identified approximately 1.11 million training objects and 114 thousand validation objects.

During exploratory data analysis, we identified several challenges:

- **Small objects:** 24.91% of training bounding boxes have an area below 32² pixels, making detection sensitive to input resolution.
- **Class imbalance:** Cars and people account for 75.44% of training annotations.
- **Crowded scenes:** Some training frames contain up to 180 objects, increasing the difficulty of localization and duplicate suppression.
- **Distribution differences:** We observed variations in class proportions and object sizes between training and validation sequences.

Based on these findings, we explored anchor-based, anchor-free, and query-based detectors, along with higher input resolutions, architectural modifications, and alternative inference strategies.

## Method

### Data preparation

We converted the original COCO annotations into YOLO format for training Ultralytics-based detectors, while preserving COCO annotations for evaluation.

We excluded ignored (`iscrowd`) regions from training targets and masked them with gray pixels, restoring annotated objects that overlapped these regions. This prevented the models from learning from ambiguous areas while preserving valid training examples.

Since consecutive video frames are highly correlated, we experimented with temporal subsampling, selecting every 5th or 10th frame to reduce redundancy and computational cost.

### Validation strategy

To avoid data leakage between correlated frames, we used **3-fold GroupKFold cross-validation**, grouping frames by their source video. All frames from the same video were assigned to the same fold.

For efficient model development, we evaluated experiments on fixed subsets of held-out training videos, sampling every 10th frame. We also maintained a separate evaluation subset from the provided validation split, sampling every 5th frame.

We used the same COCO-style evaluation procedure across architectures, reporting **mAP@[0.50:0.95]** as our primary metric, alongside AP50, AP75, and AP for small, medium, and large objects. We retained ignored regions during evaluation and allowed up to 500 detections per image, matching the competition evaluation settings.

### Model selection and optimization

We compared several detector families to investigate different approaches to localization and classification:

- **Anchor-based:** Faster R-CNN and RetinaNet.
- **Anchor-free:** YOLO-based detectors.
- **Query-based:** RT-DETR and D-FINE.

We organized experiments around input resolution, architectural modifications, optimization settings, and inference strategies. We used validation results to guide subsequent experiments and compared public leaderboard scores where available.

## Results

### Baseline Models

We first evaluated three detector families: two-stage anchor-based Faster R-CNN, one-stage anchor-based RetinaNet, and anchor-free YOLO11. We initialized the models with pretrained COCO weights and investigated how their architectures handled small objects and crowded scenes.

**Faster R-CNN.** We used ResNet-50-FPN and progressively increased the number of RPN proposals, introduced a cosine learning-rate schedule, and replaced RoI box regression with GIoU loss. These changes increased fold-0 AP from 0.2237 after the proposal modification to 0.2309 with all three changes. Across three folds, the selected configuration improved mean AP from 0.1890 to 0.1923. Its Public LB score was **0.17318**.

**RetinaNet.** We investigated anchor configurations and focal-loss parameters to address foreground-background imbalance. Our three-fold evaluation produced mean AP of 0.1748, with substantial variation between folds (0.2080, 0.1604, and 0.1558). The selected model achieved **0.18534** on the Public LB, increasing to **0.18803** with temporal postprocessing.

**YOLO11.** We tested input resolutions, an additional P2 detection head, NWD localization loss, and task-aligned assignment settings. Increasing resolution from 640 to 1152 pixels produced the clearest improvement (+0.0624 fold-0 AP). However, the P2 head reduced paired per-clip AP by approximately 0.016, while NWD and assignment modifications produced only small, inconclusive gains.

| Model | Evaluation result | Public LB |
|---|---|---|
| Faster R-CNN (selected) | 0.1923 mean 3-fold AP | 0.17318 |
| RetinaNet (selected) | 0.1748 mean 3-fold AP | 0.18534 |
| YOLO11s (af1) | 0.234 fold-0 AP | 0.19384 |

These experiments showed that higher input resolution was particularly important for small-object detection, while architectural complexity alone did not guarantee better performance. We therefore continued with query-based detectors and more advanced inference strategies.

### Query-Based and Architectural Experiments

We next investigated RT-DETR-l, using a hybrid encoder and object queries with Hungarian matching. At 1152 px, the baseline RT-DETR (qb0) achieved 0.296 fold-0 AP and 0.253 official validation AP, outperforming YOLO11s (af1) by approximately +0.054 ± 0.006 paired per-clip AP across 26 clips. RT-DETR improved recall from 62% to 77%, although inference was approximately twice as slow.

We then experimented with learning rates, warm-up, and query counts. Two initial configurations (qb1/qb2) performed poorly because an incorrect bias warm-up learning rate destabilized optimization. After correcting this issue and reducing backbone learning rate, qb3 achieved our strongest RT-DETR validation result of **0.262 AP**, compared with 0.253 for qb0. Increasing the number of queries from 300 to 500 (qb4) did not provide a consistent improvement (+0.003 ± 0.006 paired AP).

| RT-DETR configuration | Fold-0 AP | Official Val AP |
|---|---:|---:|
| qb0 — baseline | 0.296 | 0.253 |
| qb1 — incorrect warm-up | 0.175 | 0.152 |
| qb2 — incorrect warm-up | 0.211 | 0.170 |
| qb3 — corrected optimization | 0.299 | **0.262** |
| qb4 — 500 queries | **0.304** | 0.237 |

We also implemented a custom RT-DETR architecture with an additional stride-4 P2 feature level and four-scale decoder input. Despite its potential benefits for small objects, neither initialization strategy (c1/c2) produced a reliable overall AP improvement. Similarly, YOLO's P2 modification reduced paired AP by approximately 0.016.

These experiments demonstrated that optimization stability and effective use of pretrained features mattered more than simply increasing architectural complexity. We retained the standard RT-DETR architecture for subsequent pipeline experiments.

## Conclusion

> Summarize your findings here.
