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

Since consecutive video frames are highly correlated, we experimented with temporal subsampling, selecting every 5th or 10th frame to reduce redundancy and computational cost. We compared these strategies under similar optimization budgets to evaluate whether additional frames provided useful information.

### Validation strategy

To avoid data leakage between correlated frames, we used **3-fold GroupKFold cross-validation**, grouping frames by their source video. All frames from the same video were assigned to the same fold.

For efficient model development, we evaluated experiments on fixed subsets of held-out training videos, sampling every 10th frame. We also maintained a separate evaluation subset from the provided validation split, sampling every 5th frame.

We used the same COCO-style evaluation procedure across architectures, reporting **mAP@[0.50:0.95]** as our primary metric, alongside AP50, AP75, and AP for small, medium, and large objects. We retained ignored regions during evaluation and allowed up to 500 detections per image, matching the competition evaluation settings.

We used paired per-clip AP differences to compare experimental configurations on identical video sequences, reducing the influence of variation between clips.

### Model selection and optimization

We compared several detector families to investigate different approaches to localization and classification:

- **Anchor-based:** Faster R-CNN and RetinaNet.
- **Anchor-free:** YOLO11.
- **Query-based:** RT-DETR and D-FINE.

We organized experiments around input resolution, architectural modifications, learning-rate schedules, optimizer settings, and inference strategies. We used validation results to guide subsequent experiments and compared Public Leaderboard scores where available.

After identifying promising detectors, we investigated tiling for small-object detection, temporal rescoring to exploit video continuity, and Weighted Boxes Fusion (WBF) to combine complementary model predictions.

## Results

### Baseline models

We first evaluated three detector families: two-stage anchor-based Faster R-CNN, one-stage anchor-based RetinaNet, and anchor-free YOLO11. We initialized the models with pretrained COCO weights and investigated how their architectures handled small objects and crowded scenes.

**Faster R-CNN.** We used ResNet-50-FPN and progressively increased the number of RPN proposals, introduced a cosine learning-rate schedule, and replaced RoI box regression with GIoU loss. These changes increased fold-0 AP from 0.2237 after the proposal modification to 0.2309 with all three changes. Across three folds, the selected configuration improved mean AP from 0.1890 to 0.1923. Its Public LB score was **0.17318**.

**RetinaNet.** We investigated anchor configurations and focal-loss parameters to address foreground-background imbalance. Our three-fold evaluation produced mean AP of 0.1748, with substantial variation between folds (0.2080, 0.1604, and 0.1558). The selected model achieved **0.18534** on the Public LB, increasing to **0.18803** with temporal postprocessing.

**YOLO11.** We tested input resolutions, an additional P2 detection head, NWD localization loss, and task-aligned assignment settings. Increasing resolution from 640 to 1152 pixels produced the clearest improvement (+0.0624 fold-0 AP). However, the P2 head reduced paired per-clip AP by approximately 0.016, while NWD and assignment modifications produced only small, inconclusive gains.

| Model | Evaluation result | Public LB |
|---|---|---|
| Faster R-CNN (selected) | 0.1923 mean 3-fold AP | 0.17318 |
| RetinaNet (selected) | 0.1748 mean 3-fold AP | 0.18534 |
| RetinaNet + temporal | — | 0.18803 |
| YOLO11s (af1) | 0.234 fold-0 AP | 0.19384 |

These experiments showed that higher input resolution was particularly important for small-object detection, while architectural complexity alone did not guarantee better performance. We therefore continued with query-based detectors and more advanced inference strategies.

### Query-based and architectural experiments

We next investigated RT-DETR-l, which uses object queries and Hungarian matching. At 1152 px, baseline RT-DETR (qb0) achieved 0.296 fold-0 AP and 0.253 official validation AP, outperforming YOLO11s (af1) by +0.054 ± 0.006 paired per-clip AP across 26 clips. This improvement came primarily from higher recall, although inference was approximately twice as slow.

We experimented with learning rates, warm-up, and query counts. Two configurations (qb1/qb2) performed poorly because an incorrect bias warm-up learning rate destabilized optimization. After correcting this issue and reducing the backbone learning rate, qb3 achieved 0.262 validation AP. Increasing queries from 300 to 500 (qb4) did not provide a consistent improvement (+0.003 ± 0.006 paired AP).

We subsequently evaluated D-FINE-L and D-FINE-X, which use fine-grained distribution refinement for bounding-box localization. D-FINE-L (qb5) improved paired AP over baseline RT-DETR by **+0.057 ± 0.012**, outperforming it on 24 of 26 clips. D-FINE-X (qb6) achieved higher validation AP, while additional crop augmentation (qb7) provided only a marginal improvement.

| Configuration | Fold-0 AP | Val AP |
|---|---:|---:|
| RT-DETR qb0 — baseline | 0.296 | 0.253 |
| RT-DETR qb1 — incorrect warm-up | 0.175 | 0.152 |
| RT-DETR qb2 — incorrect warm-up | 0.211 | 0.170 |
| RT-DETR qb3 — corrected optimization | 0.299 | 0.262 |
| RT-DETR qb4 — 500 queries | 0.304 | 0.237 |
| D-FINE-L qb5 | 0.331 | 0.293 |
| D-FINE-X qb6 | 0.341 | 0.317 |
| D-FINE-X qb7 — crop augmentation | **0.342** | **0.318** |

We also implemented a custom RT-DETR architecture with an additional stride-4 P2 feature level and four-scale decoder input. Neither initialization strategy (c1/c2) produced a reliable overall AP improvement, indicating that additional feature levels do not necessarily improve small-object detection.

Overall, D-FINE provided the strongest single-model validation results, while the unsuccessful warm-up, query-count, and P2 experiments highlighted the importance of optimization stability and effective pretrained representations.

### Inference and ensemble experiments

We investigated inference strategies to improve small-object detection without retraining. Increasing inference resolution to 1344–1536 px provided approximately +0.008–0.010 AP. Combining full-frame inference with overlapping 2×2 tiles and class-wise NMS achieved a larger **+0.021 ± 0.003 paired per-clip AP**, improving all 26 evaluated clips. However, tiling increased inference time approximately 4.5-fold.

We also introduced temporal rescoring, using detections from neighboring video frames to adjust confidence scores. Our best configuration used smooth rescoring, radius=2, and IoU=0.3, improving RT-DETR with tiling by **+0.006 ± 0.002 paired AP** on 23/26 clips. The mean strategy was less effective because it penalized objects missing from neighboring frames.

We tested Weighted Boxes Fusion (WBF) to combine predictions. Initially, RT-DETR + YOLO11 fusion provided no meaningful improvement (approximately 0.000 ± 0.003 paired AP). We subsequently explored larger RT-DETR ensembles and incorporated D-FINE models.

| Submission | Public LB |
|---|---:|
| RT-DETR full-frame | 0.27921 |
| RT-DETR + selective tiling + temporal | 0.28935 |
| D-FINE-X single model | 0.31507 |
| D-FINE-L single model | 0.33169 |
| 13 RT-DETR models + WBF + temporal | 0.35596 |
| 14-model ensemble (+ D-FINE-L) | 0.36281 |
| 15-model ensemble (2 D-FINE-L) | 0.36621 |
| 18-model ensemble (+ 3 D-FINE-X) | 0.36655 |
| Increased D-FINE-L weight | 0.36428 |
| Additional D-FINE-L checkpoint | 0.36347 |
| 18-model ensemble + selective tiling | **0.37148** |

Increasing D-FINE-L weights or adding another checkpoint reduced performance, demonstrating that larger ensembles are not necessarily better. Our strongest submission combined 13 RT-DETR runs, two D-FINE-L runs, and three lightly weighted D-FINE-X runs. Applying selective tiling to the D-FINE-L predictions increased Public LB from 0.36655 to **0.37148**.

Overall, combining complementary detectors, temporal consistency, and selective tiling provided greater improvements than architectural modifications alone. However, the selective tiling rule was partly chosen using Public LB feedback, making independent validation particularly important.

## Final model and reproducibility

### Final model

Our best submission, **ens22_l2tiles_x3light_temporal**, achieved **0.37148 Public LB**. It combines 13 RT-DETR runs, two D-FINE-L runs, and three D-FINE-X runs using Weighted Boxes Fusion (WBF), with lower weights assigned to D-FINE-X predictions.

We applied selective tiling to the two D-FINE-L runs for frames narrower than 1600 px, followed by WBF with IoU=0.65 and smooth temporal rescoring with radius=2 and IoU=0.3. This improved the Public LB from 0.36655 for the corresponding non-tiled ensemble to 0.37148.

### Reproducibility

We implemented our pipeline using PyTorch, TorchVision, Ultralytics, Transformers, and COCO-style evaluation. Dependencies are listed in `requirements.txt`, with a CUDA-enabled PyTorch installation required for GPU training.

We prepare the full training dataset and train the final models using:

```bash
pip install -r requirements.txt
python scripts/prepare_data.py --train-step 5 --full

python scripts/train.py configs/pipeline/f1_rtdetr_full_s5.yaml
python scripts/train.py configs/pipeline/f2_rtdetr_full_short.yaml
python scripts/train.py configs/pipeline/f3_dfine_l_full.yaml
python scripts/train.py configs/pipeline/f4_dfine_x_full.yaml
```

The RT-DETR and D-FINE configurations use 1152 px input resolution. The full-data runs use every fifth frame from the combined training and validation videos. The configurations specify model checkpoints, learning rates, batch sizes, warm-up settings, and training schedules.

For inference, `scripts/infer.py` generates predictions from individual checkpoints, while `scripts/ensemble.py` combines saved detections using WBF and temporal rescoring:

```bash
python scripts/ensemble.py <SOURCES> --weights <WEIGHTS> \
  --iou 0.65 --radius 2 --temporal-iou 0.3 \
  --test --name final
```

The source names and weights must match the selected `ens22` configuration, including its selectively tiled D-FINE-L predictions. The resulting submission is saved in `submissions/final.csv`.

Because the full-data models were trained using both the original training and validation clips, we report their performance using Public LB rather than treating those clips as independent validation data.

## Conclusion

In this homework, we systematically compared anchor-based, anchor-free, and query-based object detectors for crowded video scenes. Our experiments demonstrated that higher input resolution, stable optimization, and appropriate inference strategies were particularly important for detecting small objects.

RT-DETR and D-FINE substantially outperformed our initial Faster R-CNN, RetinaNet, and YOLO11 baselines. However, several architectural modifications, including additional P2 feature levels, increased query counts, and alternative localization losses, did not provide consistent improvements.

The largest gains came from combining complementary detectors through Weighted Boxes Fusion, exploiting temporal consistency, and applying selective tiling. Our final ensemble achieved **0.37148 Public LB**, outperforming the individual models and earlier ensembles.

The main limitations are increased inference cost, reliance on multiple trained checkpoints, and potential overfitting to Public Leaderboard feedback when selecting inference settings. Overall, our results highlight the importance of systematic experimentation and validation rather than increasing model complexity alone.
