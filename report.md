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

Inline math works with single dollars: $f(x) = \sigma(Wx + b)$. Display math uses double dollars:

$$
\mathcal{L}(\theta) = -\frac{1}{N}\sum_{i=1}^{N} y_i \log \hat{y}_i
$$

Code blocks are highlighted:

```python
def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()
```

## Results

| Model    | Accuracy | F1   |
|----------|----------|------|
| Baseline | 0.71     | 0.68 |
| Ours     | 0.84     | 0.82 |

Images go into the `images/` folder:

<!-- ![Training loss](images/loss.png) -->

## Conclusion

> Summarize your findings here.
