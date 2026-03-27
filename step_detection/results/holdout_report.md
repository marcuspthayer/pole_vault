# Holdout Test Evaluation

**Date**: 2026-03-16 21:45
**Holdout videos**: 2
**Holdout samples**: 474
**Model**: MLP Neural Net

## CV vs Holdout Comparison

| Metric | Stratified 5-Fold CV | Holdout Test |
|--------|---------------------|--------------|
| Accuracy | 0.9874 | 0.8882 |
| F1 Score | 0.9874 | 0.8747 |
| Precision | — | 0.9946 |
| Recall | — | 0.7806 |
| ROC AUC | 0.0000 | 0.8660 |

## Per-Video Breakdown

| Video | Samples | Accuracy | F1 | Precision | Recall | ROC AUC |
|-------|---------|----------|----|-----------|--------|---------|
| katija4 | 334 | 0.9940 | 0.9940 | 0.9940 | 0.9940 | 0.9999 |
| sophie | 140 | 0.6357 | 0.4270 | 1.0000 | 0.2714 | 0.6188 |

## Generated Plots

| File | Description |
|------|-------------|
| `holdout_confusion_matrix.png` | Confusion matrix on holdout data |
| `holdout_roc_curve.png` | ROC curve on holdout data |
