# Step Detection — Preliminary Model Comparison

**Date**: 2026-03-16 19:07
**Videos**: 15
**Total samples**: 3338 (1669 contact, 1669 flight)
**Features**: 165
**CV Strategy**: Stratified 5-Fold

## Results (sorted by F1)

| Rank | Model | Accuracy | F1 | Precision | Recall | ROC AUC |
|------|-------|----------|----|-----------|--------|---------|
| 1 | HistGradientBoosting | 0.983 ± 0.003 | 0.983 ± 0.003 | 0.979 ± 0.006 | 0.986 ± 0.002 | 0.998 ± 0.001 |
| 2 | MLP Neural Net | 0.981 ± 0.004 | 0.981 ± 0.004 | 0.976 ± 0.007 | 0.985 ± 0.003 | 0.997 ± 0.002 |
| 3 | XGBoost | 0.977 ± 0.004 | 0.977 ± 0.004 | 0.976 ± 0.006 | 0.978 ± 0.004 | 0.996 ± 0.002 |
| 4 | Gradient Boosting | 0.971 ± 0.005 | 0.971 ± 0.005 | 0.972 ± 0.007 | 0.969 ± 0.007 | 0.995 ± 0.002 |
| 5 | K-Nearest Neighbors | 0.961 ± 0.006 | 0.961 ± 0.006 | 0.958 ± 0.005 | 0.965 ± 0.008 | 0.993 ± 0.001 |
| 6 | Random Forest | 0.960 ± 0.003 | 0.960 ± 0.003 | 0.958 ± 0.007 | 0.962 ± 0.002 | 0.994 ± 0.001 |
| 7 | SVM (RBF) | 0.940 ± 0.003 | 0.940 ± 0.003 | 0.938 ± 0.007 | 0.942 ± 0.008 | 0.987 ± 0.002 |
| 8 | Logistic Regression | 0.924 ± 0.007 | 0.924 ± 0.007 | 0.922 ± 0.010 | 0.928 ± 0.011 | 0.963 ± 0.006 |

## Key Findings

- 🥇 **Best model**: **HistGradientBoosting** with F1=0.983, AUC=0.998
- 🥉 **Worst model**: **Logistic Regression** with F1=0.924, AUC=0.963
- Spread (best-worst F1): 0.058

## Generated Plots

| File | Description |
|------|-------------|
| `model_comparison.png` | Accuracy, F1, precision, recall, AUC bar charts |
| `roc_curves.png` | Overlaid ROC curves for all models |
| `confusion_matrices.png` | Per-model confusion matrices |
| `feature_importance.png` | Top-25 feature importances (best tree model) |
| `class_distribution.png` | Class balance and per-video sample counts |

## Next Steps

- Label more videos to increase dataset size
- Add temporal features (velocity / acceleration between consecutive frames)
- Hyperparameter tuning on top-performing models
- Evaluate leave-one-video-out CV once dataset is larger
