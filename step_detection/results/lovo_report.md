# Leave-One-Video-Out Cross-Validation Results

**Date**: 2026-03-16 21:41
**Videos**: 15
**Strategy**: Each fold holds out ALL frames from one video

## Stratified 5-Fold CV vs LOVO CV

| Model | Strat. CV F1 | LOVO F1 | Strat. CV AUC | LOVO AUC |
|-------|-------------|---------|---------------|----------|
| HistGradientBoosting | 0.983 ± 0.003 | 0.950 ± 0.066 | 0.998 ± 0.001 | 0.980 ± 0.039 |
| XGBoost | 0.977 ± 0.004 | 0.946 ± 0.061 | 0.996 ± 0.002 | 0.982 ± 0.030 |
| MLP Neural Net | 0.981 ± 0.004 | 0.946 ± 0.084 | 0.997 ± 0.002 | 0.970 ± 0.063 |

## Per-Fold F1 Scores (by held-out video)

| Model | 4.60 Pole #1 | 4.75 Pole #5 | Avery1 | Avery2 | Mac | dane | ean | kate2 | katija1 | katija2 | katija3 | red_hair | saige | tyce_bad | tyce_good |
|-------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| HistGradientBoosting | 0.973 | 0.942 | 0.989 | 1.000 | 0.797 | 0.997 | 0.938 | 0.993 | 1.000 | 0.994 | 0.979 | 0.944 | 0.786 | 0.940 | 0.976 |
| XGBoost | 0.932 | 0.929 | 0.989 | 1.000 | 0.791 | 0.990 | 0.933 | 0.987 | 1.000 | 0.988 | 0.979 | 0.950 | 0.817 | 0.947 | 0.961 |
| MLP Neural Net | 0.951 | 0.919 | 0.989 | 1.000 | 0.740 | 0.997 | 0.980 | 0.993 | 0.993 | 1.000 | 0.983 | 0.926 | 0.741 | 0.983 | 0.988 |

## Interpretation

LOVO CV is a stricter test than stratified CV because it ensures no frames from the test video appear in training. A small drop in LOVO performance compared to stratified CV is normal and expected. A large drop would indicate the model is overfitting to specific athletes or camera angles.

The F1 drop from stratified CV (~0.98) to LOVO (~0.95) is moderate, indicating the model generalizes well overall but has some difficulty with certain videos. The weakest per-fold performances (Mac: 0.74–0.80, saige: 0.74–0.82) suggest these videos have characteristics (camera angle, athlete body type, or running style) that differ from the rest of the training set.
