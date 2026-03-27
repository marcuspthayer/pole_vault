# Hyperparameter Tuning Results

**Date**: 2026-03-16 21:44
**Strategy**: RandomizedSearchCV with Stratified 5-Fold CV
**Default best F1**: 0.9827

## Results

| Model | Default F1 | Tuned F1 | Improvement | Best Parameters |
|-------|-----------|----------|-------------|-----------------|
| HistGradientBoosting | 0.9827 | 0.9836 ± 0.0043 | +0.0009 | min_samples_leaf=20, max_iter=200, max_depth=6, learning_rate=0.2, l2_regularization=0.0 |
| MLP Neural Net | 0.9827 | 0.9874 ± 0.0024 | +0.0047 | learning_rate_init=0.001, hidden_layer_sizes=[128, 64, 32], batch_size=32, alpha=0.001 |
| XGBoost | 0.9827 | 0.9782 ± 0.0027 | -0.0045 | subsample=0.9, reg_lambda=2, reg_alpha=0, n_estimators=300, max_depth=6, learning_rate=0.05 |

## Best Configuration

- **Model**: MLP Neural Net
- **F1**: 0.9874 ± 0.0024
- **Parameters**:
  - `clf__learning_rate_init`: 0.001
  - `clf__hidden_layer_sizes`: [128, 64, 32]
  - `clf__batch_size`: 32
  - `clf__alpha`: 0.001
