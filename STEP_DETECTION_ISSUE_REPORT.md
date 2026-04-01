# ML Step Detection Issue: FPS Mismatch

## Summary
The ML step detection model is detecting only 3 steps for ~4 second 120fps videos (downsampled to 60fps) because **the model was trained on 120fps data but is now receiving 60fps data**. This causes the biomechanical features to be incorrect, even though the logs show "ML step detection found X steps".

## Root Cause Analysis

### 1. Training Data FPS
**The model was trained exclusively on 120fps data:**

From `/step_detection/data/*/metadata.json`:
- Most videos: ~119.94 fps (iPhone slow-mo 120fps)
- Some videos: ~239.48 fps (iPhone slow-mo 240fps)
- **NO videos at 60fps or 30fps in the training set**

Example metadata from `katija1/metadata.json`:
```json
{
  "fps": 119.94233541566554,
  "total_frames": 520,
  "processed_start": 126,
  "processed_end": 261,
  "label_count": 75
}
```

### 2. Labels at Different FPS Exist But Weren't Used

The training data directory has multiple label files per video:
```
katija1/labels.csv          (75 labels, original 120fps frame numbers)
katija1/labels_60fps.csv    (39 labels, downsampled)
katija1/labels_30fps.csv    (19 labels, downsampled)
```

However, the **training script uses only `labels.csv`** (line 103 in `train_and_compare.py`):
```python
labels_path = video_dir / "labels.csv"  # Always 120fps frame numbers
```

The `labels_60fps.csv` and `labels_30fps.csv` files exist but are **never used during training**.

### 3. Model Features Are Frame-Rate Dependent

The model uses temporal biomechanical features that **implicitly assume 120fps**:
- `target_ankle_hip_dy` (vertical deltas)
- `target_ankle_knee_dy` (vertical deltas)
- `target_knee_angle` (angles)
- `ankle_y_diff` (relative ankle heights)
- All velocity-related implicit features

These features represent spatial relationships at specific time intervals. When the frame rate changes from 120fps to 60fps without resampling the labels:
- The same frame numbers map to **different time instants**
- The feature values change while the labels stay the same
- The model makes predictions based on mismatched features

### 4. Current Inference Flow

In `/pvapp/pipelines/unified_pipeline.py` (lines 215-229):

```python
if enable_ml_steps:
    from step_detection.inference import load_model, predict_steps, clean_predictions
    
    ml_pipeline, ml_meta = load_step_model()
    feature_cols = ml_meta["feature_columns"]
    
    ml_predictions = predict_steps(
        pose_results,  # pose_results is at 120fps (from video)
        feature_cols, 
        ml_pipeline,
        start_frame=step_window_start,
        end_frame=step_window_end - 1,
    )
    ml_steps = clean_predictions(ml_predictions, fps=fps)  # fps from video (120 or 60)
```

**Problem:** The `fps` passed to `clean_predictions()` affects the thresholds:
- Line 186-187 in `inference.py`:
```python
min_step_frames = max(2, round(fps * min_step_seconds))      # 0.06s threshold
max_gap_frames = max(1, round(fps * max_gap_seconds))        # 0.025s threshold
```

For a 4-second video:
- **At 120fps:** 480 frames total
- **At 60fps:** 240 frames total
- **After downsampling to 60fps but using 120fps thresholds:** The thresholds are still calculated with the 120fps FPS value, which doubles frame counts

### 5. Why It Still Shows "ML Detection Found X Steps"

The logs show `"ML step detection found 3 steps"` (line 263 in `unified_pipeline.py`) because:
1. The model IS running (no exception thrown)
2. The model DOES output predictions
3. `clean_predictions()` groups them into steps

But these steps are **incorrect** due to the feature mismatch. The user sees visually wrong detections because the model was trained to recognize patterns in 120fps motion, not 60fps motion.

## Evidence

### Training Data Confirms 120fps-Only Training
```bash
$ grep -h "fps" step_detection/data/*/metadata.json | sort | uniq -c
      2   "fps": 119.88011988011988,
      1   "fps": 119.93942453306411,
      ...
      9   "fps": 119.94* (approximately)
      5   "fps": 239.48* (approximately, which is 2x 120fps)
      0   Videos at 60fps
      0   Videos at 30fps
```

### Labels Exist at Multiple FPS But Unused
```bash
$ ls step_detection/data/katija1/labels*.csv
labels.csv       (76 lines, original 120fps)
labels_60fps.csv (39 lines, downsampled 2x)
labels_30fps.csv (19 lines, downsampled 4x)
```

Training code uses only `labels.csv`:
```python
# Line 103 in train_and_compare.py
labels_path = video_dir / "labels.csv"  # Always loads the original
```

## Solutions

### Option 1: Retrain Model on 60fps Data (Recommended)
1. Update `train_and_compare.py` to use `labels_60fps.csv` when available
2. Resample landmarks from 120fps to 60fps (frame indices must match)
3. Retrain the model on consistent 60fps data
4. Test thoroughly with downsampled videos

### Option 2: Apply FPS-Aware Normalization During Inference
1. Detect the input video FPS
2. If FPS differs from training FPS (120fps), resample or normalize features
3. This is a post-hoc fix but risky without retraining

### Option 3: Always Use 120fps Videos for Inference
1. Upscale/interpolate 60fps videos back to 120fps before inference
2. Computationally expensive
3. May not recover lost temporal information

### Option 4: Train Multiple Models for Different FPS
1. Train separate models for 60fps, 120fps, 240fps
2. Select the appropriate model at inference time
3. More complex but most robust

## Metadata Files in Model

The model metadata (`best_step_model_meta.json`) contains:
```json
{
  "model_name": "MLP Neural Net",
  "f1": 0.9874,
  "accuracy": 0.9874,
  "n_features": 165,
  "n_samples": 3338,
  "feature_columns": [...],
  "date_trained": "2026-03-16T21:44:32.152471",
  "tuned": true,
  "best_params": {...}
}
```

**Missing:** No `trained_fps` or `training_frame_rate` field to warn users about the FPS requirement.

## Recommendation

**Add FPS metadata to the model** and validate at inference time:

1. Update `train_and_compare.py` to record the training FPS in metadata
2. Update `inference.py` to check FPS compatibility and warn/fail if mismatch is detected
3. Create training sets for different FPS rates, or standardize on a single FPS

This will prevent silent failures where the model runs but produces wrong results.
