# Presentation Outline: Automated Step Detection in Pole Vault

## Slide 1: Title
**Automated Step Detection in Pole Vault Approach Runs Using Pose Estimation and Machine Learning**
- Marcus Thayer
- BYU — Machine Learning Course Project
- March 2026

---

## Slide 2: Problem & Motivation
**Why Step Detection Matters**
- Pole vault approach run: 14–20 steps that must be executed with precise consistency
- Coaches need stride metrics: step frequency, stride length, velocity
- Current process: manual frame-by-frame video review — slow, subjective, unscalable
- **Goal**: Automate step detection from standard video using only pose estimation + ML

**Speaker notes**: Emphasize that this is a real coaching need. Show a short video clip of an approach run if possible.

---

## Slide 3: Data Pipeline Overview
**From Video to Predictions**

```
Video → YOLO Person Detection → MediaPipe Pose (33 joints) → Feature Engineering (165 features) → ML Classification → Post-Processing → Step Events
```

- YOLO localizes the athlete in each frame
- MediaPipe extracts 33 skeletal landmarks (x, y, z, visibility)
- 165 biomechanical features engineered per foot per frame
- Binary classification: is this foot on the ground?

**Speaker notes**: Walk through the pipeline left to right. Mention this is all from monocular video — no sensors needed.

---

## Slide 4: Problem Framing — The Key Insight
**Per-Foot Classification**

| Approach | Class Balance | Best F1 |
|----------|--------------|---------|
| "Is ANY foot on ground?" | 92% / 8% (imbalanced) | ~0.92 (trivial) |
| "Is THIS foot on ground?" | 50% / 50% (balanced) | 0.987 |

- Each frame → 2 samples (one per foot)
- Features are "target foot" vs "other foot" — foot-agnostic
- Doubles effective training data

**Speaker notes**: This was the biggest lesson learned. The first framing looked great (92% accuracy) but was just predicting majority class.

---

## Slide 5: Feature Engineering
**165 Features in Three Categories**

1. **Foot-relative features** (~25): ankle position, knee angle, hip deltas — expressed relative to target vs. other foot
2. **Cross-foot features** (~10): ankle spread, ankle_y_diff, heel_y_diff, body height
3. **Raw joint positions** (132): all 33 MediaPipe landmarks × 4 (x, y, z, visibility)

**Key feature**: `ankle_y_diff` = vertical position of target foot minus opposite foot
- Accounts for 30%+ of model importance
- Biomechanically intuitive: ground foot is lower than swing foot

*Show: feature_importance.png*

**Speaker notes**: The feature engineering was designed so the model learns one contact pattern that works for either foot.

---

## Slide 6: Dataset
**15 Training Videos, 3,338 Samples**

- Multiple athletes, camera angles, frame rates (120–240 fps)
- Custom Streamlit labeling app built for efficient annotation
- 2 additional holdout videos reserved for final testing

*Show: class_distribution.png*

**Speaker notes**: Mention the labeling app was a significant engineering effort — show a screenshot if time permits.

---

## Slide 7: Model Comparison & Tuning
**8 Classifiers Compared → Hyperparameter Tuning on Top 3**

*Show: model_comparison.png*

| Top 3 (default) | F1 | After Tuning | F1 |
|-------------|----------|-------------|---------|
| HistGradientBoosting | 0.983 | HistGradientBoosting | 0.984 |
| MLP Neural Net | 0.981 | **MLP Neural Net** | **0.987** |
| XGBoost | 0.977 | XGBoost | 0.978 |

- Gradient boosting and neural net methods dominate
- Tuned MLP (3-layer: 128→64→32) becomes best model
- Linear models (LR, SVM) underperform → nonlinear decision boundary

*Show: roc_curves.png*

**Speaker notes**: Tuning yielded modest improvement (+0.005 F1). The bigger story is that 8 different model families all perform well, confirming the feature engineering captures strong signal.

---

## Slide 8: Validation — LOVO Cross-Validation
**Does the Model Generalize to Unseen Athletes?**

- Leave-One-Video-Out CV: hold out ALL frames from one video per fold
- Tests generalization across athletes and camera angles
- Stricter than stratified CV (no data leakage between videos)

*Show: lovo_comparison.png*

| Model | Strat. CV F1 | LOVO F1 |
|-------|-------------|---------|
| HistGradientBoosting | 0.983 | 0.950 |
| XGBoost | 0.977 | 0.946 |
| MLP Neural Net | 0.981 | 0.946 |

- F1 drops ~3% from stratified CV to LOVO — moderate, expected
- Most videos: F1 > 0.93; outliers: Mac (0.80), saige (0.79)
- Confirms model generalizes, but some videos are harder

**Speaker notes**: The 3% drop is a healthy sign — it means the model learned genuine patterns, not video-specific artifacts. All videos are from similar sideline angles, so the outliers (Mac, saige) are likely due to differences in athlete body proportions or running style.

---

## Slide 9: Holdout Test Results
**True Unseen Test Data — Two Levels of Difficulty**

| Property | katija4 | sophie |
|----------|---------|--------|
| Frame rate | 240 fps | 120 fps |
| Athlete in training? | Yes (katija1/2/3) | No (new athlete) |
| Camera angle | Sideline, static | Sideline, static |

**Frame-level results:**

| Video | F1 | Precision | Recall |
|-------|-----|-----------|--------|
| katija4 | **0.994** | 0.994 | 0.994 |
| sophie | 0.427 | 1.000 | 0.271 |

**But here's the key finding — step-level results:**

| Video | True Steps | Detected Steps | Step Accuracy |
|-------|-----------|----------------|---------------|
| katija4 | 5 | **5** | **100%** |
| sophie | 5 | **5** | **100%** |

- sophie's low F1 is driven by **120fps** (noisier MediaPipe landmarks) and **new athlete** (no training data)
- But the post-processing pipeline **still correctly detects all 5 steps** from both videos
- Frame-level F1 ≠ practical usefulness — step detection works even on hard data

**Speaker notes**: This is the most interesting slide. Start with the scary sophie number (0.427 F1), then reveal that it still works perfectly at the step level. This shows you understand the difference between metric performance and real-world utility. Explain that 120fps gives MediaPipe half the data, and sophie is a totally new athlete. The post-processing pipeline (temporal smoothing, noise filtering) recovers correct steps from noisy predictions.

---

## Slide 10: Demo — Video Comparison
**Katija4 (Easy Case) vs Sophie (Hard Case)**

Show 4 videos for each athlete (8 total, or play side-by-side):

**For each video show:**
1. **Original** — unprocessed approach run video
2. **Raw predictions** — per-frame model output (noisy for sophie, clean for katija4)
3. **Cleaned steps** — post-processed discrete step events (correct for both!)
4. **Full analysis** — complete pole vault analysis overlay

Color coding:
- Green dots = left foot contact
- Orange dots = right foot contact
- Yellow flash = touchdown moment

**Key point**: Compare the raw predictions between katija4 (very clean) and sophie (noisy) — then show that the cleaned output is equally correct for both.

**Speaker notes**: This is the most impactful part of the presentation. Play the videos or show side-by-side screenshots. Let the audience see that even with noisy raw predictions (sophie), the final output correctly identifies every step. This demonstrates the value of the post-processing pipeline as a critical system component.

---

## Slide 11: Conclusions
**Key Takeaways**

1. **Per-foot problem framing** is essential — transforms an imbalanced problem into a balanced, learnable one
2. **Tuned MLP** achieves 98.7% F1 with single-frame pose features — no temporal modeling needed
3. **ankle_y_diff** dominates feature importance — biomechanically intuitive
4. **LOVO CV** (F1 = 0.95) confirms generalization across athletes
5. **Step-level accuracy is 100%** on both holdout videos — even the hard case (120fps, new athlete) works perfectly after post-processing
6. **Frame-level metrics don't tell the whole story** — the post-processing pipeline bridges the gap between noisy predictions and practical utility

**Future Work**:
- Temporal features (velocity/acceleration) for potential further gains
- Real-time processing optimization
- Integration into full pole vault analysis web application

---

## Slide 12: Q&A
**Questions?**

- GitHub: github.com/alphapeakio/polevault
- Contact: Marcus Thayer
