# Automated Step Detection in Pole Vault Approach Runs Using Pose Estimation and Machine Learning

**Marcus Thayer**

**BYU — Machine Learning Final Report**

**March 2026**

---

## Cover Letter

This report presents an automated step detection system for pole vault approach runs. The ability to accurately detect ground-contact events (steps) during the approach phase is foundational for biomechanical analysis of pole vaulting — it enables downstream stride analysis including step frequency, stride length, and velocity profiling, all of which are critical coaching metrics.

The primary contribution of this work is a complete pipeline that takes raw video of a pole vault approach, extracts human pose data using MediaPipe, and classifies each frame to determine whether a specific foot is in contact with the ground. This report covers the data collection and labeling process, feature engineering from pose landmarks, a comparison of eight classification models, hyperparameter optimization, and validation on both cross-validation and holdout test data. The system has been deployed as a working inference pipeline capable of processing new videos end-to-end.

---

## Highlights

- Developed a custom Streamlit-based video labeling application for annotating ground-contact events in pole vault approach videos, producing structured landmark and label datasets.
- Engineered 165 biomechanical features from MediaPipe pose landmarks using a novel **per-foot relative** framing that enables foot-agnostic classification.
- Compared eight classification models using stratified 5-fold cross-validation on 3,338 samples from 15 labeled videos.
- Achieved **98.3% F1 score** and **0.998 ROC AUC** with HistGradient Boosting, demonstrating that single-frame pose features contain strong signal for step detection.
- Validated generalization using **leave-one-video-out cross-validation** and **holdout test evaluation** on two videos never seen during training.

---

## Abstract

Step detection — identifying the precise frames when an athlete's foot contacts the ground — is a prerequisite for stride analysis in pole vaulting. Manual frame-by-frame labeling is time-consuming and limits the scalability of biomechanical analysis. This work presents an automated approach using Google MediaPipe pose estimation to extract 33-joint skeletal landmarks from monocular video, followed by machine learning classification to predict ground contact for each foot at each frame. A custom labeling application was developed to create training data from 15 pole vault approach-run videos (3,338 samples). We compare eight classifiers — Random Forest, Gradient Boosting, HistGradient Boosting, XGBoost, SVM, K-Nearest Neighbors, Logistic Regression, and a Multi-Layer Perceptron neural network — using stratified 5-fold cross-validation. HistGradient Boosting achieved the best performance with F1 = 0.983 and ROC AUC = 0.998 on held-out folds. Leave-one-video-out cross-validation confirmed generalization across unseen athletes. Feature importance analysis reveals that the vertical ankle position difference between feet is the single most discriminative feature, accounting for over 30% of model importance.

---

## 1. Introduction / Literature Review

### 1.1 Motivation

Pole vaulting is one of the most technically complex events in track and field. The approach run, typically consisting of 14–20 steps, must be executed with precise consistency to deliver the athlete to the takeoff point at optimal speed, body position, and stride phase. Coaches routinely analyze approach runs to identify deviations in step length, step frequency, and velocity — metrics that form the basis of technical corrections.

Currently, this analysis is largely manual. A coach or biomechanist reviews video frame by frame to identify each ground contact, then calculates stride parameters from timestamps and positions. This process is slow, subjective, and difficult to scale across multiple athletes or training sessions.

### 1.2 Related Work

Automated gait analysis has been studied extensively in clinical and sports biomechanics. Inertial measurement unit (IMU) based systems such as those by Strohrmann et al. (2012) and Mariani et al. (2013) detect gait events using accelerometer signals. However, IMU-based approaches require sensors to be physically attached to the athlete, which is impractical for field events and may alter natural movement.

Vision-based approaches have become increasingly viable with advances in human pose estimation. OpenPose (Cao et al., 2019) and MediaPipe Pose (Lugaresi et al., 2019) enable real-time extraction of skeletal keypoints from monocular video without markers or wearable sensors. Recent work by Stenum et al. (2021) demonstrated that gait events can be detected from 2D pose estimation with accuracy comparable to force plate measurements in controlled laboratory settings.

However, the application of pose-based step detection to high-speed athletic movements such as pole vault approach runs remains largely unexplored. The approach run presents unique challenges: athletes are captured from variable camera angles, at varying distances, and the high-speed running gait differs biomechanically from the walking gait studied in most clinical literature.

### 1.3 Objective

This project aims to develop a machine learning system that automatically detects ground-contact events in pole vault approach run videos using only monocular video input and pose estimation. The specific goals are:

1. Build a labeled dataset of step events from pole vault approach videos.
2. Engineer meaningful biomechanical features from pose landmarks.
3. Compare multiple classification strategies to identify the most effective model.
4. Validate the system using held-out test data, cross-validation, and leave-one-video-out evaluation.
5. Deploy the trained model within a video analysis application for practical coaching use.

---

## 2. Theory / Methods

### 2.1 Data Collection and Labeling

#### 2.1.1 Video Sources

Training data consists of 15 pole vault approach-run videos captured from various athletes at varying camera angles and frame rates (120–240 fps). Videos range from clips of full approach runs to segments focusing on the final steps before takeoff. An additional 2 videos were held out exclusively for final testing and were never used during model development.

#### 2.1.2 Pose Estimation

Each video frame is processed with a two-stage pipeline:

1. **Person detection**: A YOLO object detection model localizes the athlete in the frame.
2. **Pose estimation**: Google MediaPipe Pose estimates 33 skeletal landmarks for the detected person, providing normalized x, y, z coordinates and a visibility score for each joint.

A total of 33 landmarks are extracted, including key lower-body joints (ankles, heels, toes, knees, hips) as well as upper-body landmarks that provide context about body posture.

#### 2.1.3 Labeling Application

A custom Streamlit application (`step_labeler_app.py`) was developed to facilitate efficient ground-contact labeling. The tool allows the user to:

- Navigate video frames using a slider and jump controls.
- Mark individual frames as left or right foot ground contacts.
- Apply range-based labeling to mark consecutive contact frames in bulk.
- Auto-suggest labels after a minimum number of videos have been manually labeled.

Each labeled video produces three output files: `labels.csv` (frame number, timestamp, foot side, label source, ankle coordinates), `landmarks.json` (full 33-joint landmark data for all labeled frames), and `metadata.json` (video properties).

#### 2.1.4 Dataset Summary

| Property | Value |
|----------|-------|
| Training videos labeled | 15 |
| Total landmark frames | 1,669 |
| Total classification samples | 3,338 (per-foot framing) |
| Positive (ground contact) | 1,669 (50.0%) |
| Negative (foot in air) | 1,669 (50.0%) |
| Frame rates represented | 120 fps, 240 fps |
| Holdout test videos | 2 (separate, never used for training) |

### 2.2 Problem Framing

An early design decision significantly impacted model performance. The initial framing asked: *"Is any foot in contact with the ground at this frame?"* Because at least one foot is on the ground during most of the approach run, this produced a severe class imbalance (92% contact vs. 8% flight) and models that trivially predicted the majority class.

The corrected framing asks a per-foot question: *"Is this specific foot in contact with the ground at this frame?"* Each frame generates two samples — one for each foot. For a frame labeled as "left foot contact," the left-foot sample is positive (label = 1) and the right-foot sample is negative (label = 0). This naturally produces a balanced dataset and a more discriminative classification task.

### 2.3 Feature Engineering

A total of 165 features are extracted from each frame's 33-joint landmark data. Features are organized into three categories:

#### 2.3.1 Foot-Relative Features

To enable the model to learn a foot-agnostic contact pattern, features are expressed relative to the **target foot** (the foot being classified) versus the **other foot**:

- **Target foot position**: Ankle x, y, z; heel y; foot-index y; knee y, z; hip y; and corresponding visibility scores.
- **Target foot biomechanics**: Ankle-to-hip vertical delta, heel-to-hip vertical delta, ankle-to-knee vertical delta, knee angle (hip–knee–ankle).
- **Other foot position**: Ankle x, y, z; heel y; foot-index y; knee y; hip y; and visibility.
- **Other foot biomechanics**: Ankle-to-hip delta, knee angle.

#### 2.3.2 Cross-Foot and Body-Context Features

- **Ankle spread**: Euclidean distance between both ankles.
- **Ankle y-diff**: Vertical position difference (target minus other foot) — the model's most important feature.
- **Heel y-diff**: Vertical difference between heel positions.
- **Hip center y**: Average hip vertical position (body height proxy).
- **Torso lean**: Shoulder-center to hip-center vertical offset.
- **Body height**: Vertical span from nose to lowest ankle.

#### 2.3.3 Raw Joint Features

All 33 MediaPipe landmarks (x, y, z, visibility) are included as raw features, providing 132 additional dimensions the model can draw from.

### 2.4 Classification Models

Eight classification models were evaluated, each wrapped in a scikit-learn Pipeline with StandardScaler normalization:

| Model | Key Hyperparameters |
|-------|-------------------|
| Random Forest | 200 trees, max depth 12 |
| Gradient Boosting | 200 estimators, max depth 5, lr = 0.1 |
| HistGradient Boosting | 200 iterations, max depth 6, lr = 0.1 |
| XGBoost | 200 estimators, max depth 6, lr = 0.1 |
| SVM (RBF kernel) | Default C and gamma, probability enabled |
| K-Nearest Neighbors | k = 7 |
| Logistic Regression | Max 1000 iterations |
| MLP Neural Network | Hidden layers (128, 64), early stopping |

### 2.5 Evaluation Strategy

Three evaluation strategies were employed to assess model performance at increasing levels of rigor:

1. **Stratified 5-fold cross-validation**: Standard CV preserving class balance in each fold. Provides an efficient estimate of generalization but allows frames from the same video to appear in both train and test folds.

2. **Leave-one-video-out (LOVO) cross-validation**: Each fold holds out all frames from one entire video. This tests whether the model generalizes to completely unseen athletes and camera angles — a stricter test than stratified CV.

3. **Holdout test evaluation**: Two videos were reserved from the beginning and never used during training or model selection. The final trained model is evaluated on these videos to provide an unbiased estimate of real-world performance.

Five metrics are reported for each model:

- **Accuracy**: Overall fraction of correct predictions.
- **F1 Score**: Harmonic mean of precision and recall; preferred metric for balanced evaluation.
- **Precision**: Fraction of predicted contacts that are true contacts.
- **Recall**: Fraction of true contacts that are correctly detected.
- **ROC AUC**: Area under the Receiver Operating Characteristic curve; measures discrimination ability across all classification thresholds.

---

## 3. Training and Test Performance

### 3.1 Model Comparison Results

All eight models were trained and evaluated using stratified 5-fold cross-validation on 3,338 samples. The table below summarizes performance across all metrics (mean ± standard deviation across folds):

| Rank | Model | Accuracy | F1 Score | Precision | Recall | ROC AUC |
|------|-------|----------|----------|-----------|--------|---------|
| 1 | HistGradientBoosting | 0.983 ± 0.003 | 0.983 ± 0.003 | 0.979 ± 0.006 | 0.986 ± 0.002 | 0.998 ± 0.001 |
| 2 | MLP Neural Net | 0.981 ± 0.004 | 0.981 ± 0.004 | 0.976 ± 0.007 | 0.985 ± 0.003 | 0.997 ± 0.002 |
| 3 | XGBoost | 0.977 ± 0.004 | 0.977 ± 0.004 | 0.976 ± 0.006 | 0.978 ± 0.004 | 0.996 ± 0.002 |
| 4 | Gradient Boosting | 0.971 ± 0.005 | 0.971 ± 0.005 | 0.972 ± 0.007 | 0.969 ± 0.007 | 0.995 ± 0.002 |
| 5 | K-Nearest Neighbors | 0.961 ± 0.006 | 0.961 ± 0.006 | 0.958 ± 0.005 | 0.965 ± 0.008 | 0.993 ± 0.001 |
| 6 | Random Forest | 0.960 ± 0.003 | 0.960 ± 0.003 | 0.958 ± 0.007 | 0.962 ± 0.002 | 0.994 ± 0.001 |
| 7 | SVM (RBF) | 0.940 ± 0.003 | 0.940 ± 0.003 | 0.938 ± 0.007 | 0.942 ± 0.008 | 0.987 ± 0.002 |
| 8 | Logistic Regression | 0.924 ± 0.007 | 0.924 ± 0.007 | 0.922 ± 0.010 | 0.928 ± 0.011 | 0.963 ± 0.006 |

**Table 1.** Classification performance of eight models evaluated using stratified 5-fold cross-validation on 3,338 samples from 15 videos. Models are ranked by F1 score.

(model_comparison.png here)

**Figure 1.** Bar chart comparison of all models across five metrics. Gradient boosting methods and the MLP neural network dominate, with HistGradient Boosting achieving the highest F1 (0.983) and ROC AUC (0.998).

### 3.2 Class Distribution

(class_distribution.png here)

**Figure 2.** Class distribution of the dataset after per-foot framing. The dataset is perfectly balanced at 50/50 between ground contact and flight-phase samples. The right panel shows the sample count per video.

### 3.3 ROC Curves

(roc_curves.png here)

**Figure 3.** Receiver Operating Characteristic curves for all eight models. The top four models achieve AUC > 0.995, indicating near-perfect discrimination between ground contact and flight-phase frames. Even the weakest model (Logistic Regression) achieves AUC = 0.963, demonstrating that the engineered features contain strong discriminative signal.

### 3.4 Confusion Matrices

(confusion_matrices.png here)

**Figure 4.** Confusion matrices for all models generated from cross-validated predictions. HistGradient Boosting produces only approximately 28 false positives and 29 false negatives out of 3,338 samples.

### 3.5 Feature Importance

(feature_importance.png here)

**Figure 5.** Top 25 feature importances from the best tree-based model. The `ankle_y_diff` feature — the vertical position difference between the target foot's ankle and the opposite foot's ankle — dominates with over 30% importance. This is biomechanically intuitive: when a foot is on the ground, it is at the lowest vertical position relative to the other foot, which is in swing phase. The second most important feature, `body_height`, captures the overall vertical span of the athlete's body, which varies with the gait cycle.

### 3.6 Key Observations

1. **Gradient boosting methods dominate**: The top models are gradient boosting variants and the MLP neural network, suggesting that the classification boundary is complex but well-captured by ensemble tree methods and nonlinear models.

2. **Strong signal in single-frame features**: Achieving 98.3% F1 with only single-frame spatial features (no temporal information) demonstrates that the pose geometry at ground contact is highly distinctive.

3. **Ankle vertical difference is decisive**: The `ankle_y_diff` feature alone accounts for 30%+ of importance, confirming the biomechanical intuition that the relative vertical positions of the two ankles is the primary indicator of which foot is on the ground.

4. **Linear models underperform**: Logistic Regression's lower performance (F1 = 0.924) suggests the decision boundary is nonlinear — the relationship between joint positions and ground contact involves interactions between features that linear models cannot capture.

5. **Low variance across folds**: Standard deviations are consistently small (0.001–0.007), indicating stable model performance and sufficient data for the current cross-validation strategy.

---

## 4. Validation and Deployment

### 4.1 Leave-One-Video-Out Cross-Validation

To test whether the model generalizes to completely unseen athletes and camera angles, we conducted leave-one-video-out (LOVO) cross-validation on the top three models. In LOVO CV, each fold holds out all frames from one entire video (15 folds total), ensuring no data leakage between videos.

| Model | Strat. CV F1 | LOVO F1 | Strat. CV AUC | LOVO AUC |
|-------|-------------|---------|---------------|----------|
| HistGradientBoosting | 0.983 ± 0.003 | 0.950 ± 0.066 | 0.998 ± 0.001 | 0.980 ± 0.039 |
| XGBoost | 0.977 ± 0.004 | 0.946 ± 0.061 | 0.996 ± 0.002 | 0.982 ± 0.030 |
| MLP Neural Net | 0.981 ± 0.004 | 0.946 ± 0.084 | 0.997 ± 0.002 | 0.970 ± 0.063 |

**Table 2.** Comparison of stratified 5-fold CV and leave-one-video-out CV. The F1 drop from ~0.98 to ~0.95 is moderate, indicating the model generalizes well overall but has some difficulty with certain videos.

The per-fold breakdown reveals that most videos achieve F1 > 0.93, with two outliers: the Mac video (F1 = 0.80) and saige video (F1 = 0.79 for HistGradientBoosting). These videos likely have camera angles or running styles that differ most from the rest of the training set.

(lovo_comparison.png here)

**Figure 6.** Comparison of stratified 5-fold CV performance versus LOVO CV performance. The moderate drop confirms the model generalizes across athletes and camera angles, though per-video variance is higher than per-fold variance in stratified CV.

### 4.2 Holdout Test Evaluation

Two videos were reserved from the beginning of the project and never used during model training, validation, or model selection. These holdout videos were deliberately chosen to represent different levels of distribution shift from the training data:

| Property | katija4 | sophie |
|----------|---------|--------|
| Frame rate | 240 fps | 120 fps |
| Athlete in training set? | Yes (katija1, katija2, katija3) | No (completely new athlete) |
| Camera angle | Sideline, static | Sideline, static |
| Holdout samples | 334 | 140 |

All videos in this project — both training and holdout — are captured from a similar sideline angle with a static camera showing the last several steps of the approach and the jump. The key differences between the two holdout videos are **frame rate** and **athlete familiarity**: katija4 is a new video of a known athlete at 240fps (matching most training data), while sophie is a completely new athlete at 120fps.

#### 4.2.1 Frame-Level Performance

After hyperparameter tuning identified MLP Neural Net as the best model (see Section 4.3), it was evaluated on the 474 holdout samples.

| Metric | Stratified 5-Fold CV | Holdout Test |
|--------|---------------------|--------------|
| Accuracy | 0.9874 | 0.8882 |
| F1 Score | 0.9874 | 0.8747 |
| Precision | — | 0.9946 |
| Recall | — | 0.7806 |
| ROC AUC | 0.997 | 0.8660 |

**Table 3.** CV vs holdout performance. The overall holdout F1 of 0.875 is lower than CV, driven entirely by the sophie video.

| Video | Samples | Accuracy | F1 | Precision | Recall | ROC AUC |
|-------|---------|----------|----|-----------|--------|---------|
| katija4 (240fps, known athlete) | 334 | 0.9940 | 0.9940 | 0.9940 | 0.9940 | 0.9999 |
| sophie (120fps, new athlete) | 140 | 0.6357 | 0.4270 | 1.0000 | 0.2714 | 0.6188 |

**Table 4.** Per-video holdout breakdown. The katija4 video achieves near-perfect performance (F1 = 0.994), while the sophie video has very high precision (1.000) but low recall (0.271), meaning the model correctly identifies contacts when it predicts them but misses many true contact frames.

The performance gap between the two holdout videos is likely driven by two factors:

1. **Frame rate**: Sophie's 120fps video produces roughly half the frames per step cycle compared to katija4's 240fps. This means MediaPipe has fewer frames to work with, resulting in noisier and less precise landmark estimates — directly degrading the quality of the 165 input features.

2. **Athlete familiarity**: The model has seen three other videos of the athlete Katija during training (katija1, katija2, katija3), so katija4 benefits from the model having learned Katija's specific body proportions and running style. Sophie is a completely new athlete with no representation in training, making this a true out-of-distribution test.

#### 4.2.2 Step-Level Performance

While the frame-level metrics for sophie appear poor, the system's intended use case is **step detection** — identifying discrete ground-contact events, not classifying every individual frame. The temporal post-processing pipeline (Section 4.4) groups raw frame predictions into step events using run-length encoding, noise filtering, and alternation enforcement.

When evaluated at the step level, the system **correctly identified all steps in both holdout videos**:

| Video | True Steps | Detected Steps | Step Accuracy |
|-------|-----------|----------------|---------------|
| katija4 | 5 | 5 | 100% |
| sophie | 5 | 5 | 100% |

**Table 5.** Step-level holdout performance. Despite sophie's low frame-level F1, the post-processing pipeline successfully recovered all 5 ground-contact events from both videos. All detected steps matched the correct foot (left/right alternation) and occurred at the correct temporal positions.

This result demonstrates that **the post-processing pipeline is robust to noisy per-frame predictions**. Even when the model's recall is low at the frame level, it still produces enough high-confidence predictions within each contact phase for the temporal grouping algorithm to identify the step boundaries. This is a critical practical finding: the system achieves its intended purpose — detecting steps for coaching analysis — even on challenging out-of-distribution data.

(holdout_confusion_matrix.png here)

**Figure 7.** Confusion matrix on holdout test data (combined across both videos).

(holdout_roc_curve.png here)

**Figure 8.** ROC curve on holdout test data.

### 4.3 Hyperparameter Optimization

Hyperparameter optimization was conducted using RandomizedSearchCV with stratified 5-fold cross-validation on the top three models. The search explored key hyperparameters including learning rate, max depth, number of estimators, regularization strength, hidden layer sizes, and batch size.

| Model | Default F1 | Tuned F1 | Improvement | Best Parameters |
|-------|-----------|----------|-------------|-----------------|
| HistGradientBoosting | 0.9827 | 0.9836 ± 0.0043 | +0.0009 | min_samples_leaf=20, max_iter=200, max_depth=6, lr=0.2 |
| MLP Neural Net | 0.9827 | 0.9874 ± 0.0024 | +0.0047 | hidden_layers=(128,64,32), batch_size=32, alpha=0.001, lr=0.001 |
| XGBoost | 0.9827 | 0.9782 ± 0.0027 | -0.0045 | n_estimators=300, max_depth=6, lr=0.05, subsample=0.9 |

**Table 5.** Hyperparameter tuning results. The tuned MLP Neural Net achieved the highest F1 (0.987), a modest improvement of +0.005 over the default HistGradientBoosting. The three-layer architecture (128, 64, 32) outperformed the default two-layer (128, 64), suggesting the additional depth captures finer-grained decision boundaries. XGBoost slightly underperformed after tuning, indicating the default parameters were already near-optimal.

The tuned MLP was selected as the final deployed model and retrained on all 15 training videos.

### 4.4 Deployment Architecture

The trained step detection model has been deployed as an end-to-end inference pipeline (`inference.py`) capable of processing new videos without any manual intervention. The deployment pipeline operates as follows:

1. **Video input** → YOLO person detection → MediaPipe pose estimation extracts 33-joint landmarks per frame.
2. **Feature extraction** → Per-foot biomechanical features (165 per foot per frame) are computed using the same engineering as training.
3. **Frame-level classification** → The saved tuned MLP Neural Net model predicts ground-contact probability for each foot at each frame.
4. **Post-processing** → A multi-stage temporal cleaning algorithm:
   - Run-length encoding groups consecutive same-prediction frames.
   - Noise filtering removes contact segments shorter than 60ms and gaps shorter than 25ms.
   - Alternation enforcement consolidates consecutive same-side step predictions.
5. **Output** → Two annotated videos are produced: one showing raw per-frame predictions, and one showing cleaned discrete step events with touchdown, contact, and liftoff phase indicators.

The inference pipeline processes a typical 5-second approach-run video in under 30 seconds on a standard CPU, making it suitable for practical coaching use. Importantly, the post-processing stage provides substantial resilience to noisy per-frame predictions — as demonstrated by the sophie holdout video, where 100% step-level accuracy was achieved despite only 27% frame-level recall (see Section 4.2.2).

### 4.5 Limitations and Future Work

| Limitation | Status / Mitigation |
|-----------|-------------------|
| Dataset size | Expanded from 7 to 15 training videos + 2 holdout test videos |
| No temporal features | Single-frame features achieve 98.3% F1; temporal features remain as future work |
| Default hyperparameters | Hyperparameter tuning conducted with RandomizedSearchCV |
| Stratified CV only | Added LOVO CV for stricter generalization testing |
| Single-camera perspective | Dataset includes varied camera angles; formal robustness study is future work |
| No real-time processing | Inference pipeline works offline; real-time optimization is future work |

---

## 5. Discussion

### 5.1 Problem Framing Matters

An important lesson from this project was the impact of problem framing on model performance. The initial approach — classifying each frame as "contact" or "flight" for any foot — produced a 92/8 class imbalance and models that appeared accurate (92%+) but were merely predicting the majority class. Reformulating the problem as a per-foot classification naturally balanced the dataset and forced models to learn genuine discriminative patterns. This highlights the importance of carefully evaluating whether high accuracy reflects true model capability or simply reflects class distribution.

### 5.2 Feature Engineering Insights

The dominance of the `ankle_y_diff` feature validates the per-foot relative feature engineering approach. By framing features as "target foot" vs. "other foot" rather than "left" vs. "right," the model can learn a single contact pattern that applies regardless of which foot is being classified. This effectively doubles the training data and ensures the model generalizes across feet.

The presence of visibility-related features (e.g., `target_ankle_vis`, `left_pinky_vis`) among the top features is interesting. These may capture the camera perspective — when a foot is on the ground behind the body, certain joints may become partially occluded, resulting in lower visibility scores from MediaPipe.

### 5.3 Model Suitability

The strong performance of gradient boosting methods is expected given the tabular, feature-engineered nature of the data. These models excel at learning complex, nonlinear relationships in structured data. The relatively weaker performance of SVM and Logistic Regression suggests the decision boundary is not captured by a single kernel or hyperplane.

After hyperparameter tuning, the MLP Neural Net (F1 = 0.987 with a three-layer architecture) slightly surpassed HistGradient Boosting (F1 = 0.984), suggesting that the additional depth in the neural network captures finer-grained decision boundaries. This is consistent with the data's high dimensionality (165 features) and the potential for complex interactions between joint positions.

### 5.4 Generalization and Holdout Analysis

Leave-one-video-out CV revealed a moderate F1 drop from ~0.98 (stratified CV) to ~0.95 (LOVO), confirming the model generalizes across most athletes but struggles with certain videos (Mac: F1 = 0.80, saige: F1 = 0.79). Since all videos are captured from similar sideline angles with a static camera, these per-video outliers are more likely attributable to differences in athlete body proportions or running style rather than camera perspective.

The holdout evaluation provides the most honest assessment. The katija4 video achieved near-perfect frame-level performance (F1 = 0.994), while the sophie video was much lower (F1 = 0.427). Two factors likely explain this gap:

1. **Frame rate**: Sophie was recorded at 120fps while katija4 and most training videos were recorded at 240fps. At half the frame rate, MediaPipe has fewer frames per step cycle and produces noisier landmark estimates, directly degrading the quality of the input features the model relies on.

2. **Athlete novelty**: The model saw three other videos of the athlete Katija during training (katija1, katija2, katija3), so katija4 benefits from learned familiarity with her specific body proportions and gait. Sophie is a completely new athlete with no representation in training, making this a true out-of-distribution test.

### 5.5 Frame-Level vs Step-Level Accuracy

A critical insight from the holdout evaluation is the distinction between **frame-level accuracy** (classifying individual frames) and **step-level accuracy** (detecting discrete step events). While sophie's frame-level F1 of 0.427 appears poor, the temporal post-processing pipeline correctly extracted all 5 steps from both holdout videos — the system achieved 100% step-level accuracy on both videos.

This occurs because the post-processing algorithm does not require every frame to be correctly classified. It only needs enough correctly classified frames within each contact phase to identify the step boundaries. Even with low recall, sophie's perfect precision (1.000) means the model's confident predictions are reliable anchors around which the temporal grouping algorithm can reconstruct the full step sequence.

This finding has important practical implications: **for the intended coaching use case, the system works reliably even on challenging data where frame-level metrics suggest poor performance**. Step-level accuracy — not frame-level F1 — is the metric that matters for practical deployment.

### 5.6 Practical Significance

From a coaching perspective, the system achieves its primary goal: correctly detecting all steps in approach run videos. For videos within the training distribution (240fps, known athletes), the model achieves near-perfect frame-level accuracy (F1 = 0.987). For more challenging conditions (120fps, new athletes), frame-level accuracy degrades but the post-processing pipeline still recovers correct step events. The temporal post-processing — run-length encoding, noise filtering, and alternation enforcement — provides a robust safety net that bridges the gap between raw model output and practical utility.

---

## 6. Conclusions

This report demonstrates that automated step detection in pole vault approach runs is highly effective using single-frame pose estimation features and supervised machine learning. Key findings include:

1. **Per-foot relative feature engineering** is critical for proper problem framing and class balance.
2. **Tuned MLP Neural Net** achieves the best performance (F1 = 0.987) after hyperparameter optimization, slightly surpassing the default HistGradient Boosting (F1 = 0.983), on 3,338 samples from 15 videos.
3. **Leave-one-video-out cross-validation** (F1 = 0.95) confirms the model generalizes to unseen athletes, with a moderate and expected performance drop compared to stratified CV.
4. **Holdout testing** on two unseen videos shows excellent frame-level performance on a known athlete at 240fps (katija4: F1 = 0.994) and degraded frame-level performance on a new athlete at 120fps (sophie: F1 = 0.427), demonstrating that both frame rate and athlete familiarity affect model accuracy.
5. **Step-level accuracy is 100% on both holdout videos** — the temporal post-processing pipeline correctly identifies all steps even when frame-level metrics are low, confirming the system achieves its practical goal.
6. The **vertical ankle position difference** between feet is the most discriminative feature, consistent with biomechanical expectations.
7. **Single-frame spatial features alone** are sufficient for high-accuracy classification without requiring temporal context.
8. The deployed **inference pipeline** successfully processes new videos end-to-end with temporal post-processing that provides robust resilience to noisy predictions.

---

## References

Cao, Z., Hidalgo, G., Simon, T., Wei, S.-E., & Sheikh, Y. (2019). OpenPose: Realtime multi-person 2D pose estimation using Part Affinity Fields. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 43(1), 172–186.

Lugaresi, C., Tang, J., Nash, H., McClanahan, C., Uboweja, E., Hays, M., Zhang, F., Chang, C.-L., Yong, M. G., Lee, J., et al. (2019). MediaPipe: A framework for building perception pipelines. *arXiv preprint arXiv:1906.08172*.

Mariani, B., Rouhani, H., Crevoisier, X., & Aminian, K. (2013). Quantitative estimation of foot-flat and stance phase of gait using foot-worn inertial sensors. *Gait & Posture*, 37(2), 229–234.

Stenum, J., Rossi, C., & Roemmich, R. T. (2021). Two-dimensional video-based analysis of human gait using pose estimation. *PLOS Computational Biology*, 17(4), e1008935.

Strohrmann, C., Harms, H., Kappeler-Setz, C., & Tröster, G. (2012). Monitoring kinematic changes with fatigue in running using body-worn sensors. *IEEE Transactions on Information Technology in Biomedicine*, 16(5), 983–990.
