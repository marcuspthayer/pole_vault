"""
Step Detection — Model Comparison Script
=========================================
Loads labeled landmark data from step_detection/data/*, engineers
biomechanical features, trains multiple classifiers with stratified
5-fold cross-validation, and saves visual comparison charts + a
summary report to step_detection/results/.

Run from the repo root or from step_detection/:
    python step_detection/train_and_compare.py
    python train_and_compare.py
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
    classification_report,
    accuracy_score, f1_score, precision_score, recall_score,
)

# Models
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR = SCRIPT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# MediaPipe landmark names (33 joints)
MP_LANDMARK_NAMES = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

# Joint indices for key body parts
IDX = {name: i for i, name in enumerate(MP_LANDMARK_NAMES)}


# ===========================================================================
# 1.  DATA LOADING
# ===========================================================================
def load_all_data() -> pd.DataFrame:
    """Load labels + landmarks from every video folder.

    Uses **per-foot** framing: for each frame in landmarks.json we
    produce two rows — one for the left foot and one for the right foot.
    The label is 1 only if that specific foot is tagged as contacting
    the ground at that frame.

    Features are expressed relative to the *target foot* so the model
    learns a foot-agnostic contact pattern.
    """

    all_rows = []

    for video_dir in sorted(DATA_DIR.iterdir()):
        if not video_dir.is_dir():
            continue
        labels_path = video_dir / "labels.csv"
        landmarks_path = video_dir / "landmarks.json"
        metadata_path = video_dir / "metadata.json"
        if not labels_path.exists() or not landmarks_path.exists():
            continue

        video_name = video_dir.name
        labels_df = pd.read_csv(labels_path)
        with open(landmarks_path, "r") as f:
            landmarks = json.load(f)
        with open(metadata_path, "r") as f:
            meta = json.load(f)

        # Build quick lookup: frame → set of sides with contact
        contact_lookup = {}
        for _, row in labels_df.iterrows():
            fr = int(row["frame"])
            contact_lookup.setdefault(fr, set()).add(row["side"])

        n_pos, n_neg = 0, 0

        # Iterate EVERY frame that has landmark data
        for frame_str, lm in landmarks.items():
            frame = int(frame_str)
            sides_contacting = contact_lookup.get(frame, set())

            # Produce one sample per foot
            for target_side in ("left", "right"):
                features = _landmarks_to_features_per_foot(lm, target_side)
                features["video"] = video_name
                features["frame"] = frame
                features["target_foot"] = target_side
                is_contact = 1 if target_side in sides_contacting else 0
                features["label"] = is_contact
                all_rows.append(features)
                if is_contact:
                    n_pos += 1
                else:
                    n_neg += 1

        print(f"  📂 {video_name:20s} — {n_pos:4d} contact, "
              f"{n_neg:4d} non-contact  (per-foot)")

    df = pd.DataFrame(all_rows)
    return df


def _landmarks_to_features_per_foot(lm: list, target_side: str) -> dict:
    """Build features relative to a *target foot*.

    Instead of left/right, features are named 'target_*' (the foot we
    are asking about) and 'other_*' (the opposite foot).  This lets the
    model learn a single contact pattern regardless of which foot.
    """
    features = {}

    def _pt(name):
        idx = IDX[name]
        return np.array([lm[idx]["x"], lm[idx]["y"], lm[idx]["z"]])

    # Determine which joints are target vs other
    if target_side == "left":
        t_ankle, o_ankle = "left_ankle", "right_ankle"
        t_heel, o_heel = "left_heel", "right_heel"
        t_foot_idx, o_foot_idx = "left_foot_index", "right_foot_index"
        t_knee, o_knee = "left_knee", "right_knee"
        t_hip, o_hip = "left_hip", "right_hip"
        t_shoulder, o_shoulder = "left_shoulder", "right_shoulder"
        t_elbow, o_elbow = "left_elbow", "right_elbow"
        t_wrist, o_wrist = "left_wrist", "right_wrist"
    else:
        t_ankle, o_ankle = "right_ankle", "left_ankle"
        t_heel, o_heel = "right_heel", "left_heel"
        t_foot_idx, o_foot_idx = "right_foot_index", "left_foot_index"
        t_knee, o_knee = "right_knee", "left_knee"
        t_hip, o_hip = "right_hip", "left_hip"
        t_shoulder, o_shoulder = "right_shoulder", "left_shoulder"
        t_elbow, o_elbow = "right_elbow", "left_elbow"
        t_wrist, o_wrist = "right_wrist", "left_wrist"

    # --- Target foot joints ---
    ta = _pt(t_ankle)
    th = _pt(t_heel)
    tfi = _pt(t_foot_idx)
    tk = _pt(t_knee)
    thip = _pt(t_hip)

    # --- Other foot joints ---
    oa = _pt(o_ankle)
    oh = _pt(o_heel)
    ofi = _pt(o_foot_idx)
    ok = _pt(o_knee)
    ohip = _pt(o_hip)

    # --- Body reference points ---
    nose = _pt("nose")
    t_sh = _pt(t_shoulder)
    o_sh = _pt(o_shoulder)
    hip_center = (thip + ohip) / 2
    shoulder_center = (t_sh + o_sh) / 2

    # =====  TARGET FOOT features  =====
    features["target_ankle_x"] = ta[0]
    features["target_ankle_y"] = ta[1]
    features["target_ankle_z"] = ta[2]
    features["target_ankle_vis"] = lm[IDX[t_ankle]]["visibility"]
    features["target_heel_y"] = th[1]
    features["target_heel_vis"] = lm[IDX[t_heel]]["visibility"]
    features["target_foot_index_y"] = tfi[1]
    features["target_foot_index_vis"] = lm[IDX[t_foot_idx]]["visibility"]
    features["target_knee_y"] = tk[1]
    features["target_knee_z"] = tk[2]
    features["target_hip_y"] = thip[1]

    # Target foot vertical deltas
    features["target_ankle_hip_dy"] = ta[1] - thip[1]
    features["target_heel_hip_dy"] = th[1] - thip[1]
    features["target_ankle_knee_dy"] = ta[1] - tk[1]

    # Target knee angle (hip–knee–ankle)
    features["target_knee_angle"] = _angle_3pts(thip, tk, ta)

    # =====  OTHER FOOT features  =====
    features["other_ankle_x"] = oa[0]
    features["other_ankle_y"] = oa[1]
    features["other_ankle_z"] = oa[2]
    features["other_ankle_vis"] = lm[IDX[o_ankle]]["visibility"]
    features["other_heel_y"] = oh[1]
    features["other_foot_index_y"] = ofi[1]
    features["other_knee_y"] = ok[1]
    features["other_hip_y"] = ohip[1]

    features["other_ankle_hip_dy"] = oa[1] - ohip[1]
    features["other_knee_angle"] = _angle_3pts(ohip, ok, oa)

    # =====  RELATIVE / CROSS-FOOT features  =====
    features["ankle_spread"] = np.linalg.norm(ta - oa)
    features["ankle_y_diff"] = ta[1] - oa[1]  # positive = target lower
    features["ankle_x_diff"] = ta[0] - oa[0]
    features["heel_y_diff"] = th[1] - oh[1]

    # =====  BODY CONTEXT features  =====
    features["hip_center_y"] = hip_center[1]
    features["torso_lean_y"] = shoulder_center[1] - hip_center[1]
    features["body_height"] = max(ta[1], oa[1]) - nose[1]
    features["target_ankle_to_nose_dy"] = ta[1] - nose[1]

    # =====  RAW positions for all 33 joints  =====
    for i, joint in enumerate(MP_LANDMARK_NAMES):
        features[f"{joint}_x"] = lm[i]["x"]
        features[f"{joint}_y"] = lm[i]["y"]
        features[f"{joint}_z"] = lm[i]["z"]
        features[f"{joint}_vis"] = lm[i]["visibility"]

    return features


def _landmarks_to_features(lm: list) -> dict:
    """Convert 33-joint landmark list to a feature dict (legacy, not used in training)."""
    features = {}

    # ---- Raw positions (x, y, z, visibility) for all 33 joints ----
    for i, joint in enumerate(MP_LANDMARK_NAMES):
        features[f"{joint}_x"] = lm[i]["x"]
        features[f"{joint}_y"] = lm[i]["y"]
        features[f"{joint}_z"] = lm[i]["z"]
        features[f"{joint}_vis"] = lm[i]["visibility"]

    # ---- Derived biomechanical features ----
    def _pt(name):
        idx = IDX[name]
        return np.array([lm[idx]["x"], lm[idx]["y"], lm[idx]["z"]])

    # Ankle positions
    la = _pt("left_ankle")
    ra = _pt("right_ankle")
    features["left_ankle_y_pos"] = la[1]
    features["right_ankle_y_pos"] = ra[1]
    features["min_ankle_y"] = min(la[1], ra[1])

    # Hip positions
    lh = _pt("left_hip")
    rh = _pt("right_hip")
    hip_center_y = (lh[1] + rh[1]) / 2
    features["hip_center_y"] = hip_center_y

    # Ankle-to-hip vertical deltas
    features["left_ankle_hip_dy"] = la[1] - lh[1]
    features["right_ankle_hip_dy"] = ra[1] - rh[1]

    # Knee angles (hip–knee–ankle)
    lk = _pt("left_knee")
    rk = _pt("right_knee")
    features["left_knee_angle"] = _angle_3pts(lh, lk, la)
    features["right_knee_angle"] = _angle_3pts(rh, rk, ra)

    # Heel + foot index positions
    features["left_heel_y"] = _pt("left_heel")[1]
    features["right_heel_y"] = _pt("right_heel")[1]
    features["left_foot_index_y"] = _pt("left_foot_index")[1]
    features["right_foot_index_y"] = _pt("right_foot_index")[1]

    # Foot spread (distance between ankles)
    features["ankle_spread"] = np.linalg.norm(la - ra)

    # Shoulder-hip alignment (torso lean proxy)
    ls = _pt("left_shoulder")
    rs = _pt("right_shoulder")
    shoulder_center = (ls + rs) / 2
    hip_center = (lh + rh) / 2
    features["torso_lean_y"] = shoulder_center[1] - hip_center[1]

    # Vertical span (nose to lowest ankle)
    nose = _pt("nose")
    features["body_height"] = max(la[1], ra[1]) - nose[1]

    return features


def _angle_3pts(a, b, c):
    """Angle at point b formed by segments a–b and c–b, in degrees."""
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))


# ===========================================================================
# 2.  MODEL DEFINITIONS
# ===========================================================================
def get_models() -> dict:
    """Return dict of named (pipeline) models."""
    models = {
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)),
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)),
        ]),
        "HistGradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", HistGradientBoostingClassifier(
                max_iter=200, max_depth=6, learning_rate=0.1, random_state=42)),
        ]),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=42)),
        ]),
        "K-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7)),
        ]),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "MLP Neural Net": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(128, 64), max_iter=500,
                early_stopping=True, random_state=42)),
        ]),
    }

    # Try XGBoost
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                use_label_encoder=False, eval_metric="logloss",
                random_state=42, n_jobs=-1, verbosity=0)),
        ])
    except ImportError:
        print("  ⚠️  XGBoost not installed, skipping.")

    return models


# ===========================================================================
# 3.  TRAINING & EVALUATION
# ===========================================================================
def train_and_evaluate(df: pd.DataFrame):
    """Train all models with stratified 5-fold CV, return results dict."""

    # Identify feature columns (drop metadata + target)
    drop_cols = ["video", "frame", "target_foot", "label"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(int)

    print(f"\n📊 Dataset: {len(X)} samples, {X.shape[1]} features, "
          f"{y.sum()} positive / {(1-y).sum()} negative")

    models = get_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}
    scoring = ["accuracy", "f1", "precision", "recall", "roc_auc"]

    for name, pipeline in models.items():
        print(f"  🏋️  Training {name}…", end="", flush=True)
        cv_results = cross_validate(
            pipeline, X, y, cv=cv, scoring=scoring,
            return_train_score=False, n_jobs=-1,
        )
        results[name] = {
            "accuracy": cv_results["test_accuracy"].mean(),
            "accuracy_std": cv_results["test_accuracy"].std(),
            "f1": cv_results["test_f1"].mean(),
            "f1_std": cv_results["test_f1"].std(),
            "precision": cv_results["test_precision"].mean(),
            "precision_std": cv_results["test_precision"].std(),
            "recall": cv_results["test_recall"].mean(),
            "recall_std": cv_results["test_recall"].std(),
            "roc_auc": cv_results["test_roc_auc"].mean(),
            "roc_auc_std": cv_results["test_roc_auc"].std(),
        }
        print(f"  acc={results[name]['accuracy']:.3f}  "
              f"f1={results[name]['f1']:.3f}  "
              f"auc={results[name]['roc_auc']:.3f}")

    return results, models, feature_cols, X, y


# ===========================================================================
# 3b. LEAVE-ONE-VIDEO-OUT CROSS-VALIDATION
# ===========================================================================
def train_and_evaluate_lovo(df: pd.DataFrame, model_names=None):
    """Train top models with Leave-One-Video-Out CV for stricter generalization.

    Each fold holds out ALL frames from one video, testing whether the model
    generalizes to completely unseen athletes and camera angles.
    """
    drop_cols = ["video", "frame", "target_foot", "label"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(int)
    groups = df["video"].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    all_models = get_models()
    # Default to top 3 if not specified
    if model_names is None:
        model_names = ["HistGradientBoosting", "XGBoost", "MLP Neural Net"]
    models = {n: all_models[n] for n in model_names if n in all_models}

    logo = LeaveOneGroupOut()
    n_splits = logo.get_n_splits(X, y, groups)
    unique_videos = np.unique(groups)
    print(f"\n📊 LOVO CV: {n_splits} folds (one per video), "
          f"{len(X)} samples, {X.shape[1]} features")

    scoring = ["accuracy", "f1", "precision", "recall", "roc_auc"]
    results = {}

    for name, pipeline in models.items():
        print(f"  🏋️  LOVO — {name}…", end="", flush=True)
        cv_results = cross_validate(
            pipeline, X, y, cv=logo, groups=groups,
            scoring=scoring, return_train_score=False, n_jobs=-1,
        )
        results[name] = {
            "accuracy": cv_results["test_accuracy"].mean(),
            "accuracy_std": cv_results["test_accuracy"].std(),
            "f1": cv_results["test_f1"].mean(),
            "f1_std": cv_results["test_f1"].std(),
            "precision": cv_results["test_precision"].mean(),
            "precision_std": cv_results["test_precision"].std(),
            "recall": cv_results["test_recall"].mean(),
            "recall_std": cv_results["test_recall"].std(),
            "roc_auc": cv_results["test_roc_auc"].mean(),
            "roc_auc_std": cv_results["test_roc_auc"].std(),
            "per_fold_f1": cv_results["test_f1"].tolist(),
        }
        print(f"  acc={results[name]['accuracy']:.3f}  "
              f"f1={results[name]['f1']:.3f}  "
              f"auc={results[name]['roc_auc']:.3f}")

    return results, unique_videos.tolist()


def write_lovo_report(lovo_results: dict, stratified_results: dict,
                      video_names: list):
    """Write markdown report comparing stratified CV vs LOVO CV."""
    lines = [
        "# Leave-One-Video-Out Cross-Validation Results",
        "",
        f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Videos**: {len(video_names)}",
        f"**Strategy**: Each fold holds out ALL frames from one video",
        "",
        "## Stratified 5-Fold CV vs LOVO CV",
        "",
        "| Model | Strat. CV F1 | LOVO F1 | Strat. CV AUC | LOVO AUC |",
        "|-------|-------------|---------|---------------|----------|",
    ]

    for name in lovo_results:
        s = stratified_results.get(name, {})
        l = lovo_results[name]
        lines.append(
            f"| {name} | "
            f"{s.get('f1', 0):.3f} ± {s.get('f1_std', 0):.3f} | "
            f"{l['f1']:.3f} ± {l['f1_std']:.3f} | "
            f"{s.get('roc_auc', 0):.3f} ± {s.get('roc_auc_std', 0):.3f} | "
            f"{l['roc_auc']:.3f} ± {l['roc_auc_std']:.3f} |"
        )

    lines += [
        "",
        "## Per-Fold F1 Scores (by held-out video)",
        "",
    ]

    # Build per-fold table header
    header = "| Model |"
    separator = "|-------|"
    for v in video_names:
        short = v[:12]
        header += f" {short} |"
        separator += "--------|"
    lines.append(header)
    lines.append(separator)

    for name, res in lovo_results.items():
        row = f"| {name} |"
        for f1_val in res["per_fold_f1"]:
            row += f" {f1_val:.3f} |"
        lines.append(row)

    lines += [
        "",
        "## Interpretation",
        "",
        "LOVO CV is a stricter test than stratified CV because it ensures "
        "no frames from the test video appear in training. A small drop in "
        "LOVO performance compared to stratified CV is normal and expected. "
        "A large drop would indicate the model is overfitting to specific "
        "athletes or camera angles.",
        "",
    ]

    report_path = RESULTS_DIR / "lovo_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  📄 Saved lovo_report.md")


def plot_lovo_comparison(lovo_results: dict, stratified_results: dict):
    """Bar chart comparing stratified CV vs LOVO CV F1 scores."""
    model_names = list(lovo_results.keys())
    strat_f1 = [stratified_results.get(n, {}).get("f1", 0) for n in model_names]
    lovo_f1 = [lovo_results[n]["f1"] for n in model_names]
    strat_std = [stratified_results.get(n, {}).get("f1_std", 0) for n in model_names]
    lovo_std = [lovo_results[n]["f1_std"] for n in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, strat_f1, width, yerr=strat_std,
                   label="Stratified 5-Fold CV", color="#6366f1",
                   capsize=4, edgecolor="white")
    bars2 = ax.bar(x + width/2, lovo_f1, width, yerr=lovo_std,
                   label="Leave-One-Video-Out", color="#f43f5e",
                   capsize=4, edgecolor="white")

    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("Stratified CV vs Leave-One-Video-Out CV",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0.85, 1.02)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "lovo_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  📊 Saved lovo_comparison.png")


# ===========================================================================
# 4.  VISUALISATIONS
# ===========================================================================
# ---- Color palette ----
PALETTE = [
    "#6366f1",  # indigo
    "#f43f5e",  # rose
    "#10b981",  # emerald
    "#f59e0b",  # amber
    "#3b82f6",  # blue
    "#8b5cf6",  # violet
    "#ec4899",  # pink
    "#14b8a6",  # teal
]

def _style_ax(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(labelsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")


def plot_model_comparison(results: dict):
    """Bar charts comparing accuracy, F1, precision, recall, AUC."""
    metrics = ["accuracy", "f1", "precision", "recall", "roc_auc"]
    labels = ["Accuracy", "F1 Score", "Precision", "Recall", "ROC AUC"]
    model_names = list(results.keys())

    fig, axes = plt.subplots(1, len(metrics), figsize=(22, 5.5))
    fig.suptitle("Model Comparison — Stratified 5-Fold CV",
                 fontsize=16, fontweight="bold", y=1.02)

    for ax, metric, label in zip(axes, metrics, labels):
        vals = [results[m][metric] for m in model_names]
        stds = [results[m][f"{metric}_std"] for m in model_names]
        bars = ax.barh(model_names, vals, xerr=stds,
                       color=PALETTE[:len(model_names)],
                       edgecolor="white", linewidth=0.5, capsize=3)
        ax.set_xlim(0, 1.08)
        _style_ax(ax, label)
        # Value annotations
        for bar, v in zip(bars, vals):
            ax.text(v + 0.01, bar.get_y() + bar.get_height()/2,
                    f"{v:.3f}", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  📊 Saved model_comparison.png")


def plot_roc_curves(models: dict, feature_cols, X, y):
    """Overlaid ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 7))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for idx, (name, pipeline) in enumerate(models.items()):
        tprs, aucs = [], []
        mean_fpr = np.linspace(0, 1, 100)

        for train_idx, test_idx in cv.split(X, y):
            pipeline.fit(X[train_idx], y[train_idx])
            if hasattr(pipeline, "predict_proba"):
                y_score = pipeline.predict_proba(X[test_idx])[:, 1]
            else:
                y_score = pipeline.decision_function(X[test_idx])
            fpr, tpr, _ = roc_curve(y[test_idx], y_score)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            aucs.append(auc(fpr, tpr))

        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = np.mean(aucs)
        ax.plot(mean_fpr, mean_tpr, color=PALETTE[idx % len(PALETTE)],
                linewidth=2, label=f"{name} (AUC={mean_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    _style_ax(ax, "ROC Curves — All Models", "False Positive Rate", "True Positive Rate")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    fig.savefig(RESULTS_DIR / "roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  📊 Saved roc_curves.png")


def plot_confusion_matrices(models: dict, X, y):
    """Grid of confusion matrices, one per model."""
    from sklearn.model_selection import cross_val_predict

    n_models = len(models)
    ncols = 4
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.2 * nrows))
    fig.suptitle("Confusion Matrices (5-Fold CV Predictions)",
                 fontsize=16, fontweight="bold", y=1.01)
    axes = axes.flatten() if n_models > 1 else [axes]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for idx, (name, pipeline) in enumerate(models.items()):
        y_pred = cross_val_predict(pipeline, X, y, cv=cv, n_jobs=-1)
        cm = confusion_matrix(y, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Flight", "Contact"])
        disp.plot(ax=axes[idx], cmap="Blues", colorbar=False)
        axes[idx].set_title(name, fontsize=11, fontweight="bold")

    # Hide unused axes
    for i in range(n_models, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  📊 Saved confusion_matrices.png")


def plot_feature_importance(models: dict, feature_cols, X, y):
    """Top-20 feature importances from the best tree-based model."""
    # Pick the tree-based model with highest accuracy from the ones trained
    tree_models = {}
    for name, pipe in models.items():
        clf = pipe.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            tree_models[name] = pipe

    if not tree_models:
        print("  ⚠️  No tree-based models found, skipping feature importance.")
        return

    # Train each on full data and pick the one with attribute
    best_name = None
    best_imp = None
    for name, pipe in tree_models.items():
        pipe.fit(X, y)
        imp = pipe.named_steps["clf"].feature_importances_
        if best_imp is None or imp.max() >= 0:
            best_name = name
            best_imp = imp

    top_k = 25
    top_idx = np.argsort(best_imp)[-top_k:]
    top_names = [feature_cols[i] for i in top_idx]
    top_vals = best_imp[top_idx]

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_idx)))
    ax.barh(top_names, top_vals, color=colors, edgecolor="white", linewidth=0.5)
    _style_ax(ax, f"Top {top_k} Features — {best_name}", "Importance", "")

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  📊 Saved feature_importance.png")


def plot_class_distribution(df: pd.DataFrame):
    """Simple pie + bar of class balance and per-video sample counts."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Class balance pie
    counts = df["label"].value_counts()
    labels_map = {1: "Ground Contact", 0: "Flight Phase"}
    ax1.pie([counts.get(1, 0), counts.get(0, 0)],
            labels=[labels_map[1], labels_map[0]],
            autopct="%1.1f%%", colors=["#6366f1", "#f43f5e"],
            startangle=90, textprops={"fontsize": 12})
    ax1.set_title("Class Distribution", fontsize=14, fontweight="bold")

    # Per-video counts
    video_counts = df.groupby(["video", "label"]).size().unstack(fill_value=0)
    video_counts.columns = [labels_map.get(c, c) for c in video_counts.columns]
    video_counts.plot(kind="barh", stacked=True, ax=ax2,
                      color=["#f43f5e", "#6366f1"], edgecolor="white")
    _style_ax(ax2, "Samples per Video", "Count", "")
    ax2.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  📊 Saved class_distribution.png")


# ===========================================================================
# 5.  SUMMARY REPORT
# ===========================================================================
def write_summary_report(results: dict, df: pd.DataFrame, feature_cols: list):
    """Write a markdown summary report."""
    # Sort by F1
    ranked = sorted(results.items(), key=lambda x: x[1]["f1"], reverse=True)

    lines = [
        "# Step Detection — Preliminary Model Comparison",
        "",
        f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Videos**: {df['video'].nunique()}",
        f"**Total samples**: {len(df)} "
        f"({(df['label']==1).sum()} contact, {(df['label']==0).sum()} flight)",
        f"**Features**: {len(feature_cols)}",
        f"**CV Strategy**: Stratified 5-Fold",
        "",
        "## Results (sorted by F1)",
        "",
        "| Rank | Model | Accuracy | F1 | Precision | Recall | ROC AUC |",
        "|------|-------|----------|----|-----------|--------|---------|",
    ]

    for rank, (name, m) in enumerate(ranked, 1):
        lines.append(
            f"| {rank} | {name} | "
            f"{m['accuracy']:.3f} ± {m['accuracy_std']:.3f} | "
            f"{m['f1']:.3f} ± {m['f1_std']:.3f} | "
            f"{m['precision']:.3f} ± {m['precision_std']:.3f} | "
            f"{m['recall']:.3f} ± {m['recall_std']:.3f} | "
            f"{m['roc_auc']:.3f} ± {m['roc_auc_std']:.3f} |"
        )

    best_name, best = ranked[0]
    worst_name, worst = ranked[-1]

    lines += [
        "",
        "## Key Findings",
        "",
        f"- 🥇 **Best model**: **{best_name}** with F1={best['f1']:.3f}, "
        f"AUC={best['roc_auc']:.3f}",
        f"- 🥉 **Worst model**: **{worst_name}** with F1={worst['f1']:.3f}, "
        f"AUC={worst['roc_auc']:.3f}",
        f"- Spread (best-worst F1): {best['f1'] - worst['f1']:.3f}",
        "",
        "## Generated Plots",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `model_comparison.png` | Accuracy, F1, precision, recall, AUC bar charts |",
        "| `roc_curves.png` | Overlaid ROC curves for all models |",
        "| `confusion_matrices.png` | Per-model confusion matrices |",
        "| `feature_importance.png` | Top-25 feature importances (best tree model) |",
        "| `class_distribution.png` | Class balance and per-video sample counts |",
        "",
        "## Next Steps",
        "",
        "- Label more videos to increase dataset size",
        "- Add temporal features (velocity / acceleration between consecutive frames)",
        "- Hyperparameter tuning on top-performing models",
        "- Evaluate leave-one-video-out CV once dataset is larger",
        "",
    ]

    report_path = RESULTS_DIR / "summary_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  📄 Saved summary_report.md")


# ===========================================================================
# 6.  SAVE BEST MODEL
# ===========================================================================
def save_best_model(results: dict, models: dict, feature_cols: list,
                    X: np.ndarray, y: np.ndarray):
    """Retrain the best model (by F1) on the full dataset and save it."""
    ranked = sorted(results.items(), key=lambda x: x[1]["f1"], reverse=True)
    best_name = ranked[0][0]
    best_metrics = ranked[0][1]

    print(f"  🏆 Best model: {best_name} (F1={best_metrics['f1']:.3f})")
    print(f"  🔄 Retraining on full dataset ({len(X)} samples)…")

    pipeline = models[best_name]
    pipeline.fit(X, y)

    # Save the trained pipeline and metadata
    model_path = MODELS_DIR / "best_step_model.joblib"
    meta_path = MODELS_DIR / "best_step_model_meta.json"

    joblib.dump(pipeline, model_path)

    meta = {
        "model_name": best_name,
        "f1": float(best_metrics["f1"]),
        "f1_std": float(best_metrics["f1_std"]),
        "accuracy": float(best_metrics["accuracy"]),
        "roc_auc": float(best_metrics["roc_auc"]),
        "n_features": len(feature_cols),
        "n_samples": int(len(X)),
        "feature_columns": feature_cols,
        "date_trained": pd.Timestamp.now().isoformat(),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  💾 Saved model to {model_path}")
    print(f"  💾 Saved metadata to {meta_path}")
    return best_name, model_path


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Step Detection — Model Comparison")
    parser.add_argument("--lovo", action="store_true",
                        help="Also run Leave-One-Video-Out CV on top models")
    parser.add_argument("--lovo-only", action="store_true",
                        help="Run ONLY the LOVO CV (skip standard training)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Step Detection — Model Comparison")
    print("=" * 60)

    # 1. Load data
    print("\n📂 Loading data…")
    df = load_all_data()
    if df.empty:
        print("❌ No data found. Make sure labeled videos exist in step_detection/data/")
        return
    print(f"\n✅ Loaded {len(df)} total samples from {df['video'].nunique()} videos")

    # Identify feature columns
    drop_cols = ["video", "frame", "target_foot", "label"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(int)

    # Handle any NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    results = {}

    if not args.lovo_only:
        # 2. Class distribution plot
        print("\n📊 Plotting class distribution…")
        plot_class_distribution(df)

        # 3. Train & evaluate
        print("\n🏋️  Training models…")
        results, models, _, _, _ = train_and_evaluate(df)

        # 4. Visualisations (re-using X, y after cleaning)
        print("\n🎨 Generating visualisations…")
        plot_model_comparison(results)
        plot_roc_curves(models, feature_cols, X, y)
        plot_confusion_matrices(models, X, y)
        plot_feature_importance(models, feature_cols, X, y)

        # 5. Summary report
        print("\n📝 Writing summary report…")
        write_summary_report(results, df, feature_cols)

        # 6. Save best model
        print("\n💾 Saving best model…")
        save_best_model(results, models, feature_cols, X, y)

    # 7. Leave-One-Video-Out CV
    if args.lovo or args.lovo_only:
        print("\n" + "=" * 60)
        print("  Leave-One-Video-Out Cross-Validation")
        print("=" * 60)
        lovo_results, video_names = train_and_evaluate_lovo(df)
        write_lovo_report(lovo_results, results, video_names)
        plot_lovo_comparison(lovo_results, results)

    print("\n" + "=" * 60)
    print(f"  ✅ Done! Results saved to {RESULTS_DIR}")
    if not args.lovo_only:
        print(f"  ✅ Best model saved to {MODELS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
