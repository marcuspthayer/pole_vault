"""
Step Detection — Holdout Test Evaluation
==========================================
Loads the saved best model and evaluates it on holdout videos
stored in step_detection/holdout_data/ (separate from training data).

Produces:
  - Side-by-side comparison table: CV metrics vs holdout metrics
  - Confusion matrix and ROC curve plots for holdout data
  - Per-video breakdown
  - Markdown report saved to results/holdout_report.md

Run:
    python step_detection/evaluate_holdout.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, auc,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
HOLDOUT_DIR = SCRIPT_DIR / "holdout_data"
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR = SCRIPT_DIR / "models"
MODEL_PATH = MODELS_DIR / "best_step_model.joblib"
META_PATH = MODELS_DIR / "best_step_model_meta.json"

# Reuse feature extraction from training script
sys.path.insert(0, str(SCRIPT_DIR))
from train_and_compare import (
    _landmarks_to_features_per_foot,
    MP_LANDMARK_NAMES,
    IDX,
)


# ===========================================================================
# 1.  DATA LOADING (holdout)
# ===========================================================================
def load_holdout_data(holdout_dir: Path = HOLDOUT_DIR) -> pd.DataFrame:
    """Load labels + landmarks from holdout video folders.

    Uses the same per-foot framing as train_and_compare.load_all_data().
    """
    all_rows = []

    if not holdout_dir.exists():
        print(f"❌ Holdout directory not found: {holdout_dir}")
        return pd.DataFrame()

    for video_dir in sorted(holdout_dir.iterdir()):
        if not video_dir.is_dir():
            continue
        labels_path = video_dir / "labels.csv"
        landmarks_path = video_dir / "landmarks.json"
        if not labels_path.exists() or not landmarks_path.exists():
            continue

        video_name = video_dir.name
        labels_df = pd.read_csv(labels_path)
        with open(landmarks_path, "r") as f:
            landmarks = json.load(f)

        # Build lookup: frame → set of sides with contact
        contact_lookup = {}
        for _, row in labels_df.iterrows():
            fr = int(row["frame"])
            contact_lookup.setdefault(fr, set()).add(row["side"])

        n_pos, n_neg = 0, 0
        for frame_str, lm in landmarks.items():
            frame = int(frame_str)
            sides_contacting = contact_lookup.get(frame, set())

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

    return pd.DataFrame(all_rows)


# ===========================================================================
# 2.  EVALUATION
# ===========================================================================
def evaluate(pipeline, meta: dict, df: pd.DataFrame):
    """Run the saved model on holdout data and compute metrics."""
    feature_cols = meta["feature_columns"]

    # Align columns to match training feature order
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    X = df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = df["label"].values.astype(int)

    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_proba),
    }

    # Per-video breakdown
    per_video = {}
    for video_name in df["video"].unique():
        mask = df["video"] == video_name
        v_y = y[mask]
        v_pred = y_pred[mask]
        v_proba = y_proba[mask]
        per_video[video_name] = {
            "n_samples": int(mask.sum()),
            "accuracy": accuracy_score(v_y, v_pred),
            "f1": f1_score(v_y, v_pred),
            "precision": precision_score(v_y, v_pred),
            "recall": recall_score(v_y, v_pred),
            "roc_auc": roc_auc_score(v_y, v_proba),
        }

    return metrics, per_video, y, y_pred, y_proba


# ===========================================================================
# 3.  VISUALIZATIONS
# ===========================================================================
def plot_holdout_confusion_matrix(y, y_pred):
    """Save confusion matrix for holdout data."""
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Flight", "Contact"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Holdout Test — Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = RESULTS_DIR / "holdout_confusion_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Saved {path.name}")


def plot_holdout_roc_curve(y, y_proba):
    """Save ROC curve for holdout data."""
    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc_val = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#6366f1", linewidth=2.5,
            label=f"Holdout (AUC = {roc_auc_val:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Holdout Test — ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    path = RESULTS_DIR / "holdout_roc_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Saved {path.name}")


# ===========================================================================
# 4.  REPORT
# ===========================================================================
def write_holdout_report(holdout_metrics: dict, per_video: dict,
                         cv_metrics: dict, n_holdout: int):
    """Write a markdown report comparing CV vs holdout metrics."""
    lines = [
        "# Holdout Test Evaluation",
        "",
        f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Holdout videos**: {len(per_video)}",
        f"**Holdout samples**: {n_holdout}",
        f"**Model**: {cv_metrics.get('model_name', 'HistGradientBoosting')}",
        "",
        "## CV vs Holdout Comparison",
        "",
        "| Metric | Stratified 5-Fold CV | Holdout Test |",
        "|--------|---------------------|--------------|",
        f"| Accuracy | {cv_metrics.get('accuracy', 0):.4f} | {holdout_metrics['accuracy']:.4f} |",
        f"| F1 Score | {cv_metrics.get('f1', 0):.4f} | {holdout_metrics['f1']:.4f} |",
        f"| Precision | — | {holdout_metrics['precision']:.4f} |",
        f"| Recall | — | {holdout_metrics['recall']:.4f} |",
        f"| ROC AUC | {cv_metrics.get('roc_auc', 0):.4f} | {holdout_metrics['roc_auc']:.4f} |",
        "",
        "## Per-Video Breakdown",
        "",
        "| Video | Samples | Accuracy | F1 | Precision | Recall | ROC AUC |",
        "|-------|---------|----------|----|-----------|--------|---------|",
    ]

    for video_name, vm in per_video.items():
        lines.append(
            f"| {video_name} | {vm['n_samples']} | "
            f"{vm['accuracy']:.4f} | {vm['f1']:.4f} | "
            f"{vm['precision']:.4f} | {vm['recall']:.4f} | "
            f"{vm['roc_auc']:.4f} |"
        )

    lines += [
        "",
        "## Generated Plots",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `holdout_confusion_matrix.png` | Confusion matrix on holdout data |",
        "| `holdout_roc_curve.png` | ROC curve on holdout data |",
        "",
    ]

    report_path = RESULTS_DIR / "holdout_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  📄 Saved {report_path.name}")


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("=" * 60)
    print("  Step Detection — Holdout Test Evaluation")
    print("=" * 60)

    # 1. Load model
    print("\n📦 Loading saved model…")
    if not MODEL_PATH.exists():
        print(f"❌ No saved model at {MODEL_PATH}. Run train_and_compare.py first.")
        return
    pipeline = joblib.load(MODEL_PATH)
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    print(f"  Model: {meta.get('model_name', '?')} "
          f"(CV F1={meta.get('f1', 0):.4f}, trained on {meta.get('n_samples', '?')} samples)")

    # 2. Load holdout data
    print("\n📂 Loading holdout data…")
    df = load_holdout_data()
    if df.empty:
        print("❌ No holdout data found. Label holdout videos first using the "
              "step labeler app with the holdout checkbox enabled.")
        return
    print(f"\n✅ Loaded {len(df)} holdout samples from {df['video'].nunique()} videos")

    # 3. Evaluate
    print("\n🔍 Evaluating on holdout data…")
    holdout_metrics, per_video, y, y_pred, y_proba = evaluate(pipeline, meta, df)

    print(f"\n  📊 Holdout Results:")
    print(f"     Accuracy:  {holdout_metrics['accuracy']:.4f}")
    print(f"     F1 Score:  {holdout_metrics['f1']:.4f}")
    print(f"     Precision: {holdout_metrics['precision']:.4f}")
    print(f"     Recall:    {holdout_metrics['recall']:.4f}")
    print(f"     ROC AUC:   {holdout_metrics['roc_auc']:.4f}")

    # 4. Plots
    print("\n🎨 Generating plots…")
    plot_holdout_confusion_matrix(y, y_pred)
    plot_holdout_roc_curve(y, y_proba)

    # 5. Report
    print("\n📝 Writing holdout report…")
    cv_metrics = {
        "model_name": meta.get("model_name", "?"),
        "accuracy": meta.get("accuracy", 0),
        "f1": meta.get("f1", 0),
        "roc_auc": meta.get("roc_auc", 0),
    }
    write_holdout_report(holdout_metrics, per_video, cv_metrics, len(df))

    # 6. Classification report
    print("\n📋 Full Classification Report:")
    print(classification_report(y, y_pred, target_names=["Flight", "Contact"]))

    print("=" * 60)
    print(f"  ✅ Done! Holdout results saved to {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
