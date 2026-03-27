"""
Step Detection — Annotation Visualizer
=======================================
For each video, plots the ankle x/y trajectory over frames, with
ground-contact labels highlighted. Helps visually verify labeling quality.

Run:
    python visualize_annotations.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# MediaPipe landmark indices
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32
LEFT_KNEE = 25
RIGHT_KNEE = 26


def load_video_data(video_dir: Path):
    """Load labels, landmarks, and metadata for a single video."""
    labels_df = pd.read_csv(video_dir / "labels.csv")
    with open(video_dir / "landmarks.json") as f:
        landmarks = json.load(f)
    with open(video_dir / "metadata.json") as f:
        meta = json.load(f)
    return labels_df, landmarks, meta


def extract_ankle_series(landmarks: dict, labels_df: pd.DataFrame):
    """Extract time-series of ankle positions for ALL frames in landmarks.json,
    and tag each frame with its label status."""

    rows = []
    labeled_frames = set(labels_df["frame"].values)
    label_map = {}
    for _, r in labels_df.iterrows():
        label_map[int(r["frame"])] = r["side"]

    for frame_str, joints in sorted(landmarks.items(), key=lambda x: int(x[0])):
        frame = int(frame_str)
        la_x = joints[LEFT_ANKLE]["x"]
        la_y = joints[LEFT_ANKLE]["y"]
        ra_x = joints[RIGHT_ANKLE]["x"]
        ra_y = joints[RIGHT_ANKLE]["y"]
        lh_y = joints[LEFT_HEEL]["y"]
        rh_y = joints[RIGHT_HEEL]["y"]
        lfi_y = joints[LEFT_FOOT_INDEX]["y"]
        rfi_y = joints[RIGHT_FOOT_INDEX]["y"]
        lk_y = joints[LEFT_KNEE]["y"]
        rk_y = joints[RIGHT_KNEE]["y"]

        is_labeled = frame in labeled_frames
        side = label_map.get(frame, "none")

        rows.append({
            "frame": frame,
            "left_ankle_x": la_x, "left_ankle_y": la_y,
            "right_ankle_x": ra_x, "right_ankle_y": ra_y,
            "left_heel_y": lh_y, "right_heel_y": rh_y,
            "left_foot_y": lfi_y, "right_foot_y": rfi_y,
            "left_knee_y": lk_y, "right_knee_y": rk_y,
            "labeled": is_labeled,
            "side": side,
        })

    return pd.DataFrame(rows).sort_values("frame").reset_index(drop=True)


def plot_video_annotation(video_name: str, series: pd.DataFrame,
                          labels_df: pd.DataFrame, meta: dict, ax_left, ax_right, ax_path):
    """Plot ankle trajectories for one video across the 3 provided axes."""
    fps = meta.get("fps", 120)
    frames = series["frame"].values

    # --- Left ankle y over frames ---
    ax = ax_left
    ax.plot(frames, series["left_ankle_y"], color="#94a3b8", linewidth=0.8,
            alpha=0.7, label="Ankle Y", zorder=1)
    ax.plot(frames, series["left_heel_y"], color="#cbd5e1", linewidth=0.5,
            alpha=0.5, label="Heel Y", zorder=1)

    # Highlight labeled left-foot contacts
    left_mask = series["side"] == "left"
    if left_mask.any():
        ax.scatter(frames[left_mask], series.loc[left_mask, "left_ankle_y"],
                   color="#6366f1", s=12, zorder=3, label="Labeled contact")

    # Shade contact regions
    _shade_contacts(ax, series, "left", "#6366f1")
    ax.set_title(f"LEFT foot — {video_name}", fontsize=10, fontweight="bold")
    ax.set_ylabel("Y position (↓ = ground)", fontsize=9)
    ax.set_xlabel("Frame", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    ax.invert_yaxis()  # Higher y = lower in image = closer to ground → put at bottom
    ax.grid(alpha=0.2)

    # --- Right ankle y over frames ---
    ax = ax_right
    ax.plot(frames, series["right_ankle_y"], color="#94a3b8", linewidth=0.8,
            alpha=0.7, label="Ankle Y", zorder=1)
    ax.plot(frames, series["right_heel_y"], color="#cbd5e1", linewidth=0.5,
            alpha=0.5, label="Heel Y", zorder=1)

    right_mask = series["side"] == "right"
    if right_mask.any():
        ax.scatter(frames[right_mask], series.loc[right_mask, "right_ankle_y"],
                   color="#f43f5e", s=12, zorder=3, label="Labeled contact")

    _shade_contacts(ax, series, "right", "#f43f5e")
    ax.set_title(f"RIGHT foot — {video_name}", fontsize=10, fontweight="bold")
    ax.set_ylabel("Y position (↓ = ground)", fontsize=9)
    ax.set_xlabel("Frame", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    ax.invert_yaxis()
    ax.grid(alpha=0.2)

    # --- Spatial path (x vs y) with contacts colored ---
    ax = ax_path
    # Plot all ankle positions faintly
    ax.plot(series["left_ankle_x"], series["left_ankle_y"],
            color="#c7d2fe", linewidth=0.6, alpha=0.5, zorder=1)
    ax.plot(series["right_ankle_x"], series["right_ankle_y"],
            color="#fecdd3", linewidth=0.6, alpha=0.5, zorder=1)

    # Highlight contacts
    if left_mask.any():
        ax.scatter(series.loc[left_mask, "left_ankle_x"],
                   series.loc[left_mask, "left_ankle_y"],
                   color="#6366f1", s=14, alpha=0.8, label="Left contact", zorder=3)
    if right_mask.any():
        ax.scatter(series.loc[right_mask, "right_ankle_x"],
                   series.loc[right_mask, "right_ankle_y"],
                   color="#f43f5e", s=14, alpha=0.8, label="Right contact", zorder=3)

    # Non-contact frames
    no_label = series["side"] == "none"
    if no_label.any():
        ax.scatter(series.loc[no_label, "left_ankle_x"],
                   series.loc[no_label, "left_ankle_y"],
                   color="#94a3b8", s=4, alpha=0.3, zorder=2)
        ax.scatter(series.loc[no_label, "right_ankle_x"],
                   series.loc[no_label, "right_ankle_y"],
                   color="#94a3b8", s=4, alpha=0.3, zorder=2)

    # Direction arrow (start → end)
    if len(series) > 1:
        ax.annotate("", xy=(series["left_ankle_x"].iloc[-1], series["left_ankle_y"].iloc[-1]),
                     xytext=(series["left_ankle_x"].iloc[0], series["left_ankle_y"].iloc[0]),
                     arrowprops=dict(arrowstyle="->", color="#6366f1", lw=1.5, alpha=0.4))

    ax.set_title(f"Foot Path (X vs Y) — {video_name}", fontsize=10, fontweight="bold")
    ax.set_xlabel("X position (→ = right in frame)", fontsize=9)
    ax.set_ylabel("Y position (↓ = ground)", fontsize=9)
    ax.invert_yaxis()
    ax.legend(fontsize=7, loc="upper right")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.2)


def _shade_contacts(ax, series, side, color):
    """Shade vertical spans for consecutive contact frames."""
    mask = series["side"] == side
    if not mask.any():
        return
    frames = series.loc[mask, "frame"].values
    # Find consecutive runs
    runs = []
    start = frames[0]
    prev = frames[0]
    for f in frames[1:]:
        if f - prev > 3:
            runs.append((start, prev))
            start = f
        prev = f
    runs.append((start, prev))

    for s, e in runs:
        ax.axvspan(s - 0.5, e + 0.5, alpha=0.12, color=color, zorder=0)


def main():
    print("=" * 60)
    print("  Step Detection — Annotation Visualizer")
    print("=" * 60)

    video_dirs = sorted([d for d in DATA_DIR.iterdir()
                         if d.is_dir() and (d / "labels.csv").exists()])

    if not video_dirs:
        print("❌ No data found.")
        return

    n_videos = len(video_dirs)
    fig, axes = plt.subplots(n_videos, 3, figsize=(22, 5.5 * n_videos))
    if n_videos == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle("Foot Trajectory & Step Annotations — All Videos",
                 fontsize=18, fontweight="bold", y=1.0)

    for i, video_dir in enumerate(video_dirs):
        video_name = video_dir.name
        print(f"  📂 {video_name}…")

        labels_df, landmarks, meta = load_video_data(video_dir)
        series = extract_ankle_series(landmarks, labels_df)

        plot_video_annotation(video_name, series, labels_df, meta,
                              axes[i, 0], axes[i, 1], axes[i, 2])

    plt.tight_layout()
    out_path = RESULTS_DIR / "annotation_debug.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✅ Saved {out_path}")

    # --- Also make individual per-video plots (higher detail) ---
    for video_dir in video_dirs:
        video_name = video_dir.name
        labels_df, landmarks, meta = load_video_data(video_dir)
        series = extract_ankle_series(landmarks, labels_df)

        fig2, (a1, a2, a3) = plt.subplots(1, 3, figsize=(22, 6))
        fig2.suptitle(f"Annotation Detail — {video_name}",
                      fontsize=14, fontweight="bold")
        plot_video_annotation(video_name, series, labels_df, meta, a1, a2, a3)
        plt.tight_layout()
        safe_name = video_name.replace(" ", "_").replace("#", "")
        fig2.savefig(RESULTS_DIR / f"annotation_{safe_name}.png",
                     dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"  📊 Saved annotation_{safe_name}.png")

    print(f"\n{'='*60}")
    print(f"  ✅ Done! All annotation plots saved to {RESULTS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
