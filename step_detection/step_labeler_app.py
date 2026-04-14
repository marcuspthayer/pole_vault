"""
Pole Vault Step Detection — Standalone Streamlit App
=====================================================
Two tabs:
  1. **Step Labeler** — Upload a video, run pose estimation, and manually
     label ground-contact (step) events. Labels are saved to
     step_detection/data/<video_name>/.
  2. **Test Model** — Upload a video, run the trained best model, view
     annotated output, and download the result.

Launch from the repo root:
    streamlit run step_detection/step_labeler_app.py
"""

import streamlit as st
import cv2
import os
import sys
import json
import tempfile
import numpy as np
import pandas as pd
import mediapipe as mp
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Make repo root importable so we can reuse pvapp modules
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pvapp.pipelines.pose_pipeline import extract_pose_data
from pvapp.core.gait_analysis import detect_foot_strikes

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
HOLDOUT_DIR = SCRIPT_DIR / "holdout_data"
HOLDOUT_DIR.mkdir(exist_ok=True)
MODELS_DIR = SCRIPT_DIR / "models"

JUMP_SIZES = [10, 20, 50]

# ---------------------------------------------------------------------------
# Session-state defaults
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "video_path": None,
    "video_name": None,
    "fps": None,
    "total_frames": 0,
    "width": 0,
    "height": 0,
    "start_frame": 0,
    "end_frame": 0,
    "pose_results": None,        # list[PoseResultWrapper | None]
    "processed_range": None,     # (start, end) that was processed
    "current_frame": 0,
    "labels": [],                # list of dicts
    "label_side": "left",
    "step_range_start": None,    # frame index for start of ground contact
}

for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def read_frame(video_path: str, frame_idx: int):
    """Return a BGR frame (numpy array) or None."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def draw_skeleton(frame_bgr, landmarks):
    """Draw MediaPipe skeleton on a frame. Returns the annotated copy."""
    annotated = frame_bgr.copy()
    if landmarks is None:
        return annotated
    mp_drawing.draw_landmarks(
        annotated,
        landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
    )
    return annotated


def count_saved_videos() -> int:
    """Count how many video folders exist in DATA_DIR."""
    if not DATA_DIR.exists():
        return 0
    return sum(1 for p in DATA_DIR.iterdir() if p.is_dir() and (p / "labels.csv").exists())


def _jump(delta: int):
    """on_click callback — adjusts current_frame before the next render."""
    pr = st.session_state.processed_range
    if pr is None:
        return
    p_start, p_end = pr
    st.session_state.current_frame = max(p_start, min(p_end, st.session_state.current_frame + delta))


def _add_label(frame_idx, side, source="manual"):
    """Add a label for a single frame. Returns True if added, False if duplicate."""
    existing_frames = {l["frame"] for l in st.session_state.labels}
    if frame_idx in existing_frames:
        return False
    x_ankle, y_ankle = None, None
    pr = st.session_state.pose_results
    if pr and frame_idx < len(pr) and pr[frame_idx]:
        pose_res = pr[frame_idx]
        if pose_res.pose_landmarks:
            lm_idx = mp_pose.PoseLandmark.LEFT_ANKLE if side == "left" else mp_pose.PoseLandmark.RIGHT_ANKLE
            ankle = pose_res.pose_landmarks.landmark[lm_idx]
            x_ankle = float(ankle.x)
            y_ankle = float(ankle.y)
    st.session_state.labels.append({
        "frame": frame_idx,
        "time_sec": round(frame_idx / st.session_state.fps, 4) if st.session_state.fps else 0,
        "side": side,
        "source": source,
        "x_ankle": x_ankle,
        "y_ankle": y_ankle,
    })
    return True


def _generate_fps_downsamples(labels_df: pd.DataFrame, original_fps: float, save_dir: Path):
    """Generate synthetic lower-FPS label files by temporal decimation."""
    if original_fps is None or original_fps <= 0 or "frame" not in labels_df.columns:
        return

    target_rates = [120, 60, 30]
    current_fps = original_fps
    current_df = labels_df.copy()

    for target in target_rates:
        if current_fps <= target:
            continue
        ratio = int(round(current_fps / target))
        if ratio < 2:
            continue
        downsampled = current_df[current_df["frame"] % ratio == 0].copy()
        downsampled["frame"] = downsampled["frame"] // ratio
        downsampled["time_sec"] = downsampled["frame"] / target
        downsampled.to_csv(save_dir / f"labels_{int(target)}fps.csv", index=False)
        current_fps = target
        current_df = downsampled


# ============================================================================
# UI
# ============================================================================
st.set_page_config(page_title="PV Step Detection", layout="wide")
st.title("🏃 Pole Vault Step Detection")

tab_label, tab_test = st.tabs(["Step Labeler", "Test Model"])

# ============================================================================
# TAB 1: STEP LABELER
# ============================================================================
with tab_label:
    # ---- Sidebar: label list (renders in sidebar regardless of tab) --------
    with st.sidebar:
        st.header("Step Labels")
        labels = st.session_state.labels

        if labels:
            for i, lbl in enumerate(labels):
                cols = st.columns([3, 1])
                tag = "🤖" if lbl.get("source") == "auto" else "✏️"
                side_emoji = "🦶L" if lbl["side"] == "left" else "🦶R"
                with cols[0]:
                    if st.button(f'{tag} F{lbl["frame"]}  {side_emoji}', key=f"goto_{i}"):
                        st.session_state.current_frame = lbl["frame"]
                        st.rerun()
                with cols[1]:
                    if st.button("🗑️", key=f"del_{i}"):
                        st.session_state.labels.pop(i)
                        st.rerun()
        else:
            st.info("No labels yet. Process a video and start labeling!")

        st.divider()
        st.caption(f"Videos saved: **{count_saved_videos()}**")

    # --- 1. VIDEO UPLOAD ---
    st.header("1 · Upload Video")
    uploaded = st.file_uploader("Upload a pole vault video", type=["mp4", "mov", "avi"])

    if uploaded:
        if st.session_state.video_path is None or st.session_state.video_name != uploaded.name:
            suffix = os.path.splitext(uploaded.name)[1].lower() or ".mp4"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(uploaded.read())
            tmp.close()
            st.session_state.video_path = tmp.name
            st.session_state.video_name = uploaded.name

            cap = cv2.VideoCapture(tmp.name)
            st.session_state.fps = cap.get(cv2.CAP_PROP_FPS)
            st.session_state.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            st.session_state.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            st.session_state.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            st.session_state.start_frame = 0
            st.session_state.end_frame = st.session_state.total_frames - 1
            st.session_state.pose_results = None
            st.session_state.labels = []
            st.session_state.processed_range = None

        fps = st.session_state.fps
        total = st.session_state.total_frames
        w, h = st.session_state.width, st.session_state.height
        duration = total / fps if fps else 0

        st.success(f"**{uploaded.name}** — {w}×{h} · {fps:.1f} fps · {total} frames · {duration:.2f}s")

    # --- 2. START / END FRAME SELECTION ---
    if st.session_state.video_path:
        st.header("2 · Set Approach-Run Window")
        st.caption("Trim to the approach run to avoid processing idle or post-takeoff frames.")

        total = st.session_state.total_frames

        col_s, col_e = st.columns(2)
        with col_s:
            start_f = st.slider("Start Frame", 0, total - 1, st.session_state.start_frame, key="slider_start")
            preview_start = read_frame(st.session_state.video_path, start_f)
            if preview_start is not None:
                st.image(cv2.cvtColor(preview_start, cv2.COLOR_BGR2RGB), caption=f"Start: frame {start_f}", width="stretch")
        with col_e:
            end_f = st.slider("End Frame", 0, total - 1, st.session_state.end_frame, key="slider_end")
            preview_end = read_frame(st.session_state.video_path, end_f)
            if preview_end is not None:
                st.image(cv2.cvtColor(preview_end, cv2.COLOR_BGR2RGB), caption=f"End: frame {end_f}", width="stretch")

        st.session_state.start_frame = start_f
        st.session_state.end_frame = max(end_f, start_f + 1)

        window_frames = st.session_state.end_frame - st.session_state.start_frame
        window_sec = window_frames / st.session_state.fps if st.session_state.fps else 0
        st.info(f"Processing window: **{window_frames}** frames ({window_sec:.2f}s)")

    # --- 3. POSE PROCESSING ---
    if st.session_state.video_path:
        st.header("3 · Run Pose Estimation")

        if st.session_state.pose_results is not None:
            p_start, p_end = st.session_state.processed_range or (0, 0)
            st.success(f"✅ Pose data loaded (frames {p_start}–{p_end})")
        else:
            st.warning("Pose data not yet extracted. Click below to process.")

        if st.button("▶️  Process Video", type="primary"):
            progress = st.progress(0.0)
            status = st.empty()

            def _cb(pct, msg=""):
                progress.progress(min(pct, 1.0))
                status.text(msg)

            with st.spinner("Running YOLO + MediaPipe…"):
                results = extract_pose_data(
                    st.session_state.video_path,
                    start_frame=st.session_state.start_frame,
                    end_frame=st.session_state.end_frame,
                    progress_callback=_cb,
                )
            st.session_state.pose_results = results
            st.session_state.processed_range = (st.session_state.start_frame, st.session_state.end_frame)
            st.session_state.current_frame = st.session_state.start_frame
            st.session_state.labels = []
            progress.progress(1.0)
            status.text("✅ Done!")
            st.rerun()

    # --- 4. FRAME NAVIGATION & LABELING ---
    if st.session_state.pose_results is not None:
        st.header("4 · Navigate & Label Steps")

        p_start, p_end = st.session_state.processed_range
        cur = st.session_state.current_frame

        # Jump buttons
        jump_cols = st.columns(len(JUMP_SIZES) * 2 + 2)
        for idx, size in enumerate(JUMP_SIZES):
            with jump_cols[idx]:
                st.button(f"◀ {size}", key=f"back_{size}", on_click=_jump, args=(-size,))
        with jump_cols[len(JUMP_SIZES)]:
            st.button("◀ 1", key="back_1", on_click=_jump, args=(-1,))
        with jump_cols[len(JUMP_SIZES) + 1]:
            st.button("1 ▶", key="fwd_1", on_click=_jump, args=(1,))
        for idx, size in enumerate(JUMP_SIZES):
            with jump_cols[len(JUMP_SIZES) + 2 + idx]:
                st.button(f"{size} ▶", key=f"fwd_{size}", on_click=_jump, args=(size,))

        # Frame slider
        cur = st.slider("Frame", p_start, p_end, st.session_state.current_frame)
        st.session_state.current_frame = cur

        # Render frame with skeleton
        frame_col, info_col = st.columns([3, 1])
        with frame_col:
            frame_bgr = read_frame(st.session_state.video_path, cur)
            if frame_bgr is not None:
                pose_res = st.session_state.pose_results[cur] if cur < len(st.session_state.pose_results) else None
                lms = pose_res.pose_landmarks if pose_res else None
                annotated = draw_skeleton(frame_bgr, lms)

                for lbl in st.session_state.labels:
                    if lbl["frame"] == cur:
                        color = (0, 255, 0) if lbl["side"] == "left" else (0, 165, 255)
                        cv2.putText(annotated, f'STEP ({lbl["side"].upper()})', (30, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

                if st.session_state.step_range_start is not None and cur == st.session_state.step_range_start:
                    cv2.putText(annotated, "RANGE START", (30, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)

                time_sec = cur / st.session_state.fps if st.session_state.fps else 0
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                         caption=f"Frame {cur} · {time_sec:.3f}s",
                         width="stretch")

        with info_col:
            time_sec = cur / st.session_state.fps if st.session_state.fps else 0
            st.metric("Frame", cur)
            st.metric("Time", f"{time_sec:.3f}s")
            labeled_here = [l for l in st.session_state.labels if l["frame"] == cur]
            if labeled_here:
                st.success(f"Labeled: {labeled_here[0]['side'].upper()}")
            if st.session_state.step_range_start is not None:
                st.warning(f"Range start: F{st.session_state.step_range_start}")

        # Labeling controls
        st.subheader("Label Controls")
        side = st.selectbox("Foot", ["left", "right"], key="side_select")

        single_col, _, range_start_col, range_end_col = st.columns([2, 0.5, 2, 2])
        with single_col:
            if st.button("🏷️  Mark Single Frame", type="primary", width="stretch"):
                if _add_label(cur, side):
                    st.session_state.labels.sort(key=lambda l: l["frame"])
                    st.rerun()
                else:
                    st.warning("Frame already labeled.")

        with range_start_col:
            if st.button("📍 Mark Range Start", width="stretch"):
                st.session_state.step_range_start = cur
                st.rerun()

        with range_end_col:
            range_start = st.session_state.step_range_start
            btn_disabled = range_start is None
            label_text = "📍 Mark Range End & Fill" if not btn_disabled else "📍 Set Start First"
            if st.button(label_text, disabled=btn_disabled, width="stretch"):
                if range_start is not None:
                    r_start = min(range_start, cur)
                    r_end = max(range_start, cur)
                    added = 0
                    for f in range(r_start, r_end + 1):
                        if _add_label(f, side, source="range"):
                            added += 1
                    st.session_state.labels.sort(key=lambda l: l["frame"])
                    st.session_state.step_range_start = None
                    st.toast(f"Filled {added} frames (F{r_start}–F{r_end})")
                    st.rerun()

        # Timeline visualization
        if st.session_state.labels:
            st.subheader("Label Timeline")
            label_frames = [l["frame"] for l in st.session_state.labels]
            label_sides = [l["side"] for l in st.session_state.labels]
            timeline_df = pd.DataFrame({
                "frame": label_frames,
                "side": label_sides,
                "value": [1 if s == "left" else -1 for s in label_sides],
            })
            st.bar_chart(timeline_df.set_index("frame")["value"], height=120)
            st.caption("Positive = left foot · Negative = right foot")

    # --- 5. AUTO-SUGGEST STEPS ---
    if st.session_state.pose_results is not None:
        st.header("5 · Auto-Suggest Steps")

        saved_count = count_saved_videos()
        if saved_count < 3:
            st.warning(
                f"Auto-suggest is **unlocked after 3 videos** have been manually labeled and saved. "
                f"You have **{saved_count}** so far. Keep labeling!"
            )
        else:
            st.info("Uses the heuristic foot-strike detector from the main app as a starting point.")
            if st.button("🤖  Auto-Suggest"):
                p_start, p_end = st.session_state.processed_range
                pose_list = st.session_state.pose_results

                landmarks_window = []
                frame_indices = []
                for i in range(p_start, p_end + 1):
                    pr = pose_list[i] if i < len(pose_list) else None
                    lm = pr.pose_landmarks if pr else None
                    landmarks_window.append(lm)
                    frame_indices.append(i)

                strikes, _, _ = detect_foot_strikes(
                    landmarks_window,
                    frame_indices,
                    fps=st.session_state.fps or 30,
                )

                existing_frames = {l["frame"] for l in st.session_state.labels}
                added = 0
                for s in strikes:
                    if s["frame"] not in existing_frames:
                        x_a, y_a = s.get("pt", (None, None)) or (None, None)
                        st.session_state.labels.append({
                            "frame": s["frame"],
                            "time_sec": round(s["frame"] / st.session_state.fps, 4) if st.session_state.fps else 0,
                            "side": s.get("side", "unknown"),
                            "source": "auto",
                            "x_ankle": round(float(x_a), 6) if x_a is not None else None,
                            "y_ankle": round(float(y_a), 6) if y_a is not None else None,
                        })
                        added += 1

                st.session_state.labels.sort(key=lambda l: l["frame"])
                st.success(f"Added **{added}** auto-detected steps.")
                st.rerun()

    # --- 6. SAVE DATA ---
    if st.session_state.pose_results is not None and st.session_state.labels:
        st.header("6 · Save Labels")

        video_stem = Path(st.session_state.video_name or "unknown").stem
        holdout_mode = st.checkbox(
            "🧪 Save as **holdout** test data (separate from training data)",
            help="When checked, labels are saved to `step_detection/holdout_data/` "
                 "instead of `step_detection/data/` so they are never used for training.",
        )
        base_dir = HOLDOUT_DIR if holdout_mode else DATA_DIR
        save_dir = base_dir / video_stem
        dir_label = "holdout_data" if holdout_mode else "data"
        st.caption(f"Data will be saved to `step_detection/{dir_label}/{video_stem}/`")

        if st.button("💾  Save Labels & Landmarks", type="primary"):
            save_dir.mkdir(parents=True, exist_ok=True)

            # labels.csv
            labels_df = pd.DataFrame(st.session_state.labels)
            labels_df.to_csv(save_dir / "labels.csv", index=False)

            # landmarks.json
            landmarks_export = {}
            for lbl in st.session_state.labels:
                fidx = lbl["frame"]
                pr = st.session_state.pose_results[fidx] if fidx < len(st.session_state.pose_results) else None
                if pr and pr.pose_landmarks:
                    lm_data = []
                    for lm in pr.pose_landmarks.landmark:
                        lm_data.append({
                            "x": float(lm.x),
                            "y": float(lm.y),
                            "z": float(lm.z),
                            "visibility": float(lm.visibility),
                        })
                    landmarks_export[str(fidx)] = lm_data
            with open(save_dir / "landmarks.json", "w") as f:
                json.dump(landmarks_export, f, indent=2)

            # metadata.json
            meta = {
                "video_name": st.session_state.video_name,
                "fps": st.session_state.fps,
                "width": st.session_state.width,
                "height": st.session_state.height,
                "total_frames": st.session_state.total_frames,
                "processed_start": st.session_state.processed_range[0],
                "processed_end": st.session_state.processed_range[1],
                "label_count": len(st.session_state.labels),
                "date_labeled": datetime.now().isoformat(),
            }
            with open(save_dir / "metadata.json", "w") as f:
                json.dump(meta, f, indent=2)

            # FPS Downsampling
            _generate_fps_downsamples(labels_df, st.session_state.fps, save_dir)

            st.success(f"✅ Saved {len(st.session_state.labels)} labels to `{save_dir.relative_to(REPO_ROOT)}`")


# ============================================================================
# TAB 2: TEST MODEL
# ============================================================================
with tab_test:
    st.header("Test Step Detection Model")

    # Check if a trained model exists
    model_path = MODELS_DIR / "best_step_model.joblib"
    meta_path = MODELS_DIR / "best_step_model_meta.json"
    model_exists = model_path.exists() and meta_path.exists()

    if not model_exists:
        st.warning(
            "No trained model found. Run `python step_detection/train_and_compare.py` "
            "to train and save the best model first."
        )
    else:
        # Show model info
        with open(meta_path, "r") as f:
            model_meta = json.load(f)
        st.success(
            f"Model: **{model_meta['model_name']}** — "
            f"F1={model_meta['f1']:.3f}, AUC={model_meta['roc_auc']:.3f} — "
            f"Trained on {model_meta['n_samples']} samples"
        )

        # Video upload for testing
        test_video = st.file_uploader(
            "Upload a video to test", type=["mp4", "mov", "avi"],
            key="test_video_upload",
        )

        if test_video:
            # Save to temp
            if "test_video_path" not in st.session_state or st.session_state.get("test_video_name") != test_video.name:
                suffix = os.path.splitext(test_video.name)[1].lower() or ".mp4"
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.write(test_video.read())
                tmp.close()
                st.session_state.test_video_path = tmp.name
                st.session_state.test_video_name = test_video.name

                cap = cv2.VideoCapture(tmp.name)
                st.session_state.test_fps = cap.get(cv2.CAP_PROP_FPS)
                st.session_state.test_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                st.session_state.test_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                st.session_state.test_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()

                st.session_state.test_result = None

            t_fps = st.session_state.test_fps
            t_total = st.session_state.test_total_frames
            t_w, t_h = st.session_state.test_width, st.session_state.test_height
            t_dur = t_total / t_fps if t_fps else 0

            st.info(
                f"**{test_video.name}** — {t_w}x{t_h} · {t_fps:.1f} fps · "
                f"{t_total} frames · {t_dur:.2f}s"
            )

            # Frame range selection
            st.subheader("Frame Range")
            t_col_s, t_col_e = st.columns(2)
            with t_col_s:
                t_start = st.slider("Start Frame", 0, t_total - 1, 0, key="test_start")
                preview = read_frame(st.session_state.test_video_path, t_start)
                if preview is not None:
                    st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                             caption=f"Start: frame {t_start}", width="stretch")
            with t_col_e:
                t_end = st.slider("End Frame", 0, t_total - 1, t_total - 1, key="test_end")
                preview = read_frame(st.session_state.test_video_path, t_end)
                if preview is not None:
                    st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                             caption=f"End: frame {t_end}", width="stretch")

            t_end = max(t_end, t_start + 1)
            window_frames = t_end - t_start
            window_sec = window_frames / t_fps if t_fps else 0
            st.caption(f"Processing window: **{window_frames}** frames ({window_sec:.2f}s)")

            # Model selection
            st.subheader("Model")
            from step_detection.inference import CNN_MODEL_PATH, MODEL_60FPS_PATH

            model_options = ["MLP (original)"]
            if MODEL_60FPS_PATH.exists():
                model_options.insert(0, "MLP (60fps)")
            if CNN_MODEL_PATH.exists():
                model_options.insert(0, "CNN (60fps)")
            model_choice = st.radio(
                "Step detection model",
                model_options,
                horizontal=True,
                key="test_model_choice",
            )
            if "CNN" in model_choice:
                model_type = "cnn"
            elif "60fps" in model_choice:
                model_type = "mlp_60fps"
            else:
                model_type = "mlp"

            # Train buttons
            col_train_mlp, col_train_cnn, col_run = st.columns(3)
            with col_train_mlp:
                if st.button("Train MLP 60fps", key="train_mlp60"):
                    train_status = st.empty()
                    train_status.info("Training MLP + others on 60fps data...")
                    try:
                        import subprocess
                        result = subprocess.run(
                            [sys.executable, "step_detection/train_and_compare.py", "--fps60"],
                            capture_output=True, text=True, cwd=str(REPO_ROOT),
                        )
                        if result.returncode == 0:
                            train_status.success("60fps models trained! Select 'MLP (60fps)' above.")
                        else:
                            train_status.error(f"Training failed:\n{result.stderr[-500:]}")
                        st.rerun()
                    except Exception as e:
                        train_status.error(f"Training failed: {e}")
            with col_train_cnn:
                if st.button("Train CNN", key="train_cnn"):
                    train_status = st.empty()
                    train_status.info("Training CNN on 60fps data...")
                    try:
                        from step_detection.train_cnn import train as train_cnn
                        train_cnn()
                        train_status.success("CNN trained! Select 'CNN (60fps)' above.")
                        st.rerun()
                    except Exception as e:
                        train_status.error(f"Training failed: {e}")

            # Run button
            with col_run:
                run_clicked = st.button("Run Step Detection", type="primary", key="run_test")

            if run_clicked:
                from step_detection.inference import run_step_inference

                progress = st.progress(0.0)
                status = st.empty()

                def _test_cb(pct, msg=""):
                    progress.progress(min(pct, 1.0))
                    status.text(msg)

                with st.spinner("Running step detection..."):
                    result = run_step_inference(
                        st.session_state.test_video_path,
                        start_frame=t_start,
                        end_frame=t_end,
                        progress_callback=_test_cb,
                        model_type=model_type,
                    )

                st.session_state.test_result = result
                progress.progress(1.0)
                status.text("Done!")
                st.rerun()

            # Display results
            if st.session_state.get("test_result"):
                result = st.session_state.test_result
                preds = result["predictions"]
                steps = result["steps"]

                # ---- Summary metrics ----
                st.subheader("Results")

                raw_left = sum(1 for p in preds if p["left_contact"])
                raw_right = sum(1 for p in preds if p["right_contact"])
                clean_left = sum(1 for s in steps if s["side"] == "left")
                clean_right = sum(1 for s in steps if s["side"] == "right")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Clean Steps", len(steps))
                m2.metric("Left / Right", f"{clean_left} / {clean_right}")
                m3.metric("Raw Contact Frames (L)", raw_left)
                m4.metric("Raw Contact Frames (R)", raw_right)

                # ---- Step table ----
                if steps:
                    st.subheader("Detected Steps")
                    step_rows = []
                    for s in steps:
                        dur = s["end_frame"] - s["start_frame"] + 1
                        step_rows.append({
                            "Step": s["step_number"],
                            "Side": s["side"].capitalize(),
                            "Start Frame": s["start_frame"],
                            "End Frame": s["end_frame"],
                            "Duration (frames)": dur,
                        })
                    st.dataframe(pd.DataFrame(step_rows), width="stretch",
                                 hide_index=True)

                # ---- Clean video ----
                st.subheader("Clean Steps Video")
                st.caption(
                    "Filtered steps with persistent dot markers. "
                    "Yellow flash = touchdown, bright = active contact, "
                    "dim = lifted off (dot persists)."
                )
                clean_path = result["clean_video_path"]
                if Path(clean_path).exists():
                    st.video(clean_path)
                    with open(clean_path, "rb") as vf:
                        st.download_button(
                            "Download Clean Video",
                            data=vf.read(),
                            file_name=f"steps_clean_{Path(st.session_state.test_video_name).stem}.mp4",
                            mime="video/mp4",
                        )

                # ---- Raw video ----
                st.subheader("Raw Predictions Video")
                st.caption(
                    "Every per-frame model prediction shown as-is (no filtering)."
                )
                raw_path = result["raw_video_path"]
                if Path(raw_path).exists():
                    st.video(raw_path)
                    with open(raw_path, "rb") as vf:
                        st.download_button(
                            "Download Raw Video",
                            data=vf.read(),
                            file_name=f"steps_raw_{Path(st.session_state.test_video_name).stem}.mp4",
                            mime="video/mp4",
                        )

                # ---- Step timeline chart ----
                st.subheader("Step Timeline")
                timeline_data = []
                for p in preds:
                    if p["left_contact"]:
                        timeline_data.append({
                            "frame": p["frame"],
                            "side": "left",
                            "probability": p["left_prob"],
                        })
                    if p["right_contact"]:
                        timeline_data.append({
                            "frame": p["frame"],
                            "side": "right",
                            "probability": p["right_prob"],
                        })

                if timeline_data:
                    tl_df = pd.DataFrame(timeline_data)
                    tl_df["value"] = tl_df["side"].map({"left": 1, "right": -1})
                    st.bar_chart(tl_df.set_index("frame")["value"], height=150)
                    st.caption("Positive = left foot · Negative = right foot")

                # ---- CSV exports ----
                st.subheader("Export Data")
                exp1, exp2 = st.columns(2)
                with exp1:
                    pred_df = pd.DataFrame(preds)
                    st.download_button(
                        "Download Raw Predictions CSV",
                        data=pred_df.to_csv(index=False),
                        file_name=f"step_predictions_{Path(st.session_state.test_video_name).stem}.csv",
                        mime="text/csv",
                    )
                with exp2:
                    if steps:
                        steps_df = pd.DataFrame(steps)
                        st.download_button(
                            "Download Clean Steps CSV",
                            data=steps_df.to_csv(index=False),
                            file_name=f"step_clean_{Path(st.session_state.test_video_name).stem}.csv",
                            mime="text/csv",
                        )
