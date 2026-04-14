"""
Step Detection — Video Inference
=================================
Loads the saved best model, runs pose estimation on a video,
extracts per-foot biomechanical features, predicts ground-contact
frames, and renders annotated output videos.

Produces two videos:
  1. **Raw** — shows every per-frame model prediction as-is
  2. **Clean** — groups predictions into discrete steps, filters noise,
     and renders a persistent-dot visualization showing touchdown,
     contact, and liftoff phases.

Usage (standalone):
    python step_detection/inference.py path/to/video.mp4

As a module:
    from step_detection.inference import run_step_inference
    result = run_step_inference("video.mp4", progress_callback=...)
"""

import json
import sys
import tempfile
import subprocess
from pathlib import Path

import cv2
import joblib
import numpy as np
import mediapipe as mp_lib

# Make repo root importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pvapp.pipelines.pose_pipeline import extract_pose_data
from step_detection.train_and_compare import (
    _landmarks_to_features_per_foot,
    MP_LANDMARK_NAMES,
    IDX,
)

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"
MODEL_PATH = MODELS_DIR / "best_step_model.joblib"
META_PATH = MODELS_DIR / "best_step_model_meta.json"
MODEL_60FPS_PATH = MODELS_DIR / "best_step_model_60fps.joblib"
META_60FPS_PATH = MODELS_DIR / "best_step_model_60fps_meta.json"
CNN_MODEL_PATH = MODELS_DIR / "step_cnn.pt"

# MediaPipe landmark indices
L_ANKLE = 27
R_ANKLE = 28

# Visualization colors (BGR)
CLR_LEFT_ACTIVE   = (0, 230, 0)       # bright green
CLR_LEFT_LIFTOFF  = (0, 140, 0)       # dim green
CLR_RIGHT_ACTIVE  = (0, 165, 255)     # bright orange
CLR_RIGHT_LIFTOFF = (0, 100, 180)     # dim orange
CLR_TOUCHDOWN     = (0, 255, 255)     # yellow flash
CLR_WHITE         = (255, 255, 255)

mp_drawing = mp_lib.solutions.drawing_utils
mp_pose = mp_lib.solutions.pose
mp_drawing_styles = mp_lib.solutions.drawing_styles


# ============================================================================
# Model loading
# ============================================================================
def load_model(prefer_60fps=False):
    """Load the saved best model and its metadata.

    If prefer_60fps=True and the 60fps model exists, load that instead.
    """
    if prefer_60fps and MODEL_60FPS_PATH.exists():
        pipeline = joblib.load(MODEL_60FPS_PATH)
        with open(META_60FPS_PATH, "r") as f:
            meta = json.load(f)
        meta["model_name"] = meta.get("model_name", "unknown") + " (60fps)"
        return pipeline, meta

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No saved model found at {MODEL_PATH}. "
            "Run train_and_compare.py first to train and save the best model."
        )
    pipeline = joblib.load(MODEL_PATH)
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    return pipeline, meta


# ============================================================================
# Feature extraction & prediction
# ============================================================================
def _pose_to_landmark_list(pose_result):
    """Convert a pose pipeline result to the landmark dict list format
    used by train_and_compare feature extraction."""
    if pose_result is None or pose_result.pose_landmarks is None:
        return None
    lm_list = []
    for lm in pose_result.pose_landmarks.landmark:
        lm_list.append({
            "x": float(lm.x),
            "y": float(lm.y),
            "z": float(lm.z),
            "visibility": float(lm.visibility),
        })
    return lm_list


def predict_steps(pose_results, feature_cols, pipeline,
                   start_frame=0, end_frame=None):
    """Run the model on pose results, return per-frame predictions.

    pose_results is indexed by absolute frame number (with None padding
    for frames before start_frame).

    Returns a list of dicts:
        [{"frame": int, "left_contact": bool, "right_contact": bool,
          "left_prob": float, "right_prob": float}, ...]
    """
    if end_frame is None:
        end_frame = len(pose_results) - 1

    predictions = []

    for frame_idx in range(start_frame, end_frame + 1):
        pose_res = pose_results[frame_idx] if frame_idx < len(pose_results) else None
        lm_list = _pose_to_landmark_list(pose_res)

        entry = {
            "frame": frame_idx,
            "left_contact": False,
            "right_contact": False,
            "left_prob": 0.0,
            "right_prob": 0.0,
        }

        if lm_list is None:
            predictions.append(entry)
            continue

        for side in ("left", "right"):
            features = _landmarks_to_features_per_foot(lm_list, side)
            feat_vec = np.array(
                [features.get(col, 0.0) for col in feature_cols],
                dtype=np.float32,
            ).reshape(1, -1)
            feat_vec = np.nan_to_num(feat_vec, nan=0.0, posinf=0.0, neginf=0.0)

            pred = pipeline.predict(feat_vec)[0]
            prob = 0.0
            if hasattr(pipeline, "predict_proba"):
                prob = float(pipeline.predict_proba(feat_vec)[0, 1])

            entry[f"{side}_contact"] = bool(pred)
            entry[f"{side}_prob"] = prob

        predictions.append(entry)

    return predictions


# ============================================================================
# CNN model loading & prediction
# ============================================================================
def load_cnn_model():
    """Load the CNN step detection model. Returns (model, checkpoint) or None."""
    if not CNN_MODEL_PATH.exists():
        return None
    try:
        import torch
        from step_detection.train_cnn import build_cnn_model
        checkpoint = torch.load(str(CNN_MODEL_PATH), map_location="cpu", weights_only=False)
        model = build_cnn_model(input_shape=(checkpoint['window_size'], checkpoint['input_features']))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, checkpoint
    except Exception as e:
        print(f"Failed to load CNN model: {e}")
        return None


def _swap_sides_array(frame_arr):
    """Mirror left/right joints for per-foot framing (numpy array version)."""
    from step_detection.train_cnn import _SWAP_PAIRS, IDX as CNN_IDX
    swapped = frame_arr.copy()
    for left_name, right_name in _SWAP_PAIRS:
        li, ri = CNN_IDX[left_name], CNN_IDX[right_name]
        swapped[li], swapped[ri] = frame_arr[ri].copy(), frame_arr[li].copy()
    return swapped


def predict_steps_cnn(pose_results, model, checkpoint,
                      start_frame=0, end_frame=None, fps=60.0):
    """Run the CNN model on pose results, return per-frame predictions.

    Uses sliding windows of landmark sequences with per-foot framing.
    If fps > 60, subsamples frames so the CNN sees 60fps temporal spacing.
    Predictions are mapped back to original frame indices.
    Same output format as predict_steps() for drop-in replacement.
    """
    import torch

    window_size = checkpoint['window_size']
    target_fps = checkpoint.get('target_fps', 60)
    train_mean = checkpoint['train_mean']  # (1, window_size, 132)
    train_std = checkpoint['train_std']
    half = window_size // 2

    if end_frame is None:
        end_frame = len(pose_results) - 1

    # Build dense landmark array for the frame range
    total = end_frame - start_frame + 1
    landmark_seq_full = np.zeros((total, 33, 4), dtype=np.float32)

    for i in range(total):
        frame_idx = start_frame + i
        if frame_idx < len(pose_results):
            lm_list = _pose_to_landmark_list(pose_results[frame_idx])
            if lm_list is not None:
                for j, joint in enumerate(lm_list):
                    landmark_seq_full[i, j] = [joint["x"], joint["y"], joint["z"], joint["visibility"]]

    # Subsample to 60fps if video is higher FPS
    stride = max(1, round(fps / target_fps))
    if stride > 1:
        # Pick every stride-th frame for CNN input
        subsample_indices = list(range(0, total, stride))
        landmark_seq = landmark_seq_full[subsample_indices]
    else:
        subsample_indices = list(range(total))
        landmark_seq = landmark_seq_full

    seq_len = len(landmark_seq)

    # Run CNN on subsampled sequence, then map predictions back
    sub_predictions = {}  # subsample index -> (left_prob, right_prob)

    for i in range(seq_len):
        if np.sum(np.abs(landmark_seq[i])) < 0.01:
            sub_predictions[i] = (0.0, 0.0)
            continue

        # Build padded window
        pad_start = max(0, i - half)
        pad_end = min(seq_len, i + half + 1)
        window = landmark_seq[pad_start:pad_end]

        if len(window) < window_size:
            if pad_start == 0:
                pad_needed = window_size - len(window)
                window = np.concatenate([np.tile(window[0:1], (pad_needed, 1, 1)), window])
            else:
                pad_needed = window_size - len(window)
                window = np.concatenate([window, np.tile(window[-1:], (pad_needed, 1, 1))])

        left_prob = 0.0
        right_prob = 0.0

        for side in ("left", "right"):
            if side == "right":
                w = np.stack([_swap_sides_array(window[t]) for t in range(window_size)])
            else:
                w = window.copy()

            w_flat = w.reshape(window_size, -1).astype(np.float32)
            w_norm = (w_flat[np.newaxis] - train_mean) / train_std

            with torch.no_grad():
                logits = model(torch.from_numpy(w_norm))
                prob = float(torch.sigmoid(logits).item())

            if side == "left":
                left_prob = prob
            else:
                right_prob = prob

        sub_predictions[i] = (left_prob, right_prob)

    # Map back to original frame indices
    # For frames between subsampled points, use nearest subsampled prediction
    predictions = []

    for i in range(total):
        frame_idx = start_frame + i

        # Find nearest subsampled index
        sub_idx = min(range(len(subsample_indices)),
                      key=lambda si: abs(subsample_indices[si] - i))
        left_prob, right_prob = sub_predictions.get(sub_idx, (0.0, 0.0))

        predictions.append({
            "frame": frame_idx,
            "left_contact": left_prob > 0.5,
            "right_contact": right_prob > 0.5,
            "left_prob": left_prob,
            "right_prob": right_prob,
        })

    return predictions


# ============================================================================
# Step cleaning / grouping
# ============================================================================
def clean_predictions(predictions, fps=240,
                      min_step_seconds=0.05, max_gap_seconds=0.05,
                      min_inter_step_seconds=0.12):
    """Convert per-frame contact predictions into clean, discrete steps.

    Algorithm:
      1. For each frame, pick the dominant side: whichever foot has
         higher probability (left_prob vs right_prob). If neither foot
         crosses 0.3 probability, mark as no-contact (None).
      2. Run-length encode the dominant side sequence into segments
         of consecutive same-side frames.
      3. Absorb short segments (< min_step_frames) into their neighbors.
      4. Merge adjacent segments of the same side.
      5. Drop remaining short segments.
      6. Place the touchdown at each segment's midpoint.

    Returns:
        list of dicts with: side, start_frame, end_frame,
        touchdown_frame, liftoff_frame, step_number
    """
    if not predictions:
        return []

    min_step_frames = max(3, round(fps * min_step_seconds))
    min_contact_prob = 0.3  # at least one foot must exceed this

    # Step 1: assign dominant side per frame based on which prob is higher
    frame_sides = []
    for p in predictions:
        lp = p["left_prob"]
        rp = p["right_prob"]

        # Neither foot has meaningful contact probability
        if max(lp, rp) < min_contact_prob:
            frame_sides.append((p["frame"], None))
        else:
            side = "left" if lp > rp else "right"
            frame_sides.append((p["frame"], side))

    if not frame_sides:
        return []

    # Step 2: run-length encode into segments of (side, start_idx, end_idx)
    segments = []  # (side, start_frame, end_frame)
    cur_side = frame_sides[0][1]
    cur_start = frame_sides[0][0]

    for i in range(1, len(frame_sides)):
        frame, side = frame_sides[i]
        if side != cur_side:
            segments.append((cur_side, cur_start, frame_sides[i - 1][0]))
            cur_side = side
            cur_start = frame
    segments.append((cur_side, cur_start, frame_sides[-1][0]))

    # Step 3: absorb short contact segments into their longest neighbor
    changed = True
    while changed:
        changed = False
        new_segments = []
        i = 0
        while i < len(segments):
            side, start, end = segments[i]
            duration = end - start + 1

            if side is not None and duration < min_step_frames:
                # Find the longest adjacent contact neighbor
                prev_side = segments[i - 1][0] if i > 0 else None
                prev_len = (segments[i - 1][2] - segments[i - 1][1] + 1) if i > 0 else 0
                next_side = segments[i + 1][0] if i + 1 < len(segments) else None
                next_len = (segments[i + 1][2] - segments[i + 1][1] + 1) if i + 1 < len(segments) else 0

                # Merge into the longer neighbor
                if prev_len >= next_len and prev_side is not None and new_segments:
                    # Extend previous segment
                    ps, ps_start, ps_end = new_segments[-1]
                    new_segments[-1] = (ps, ps_start, end)
                    changed = True
                elif next_side is not None and i + 1 < len(segments):
                    # Extend next segment to include this one
                    ns, ns_start, ns_end = segments[i + 1]
                    segments[i + 1] = (ns, start, ns_end)
                    changed = True
                else:
                    new_segments.append((side, start, end))
            else:
                new_segments.append((side, start, end))
            i += 1
        segments = new_segments

    # Step 4: merge adjacent segments of the same side
    merged = []
    for side, start, end in segments:
        if merged and merged[-1][0] == side:
            merged[-1] = (side, merged[-1][1], end)
        else:
            merged.append((side, start, end))
    segments = merged

    # Step 5: build steps from contact segments, drop short ones and None
    steps = []
    for side, start, end in segments:
        if side is None:
            continue
        duration = end - start + 1
        if duration < min_step_frames:
            continue
        mid = start + duration // 2
        steps.append({
            "side": side,
            "start_frame": start,
            "end_frame": end,
            "touchdown_frame": mid,
            "liftoff_frame": end,
        })

    # Step 6: enforce alternation — consecutive same-side keeps the longer
    if len(steps) > 1:
        cleaned = [steps[0]]
        for s in steps[1:]:
            if s["side"] == cleaned[-1]["side"]:
                prev_len = cleaned[-1]["end_frame"] - cleaned[-1]["start_frame"]
                cur_len = s["end_frame"] - s["start_frame"]
                if cur_len > prev_len:
                    cleaned[-1] = s
            else:
                cleaned.append(s)
        steps = cleaned

    # Number them
    for i, s in enumerate(steps):
        s["step_number"] = i + 1

    return steps


# ============================================================================
# Video rendering: RAW predictions
# ============================================================================
def render_raw_video(video_path, pose_results, predictions,
                     start_frame, end_frame, output_path=None):
    """Render video with raw per-frame prediction overlays.

    Shows a colored circle at the ankle for every frame where the model
    predicts contact, with probability HUD text.
    """
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Cap output at 120 fps to avoid browser/player compatibility issues
    MAX_FPS = 120
    out_fps = min(src_fps, MAX_FPS)
    frame_skip = max(1, round(src_fps / out_fps))

    if output_path is None:
        output_path = tempfile.mktemp(suffix=".mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, out_fps, (w, h))

    pred_lookup = {p["frame"]: p for p in predictions}

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for frame_idx in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames to match target fps
        if (frame_idx - start_frame) % frame_skip != 0:
            continue

        pose_res = pose_results[frame_idx] if frame_idx < len(pose_results) else None
        pred = pred_lookup.get(frame_idx)

        # Skeleton
        if pose_res and pose_res.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            )

        # Raw prediction overlays
        if pred and pose_res and pose_res.pose_landmarks:
            lms = pose_res.pose_landmarks.landmark

            for side, color, ankle_idx in [
                ("left", CLR_LEFT_ACTIVE, L_ANKLE),
                ("right", CLR_RIGHT_ACTIVE, R_ANKLE),
            ]:
                if pred[f"{side}_contact"]:
                    ax = int(lms[ankle_idx].x * w)
                    ay = int(lms[ankle_idx].y * h)
                    prob = pred[f"{side}_prob"]

                    cv2.circle(frame, (ax, ay), 18, color, -1)
                    cv2.circle(frame, (ax, ay), 18, CLR_WHITE, 2)

                    ring_r = int(18 + 12 * prob)
                    cv2.circle(frame, (ax, ay), ring_r, color, 2)

            left_on = pred["left_contact"]
            right_on = pred["right_contact"]
            if left_on or right_on:
                parts = []
                if left_on:
                    parts.append(f"L:{pred['left_prob']:.0%}")
                if right_on:
                    parts.append(f"R:{pred['right_prob']:.0%}")
                text = "RAW  " + "  ".join(parts)
                cv2.putText(frame, text, (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # Frame counter
        cv2.putText(frame, f"F{frame_idx}", (w - 120, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        writer.write(frame)

    cap.release()
    writer.release()
    _reencode_h264(output_path)
    return output_path


# ============================================================================
# Video rendering: CLEAN steps
# ============================================================================
def render_clean_video(video_path, pose_results, steps, predictions,
                       start_frame, end_frame, output_path=None):
    """Render video with cleaned step visualization.

    For each step:
      - Touchdown: dot appears at ankle (yellow flash for 1-2 frames)
      - During contact: dot follows ankle in the foot's active color
      - Liftoff: dot freezes at last ankle position, switches to dim color
      - Dot persists at frozen position for the rest of the video

    Also draws a step counter HUD and labels each step with its number.
    """
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Cap output at 120 fps to avoid browser/player compatibility issues
    MAX_FPS = 120
    out_fps = min(src_fps, MAX_FPS)
    frame_skip = max(1, round(src_fps / out_fps))

    if output_path is None:
        output_path = tempfile.mktemp(suffix=".mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, out_fps, (w, h))

    # Track persistent dots for steps that have completed (lifted off)
    completed_dots = []

    # Build frame→step lookup
    frame_to_step = {}
    for si, step in enumerate(steps):
        for f in range(step["start_frame"], step["end_frame"] + 1):
            frame_to_step[f] = si

    # Pre-compute grounded ankle positions per step.
    # MediaPipe sometimes swaps the tracked ankle to the airborne foot for
    # some frames within a step.  We detect those outliers (high on screen =
    # low y pixel value) and replace them with the nearest grounded frame's
    # position.  This gives us:
    #   step_dot_pos   — persistent-dot placement (temporal midpoint of
    #                    grounded frames)
    #   step_frame_pos — per-frame corrected positions for active-dot snapping
    step_dot_pos = {}      # step_number -> (x, y)
    step_frame_pos = {}    # step_number -> {frame_idx: (x, y)}

    for step in steps:
        sn = step["step_number"]
        ankle_idx = L_ANKLE if step["side"] == "left" else R_ANKLE
        raw_positions = {}  # frame_idx -> (x, y)

        for f in range(step["start_frame"], step["end_frame"] + 1):
            pr = pose_results[f] if f < len(pose_results) else None
            if pr and pr.pose_landmarks:
                lm = pr.pose_landmarks.landmark[ankle_idx]
                raw_positions[f] = (int(lm.x * w), int(lm.y * h))

        if not raw_positions:
            continue

        # Identify grounded vs airborne frames.
        # In pixel coords, higher y = lower on screen = closer to ground.
        ys = np.array([p[1] for p in raw_positions.values()])
        median_y = float(np.median(ys))
        y_range = float(ys.max() - ys.min())

        if y_range > 0:
            threshold = median_y - 0.3 * y_range
            grounded_frames = {f: pos for f, pos in raw_positions.items()
                               if pos[1] >= threshold}
        else:
            grounded_frames = dict(raw_positions)

        if not grounded_frames:
            grounded_frames = dict(raw_positions)  # fallback

        # Replace outlier positions with nearest grounded frame's position
        corrected = {}
        sorted_grounded = sorted(grounded_frames.keys())
        for f in range(step["start_frame"], step["end_frame"] + 1):
            if f in grounded_frames:
                corrected[f] = grounded_frames[f]
            elif f in raw_positions:
                nearest = min(sorted_grounded, key=lambda gf: abs(gf - f))
                corrected[f] = grounded_frames[nearest]

        step_frame_pos[sn] = corrected

        # Persistent dot: position at the temporal midpoint of grounded frames
        mid_idx = len(sorted_grounded) // 2
        mid_frame = sorted_grounded[mid_idx]
        step_dot_pos[sn] = grounded_frames[mid_frame]

    # Touchdown flash duration (source frames, not output frames)
    td_flash = max(1, int(src_fps * 0.02))  # ~2 frames at 240fps, 1 at 30fps

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for frame_idx in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break

        pose_res = pose_results[frame_idx] if frame_idx < len(pose_results) else None

        lms = None
        if pose_res and pose_res.pose_landmarks:
            lms = pose_res.pose_landmarks.landmark

        # Check if a step just ended (transition from active to not-active)
        # This state tracking must run on EVERY source frame even if we skip
        # writing, so that liftoff positions and completed_dots stay correct.
        si = frame_to_step.get(frame_idx)
        prev_si = frame_to_step.get(frame_idx - 1) if frame_idx > start_frame else None

        # -- State tracking (runs every source frame) --

        if prev_si is not None and si != prev_si:
            # Previous step just ended — place persistent dot
            prev_step = steps[prev_si]
            sn = prev_step["step_number"]
            if sn in step_dot_pos:
                completed_dots.append({
                    "pos": step_dot_pos[sn],
                    "side": prev_step["side"],
                    "step_number": sn,
                })

        # If the current step ends on this frame (last frame of step or last
        # frame of the video), add its persistent dot now so it appears on
        # this frame and any remaining frames.  Without this, the last step
        # in the video never transitions and its dot is never created.
        if si is not None:
            step = steps[si]
            sn = step["step_number"]
            already_added = any(d["step_number"] == sn for d in completed_dots)
            is_last_frame_of_step = (frame_idx == step["end_frame"])
            is_last_frame_of_video = (frame_idx == end_frame)
            if not already_added and (is_last_frame_of_step or is_last_frame_of_video):
                if sn in step_dot_pos:
                    completed_dots.append({
                        "pos": step_dot_pos[sn],
                        "side": step["side"],
                        "step_number": sn,
                    })

        # -- Skip frames to match target fps --
        if (frame_idx - start_frame) % frame_skip != 0:
            continue

        # -- Rendering (only on output frames) --

        # Skeleton
        if pose_res and pose_res.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            )

        # Draw completed (persistent) dots
        for dot in completed_dots:
            x, y = dot["pos"]
            if dot["side"] == "left":
                color = CLR_LEFT_LIFTOFF
            else:
                color = CLR_RIGHT_LIFTOFF

            cv2.circle(frame, (x, y), 12, color, -1)
            cv2.circle(frame, (x, y), 12, CLR_WHITE, 1)
            cv2.putText(frame, str(dot["step_number"]),
                        (x - 6, y + 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, CLR_WHITE, 1, cv2.LINE_AA)

        # Draw active step dot (using corrected positions to avoid airborne jumps)
        if si is not None:
            step = steps[si]
            side = step["side"]
            sn = step["step_number"]
            frame_positions = step_frame_pos.get(sn, {})

            # Use corrected position if available, fall back to raw landmarks
            if frame_idx in frame_positions:
                ax, ay = frame_positions[frame_idx]
            elif lms is not None:
                ankle_idx = L_ANKLE if side == "left" else R_ANKLE
                ax = int(lms[ankle_idx].x * w)
                ay = int(lms[ankle_idx].y * h)
            else:
                ax, ay = None, None

            if ax is not None:
                frames_since_td = frame_idx - step["touchdown_frame"]

                if frames_since_td < td_flash:
                    cv2.circle(frame, (ax, ay), 24, CLR_TOUCHDOWN, -1)
                    cv2.circle(frame, (ax, ay), 30, CLR_TOUCHDOWN, 3)
                    cv2.circle(frame, (ax, ay), 24, CLR_WHITE, 2)
                else:
                    if side == "left":
                        color = CLR_LEFT_ACTIVE
                    else:
                        color = CLR_RIGHT_ACTIVE
                    cv2.circle(frame, (ax, ay), 16, color, -1)
                    cv2.circle(frame, (ax, ay), 16, CLR_WHITE, 2)

                cv2.putText(frame, str(sn),
                            (ax - 6, ay + 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, CLR_WHITE, 1, cv2.LINE_AA)

        # HUD
        total_steps = len(steps)
        completed_count = len(completed_dots)
        current_step_num = steps[si]["step_number"] if si is not None else None

        cv2.rectangle(frame, (0, 0), (350, 70), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (350, 70), (80, 80, 80), 1)

        if current_step_num is not None:
            side_label = steps[si]["side"].upper()[0]
            hud = f"Step {current_step_num}/{total_steps} ({side_label})"
            cv2.putText(frame, hud, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, "CONTACT", (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, CLR_TOUCHDOWN, 2)
        else:
            if completed_count > 0:
                hud = f"Steps detected: {total_steps}"
            else:
                hud = "Waiting for steps..."
            cv2.putText(frame, hud, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            cv2.putText(frame, "FLIGHT", (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)

        cv2.putText(frame, f"F{frame_idx}", (w - 120, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        writer.write(frame)

    cap.release()
    writer.release()
    _reencode_h264(output_path)
    return output_path


# ============================================================================
# H.264 re-encoding helper
# ============================================================================
def _reencode_h264(output_path):
    """Re-encode to H.264 for browser compatibility. Overwrites in-place."""
    h264_path = output_path.replace(".mp4", "_h264.mp4")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", output_path, "-c:v", "libx264",
             "-preset", "fast", "-crf", "23", "-c:a", "copy", h264_path],
            capture_output=True, timeout=120,
        )
        if Path(h264_path).exists() and Path(h264_path).stat().st_size > 0:
            Path(output_path).unlink(missing_ok=True)
            Path(h264_path).rename(output_path)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass


# ============================================================================
# Full inference pipeline
# ============================================================================
def run_step_inference(video_path, start_frame=None, end_frame=None,
                       progress_callback=None, model_type="auto"):
    """Full inference pipeline: pose → predict → clean → render two videos.

    Args:
        model_type: "auto" (CNN if available, else MLP), "cnn", or "mlp"

    Returns:
        dict with keys:
            raw_video_path: str - path to raw predictions video
            clean_video_path: str - path to cleaned steps video
            predictions: list[dict] - per-frame raw predictions
            steps: list[dict] - cleaned step objects
            model_name: str
            model_meta: dict
    """
    # Load model
    use_cnn = False
    use_60fps_mlp = model_type == "mlp_60fps"

    if model_type in ("auto", "cnn"):
        cnn_result = load_cnn_model()
        if cnn_result is not None:
            use_cnn = True

    if model_type == "auto" and not use_cnn and MODEL_60FPS_PATH.exists():
        use_60fps_mlp = True

    if use_cnn:
        cnn_model, cnn_checkpoint = cnn_result
        model_name = f"CNN (window={cnn_checkpoint['window_size']})"
        meta = {"model_name": model_name, "type": "cnn"}
    else:
        if model_type == "cnn":
            raise FileNotFoundError("CNN model not found. Run train_cnn.py first.")
        pipeline, meta = load_model(prefer_60fps=use_60fps_mlp)
        feature_cols = meta["feature_columns"]
        model_name = meta["model_name"]

    if progress_callback:
        progress_callback(0.05, "Model loaded")

    # Get video info
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = total_frames - 1

    # Run pose estimation
    if progress_callback:
        progress_callback(0.10, "Running pose estimation...")

    def _pose_cb(pct, msg=""):
        if progress_callback:
            progress_callback(0.10 + pct * 0.45, msg)

    pose_results = extract_pose_data(
        video_path,
        start_frame=start_frame,
        end_frame=end_frame,
        progress_callback=_pose_cb,
    )

    if progress_callback:
        progress_callback(0.58, "Predicting steps...")

    # Predict
    if use_cnn:
        predictions = predict_steps_cnn(pose_results, cnn_model, cnn_checkpoint, start_frame, end_frame, fps=src_fps)
    else:
        predictions = predict_steps(pose_results, feature_cols, pipeline, start_frame, end_frame)

    if progress_callback:
        progress_callback(0.62, "Cleaning predictions...")

    # Clean
    steps = clean_predictions(predictions, fps=src_fps)

    if progress_callback:
        progress_callback(0.65, "Rendering raw video...")

    # Render raw video
    raw_path = tempfile.mktemp(suffix=".mp4")
    render_raw_video(
        video_path, pose_results, predictions,
        start_frame, end_frame, raw_path,
    )

    if progress_callback:
        progress_callback(0.82, "Rendering clean video...")

    # Render clean video
    clean_path = tempfile.mktemp(suffix=".mp4")
    render_clean_video(
        video_path, pose_results, steps, predictions,
        start_frame, end_frame, clean_path,
    )

    if progress_callback:
        progress_callback(1.0, "Done!")

    return {
        "raw_video_path": raw_path,
        "clean_video_path": clean_path,
        "predictions": predictions,
        "steps": steps,
        "model_name": model_name,
        "model_meta": meta,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python step_detection/inference.py <video_path> [start_frame] [end_frame]")
        sys.exit(1)

    video = sys.argv[1]
    s = int(sys.argv[2]) if len(sys.argv) > 2 else None
    e = int(sys.argv[3]) if len(sys.argv) > 3 else None

    def _print_progress(pct, msg=""):
        print(f"  [{pct:5.1%}] {msg}")

    result = run_step_inference(video, s, e, progress_callback=_print_progress)
    print(f"\nRaw video:   {result['raw_video_path']}")
    print(f"Clean video: {result['clean_video_path']}")
    print(f"\nDetected {len(result['steps'])} clean steps "
          f"(using {result['model_name']}):")
    for step in result["steps"]:
        dur = step["end_frame"] - step["start_frame"] + 1
        print(f"  Step {step['step_number']:2d}: {step['side']:5s}  "
              f"F{step['start_frame']}–F{step['end_frame']} ({dur} frames)")
