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
def load_model():
    """Load the saved best model and its metadata."""
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
# Step cleaning / grouping
# ============================================================================
def clean_predictions(predictions, fps=240,
                      min_step_seconds=0.06, max_gap_seconds=0.025):
    """Convert noisy per-frame predictions into clean, discrete steps.

    Thresholds are specified in seconds and converted to frames using the
    video's FPS, so the same logic works across 30fps–240fps+ footage.

    Algorithm:
      1. Assign each frame a dominant side (left/right/none) based on
         which foot has higher contact probability.
      2. Smooth short side-flips: RLE the per-frame sides, then replace
         any contact segment shorter than *min_step_frames* with the side
         of its longest adjacent neighbour (eliminates brief L↔R noise).
      3. Run-length encode the smoothed sides.
      4. Bridge short gaps (<=max_gap_frames) between same-side segments.
      5. Absorb short opposite-side segments (<= min_step_frames) into
         their neighbours.
      6. Drop remaining short segments.
      7. Enforce alternation — consecutive same-side steps keep the longer.
      8. Number and return.

    Returns:
        list of dicts, each with:
            side, start_frame, end_frame, touchdown_frame, liftoff_frame,
            step_number
    """
    if not predictions:
        return []

    min_step_frames = max(2, round(fps * min_step_seconds))
    max_gap_frames = max(1, round(fps * max_gap_seconds))

    # Step 1: assign dominant side per frame
    frame_sides = []  # list of (frame_idx, side_or_none)
    for p in predictions:
        l = p["left_contact"]
        r = p["right_contact"]
        lp = p["left_prob"]
        rp = p["right_prob"]

        if l and r:
            side = "left" if lp >= rp else "right"
        elif l:
            side = "left"
        elif r:
            side = "right"
        else:
            side = None

        frame_sides.append((p["frame"], side))

    if not frame_sides:
        return []

    # Step 2: smooth short side-flips before main RLE.
    # Quick RLE pass to find segments, then replace short contact segments
    # with the side of their longest adjacent contact neighbour.
    def _rle(sides_list):
        """Run-length encode (side, start_idx, end_idx) over a list."""
        segs = []
        cur = sides_list[0][1]
        start = 0
        for i in range(1, len(sides_list)):
            if sides_list[i][1] != cur:
                segs.append((cur, start, i - 1))
                cur = sides_list[i][1]
                start = i
        segs.append((cur, start, len(sides_list) - 1))
        return segs

    pre_segs = _rle(frame_sides)

    # For each short contact segment, replace with the dominant neighbour
    changed = True
    while changed:
        changed = False
        for seg_i, (seg_side, seg_start, seg_end) in enumerate(pre_segs):
            if seg_side is None:
                continue
            seg_len = seg_end - seg_start + 1
            if seg_len >= min_step_frames:
                continue

            # Find adjacent contact neighbours
            prev_side, prev_len = None, 0
            for j in range(seg_i - 1, -1, -1):
                if pre_segs[j][0] is not None:
                    prev_side = pre_segs[j][0]
                    prev_len = pre_segs[j][2] - pre_segs[j][1] + 1
                    break

            next_side, next_len = None, 0
            for j in range(seg_i + 1, len(pre_segs)):
                if pre_segs[j][0] is not None:
                    next_side = pre_segs[j][0]
                    next_len = pre_segs[j][2] - pre_segs[j][1] + 1
                    break

            # Pick the longer neighbour's side
            if prev_len >= next_len and prev_side is not None:
                replace_side = prev_side
            elif next_side is not None:
                replace_side = next_side
            else:
                continue

            if replace_side != seg_side:
                for k in range(seg_start, seg_end + 1):
                    frame_sides[k] = (frame_sides[k][0], replace_side)
                changed = True

        if changed:
            pre_segs = _rle(frame_sides)

    # Step 3: run-length encode (on the smoothed sides, using frame indices)
    segments = []  # list of (side, start_frame, end_frame)
    cur_side = frame_sides[0][1]
    cur_start = frame_sides[0][0]

    for i in range(1, len(frame_sides)):
        frame, side = frame_sides[i]
        if side != cur_side:
            segments.append((cur_side, cur_start, frame_sides[i - 1][0]))
            cur_side = side
            cur_start = frame
    segments.append((cur_side, cur_start, frame_sides[-1][0]))

    # Step 4: bridge short gaps between same-side segments
    merged = True
    while merged:
        merged = False
        new_segments = []
        i = 0
        while i < len(segments):
            if i + 2 < len(segments):
                s1_side, s1_start, s1_end = segments[i]
                gap_side, gap_start, gap_end = segments[i + 1]
                s2_side, s2_start, s2_end = segments[i + 2]

                gap_len = gap_end - gap_start + 1

                # Bridge None gaps between same-side contact segments
                if (s1_side is not None and s1_side == s2_side
                        and gap_side is None and gap_len <= max_gap_frames):
                    new_segments.append((s1_side, s1_start, s2_end))
                    i += 3
                    merged = True
                    continue

                # Absorb short opposite-side blips between same-side segments
                if (s1_side is not None and s1_side == s2_side
                        and gap_side is not None and gap_side != s1_side
                        and gap_len <= min_step_frames):
                    new_segments.append((s1_side, s1_start, s2_end))
                    i += 3
                    merged = True
                    continue

            new_segments.append(segments[i])
            i += 1

        segments = new_segments

    # Step 5: drop short contact segments and None segments
    steps = []
    for side, start, end in segments:
        if side is None:
            continue
        duration = end - start + 1
        if duration < min_step_frames:
            continue
        steps.append({
            "side": side,
            "start_frame": start,
            "end_frame": end,
            "touchdown_frame": start,
            "liftoff_frame": end,
        })

    # Step 6: enforce alternation — if consecutive steps are same side,
    # keep the longer one
    if len(steps) > 1:
        cleaned = [steps[0]]
        for s in steps[1:]:
            if s["side"] == cleaned[-1]["side"]:
                # Keep whichever is longer
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
                       progress_callback=None):
    """Full inference pipeline: pose → predict → clean → render two videos.

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
    pipeline, meta = load_model()
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
