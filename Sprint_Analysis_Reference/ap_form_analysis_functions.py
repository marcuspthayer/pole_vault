import time

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from src.db.os.database_paths import join_paths, path_exists, remove
from src.db.sqlite.analysis_data.initialize import initialize_database
from src.utils.video.convert import convert_to_web_compatible

# Global MediaPipe components
mp_pose = mp.solutions.pose

# Landmark indices (MediaPipe Pose)
NOSE = 0
LEFT_EYE = 2
RIGHT_EYE = 5
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

# Joints we persist to position_data (must match create_empty_dataframe)
SELECTED_LANDMARK_INDICES = [
    LEFT_EYE,
    RIGHT_EYE,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_ELBOW,
    RIGHT_ELBOW,
    LEFT_WRIST,
    RIGHT_WRIST,
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_KNEE,
    RIGHT_KNEE,
    LEFT_ANKLE,
    RIGHT_ANKLE,
    LEFT_HEEL,
    RIGHT_HEEL,
    LEFT_FOOT_INDEX,
    RIGHT_FOOT_INDEX,
]

# Side pairs used for left/right consistency checks
SIDE_PAIRS = [
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_ELBOW, RIGHT_ELBOW),
    (LEFT_WRIST, RIGHT_WRIST),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_KNEE, RIGHT_KNEE),
    (LEFT_ANKLE, RIGHT_ANKLE),
    (LEFT_HEEL, RIGHT_HEEL),
    (LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX),
]

# Smoothing and robustness parameters
VISIBILITY_THRESHOLD = 0.4
FOOT_VISIBILITY_THRESHOLD = 0.6
# SMOOTHING_ALPHA = 1 means no temporal averaging (current frame only)
SMOOTHING_ALPHA = 1
# Per-frame movement clamp in normalized coordinates
# Increased from 0.2 to 0.5 to allow faster movements (reduce over-smoothing)
MAX_JOINT_DELTA = 0.5  # max normalized movement per frame
# Swap constraints: extremely conservative so sides almost never switch
# Made stricter to prevent side switching
MAX_SWAP_DISTANCE_SQ = 0.01           # non-distal pairs: require very tiny movement (reduced from 0.02)
MAX_SWAP_DISTANCE_SQ_DISTAL = 0.002   # ankles/feet: even stricter (reduced from 0.005)
SWAP_MARGIN = 1e-4
# Reduced from 0.2 to 0.1 - swapped cost must be < 10% of original cost (much stricter)
SWAP_COST_RATIO = 0.1  # swapped_cost must be < SWAP_COST_RATIO * orig_cost to allow swap
LEG_LENGTH_MAX_SCALE = 1.02  # leg segment length can't grow more than this factor vs previous
LEG_LENGTH_MAX_DIFF_RATIO = 0.05  # left/right leg lengths must stay within this relative difference

# Drawing configuration
LINE_THICKNESS = 6
OUTLINE_THICKNESS = 6  # kept for compatibility, not used with transparent drawing
JOINT_RADIUS = 8
JOINT_OUTLINE_RADIUS = 8  # kept for compatibility, not used with transparent drawing
LINE_TRANSPARENCY = 0.8   # 0 = invisible, 1 = fully opaque
JOINT_TRANSPARENCY = 0.9  # 0 = invisible, 1 = fully opaque

# Single skeleton color (white) for all lines and joints
COLOR_SKELETON = (255, 255, 255)
COLOR_JOINT_CENTER = (0, 0, 255)  # small red dot at joint centers
JOINT_CENTER_RADIUS = 0

def apform(fn, destination_folder, fps):
    conn, start = initialize_processing(fn, destination_folder)
    print('Processing', fn)

    df = create_empty_dataframe()
    cap, frame_count, fps, out, temp_output, final_output = setup_video_io(destination_folder, fn)
    t1, t2 = process_video_frames(out, df, cap, frame_count, fps)

    if df.empty:
        print("[WARNING] No pose data extracted. Check video quality or MediaPipe config.")

    out.release()
    
    # Verify temp_output exists before finalizing
    if not path_exists(temp_output):
        error_msg = f"ERROR: apform temp_output not found at {temp_output}. Video processing may have failed."
        print(error_msg)
        cap.release()
        conn.close()
        raise FileNotFoundError(error_msg)
    
    print(f"DEBUG apform: temp_output exists: {temp_output}")
    print(f"DEBUG apform: final_output will be: {final_output}")
    
    try:
        finalize_output(temp_output, final_output, df, conn)
    except Exception as e:
        error_msg = f"ERROR: apform finalize_output failed: {str(e)}"
        print(error_msg)
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")
        cap.release()
        conn.close()
        raise
    
    # Verify final_output was created
    if not path_exists(final_output):
        error_msg = f"ERROR: apform final_output not created at {final_output}. Conversion may have failed."
        print(error_msg)
        # Check if temp_output still exists (conversion might have failed)
        if path_exists(temp_output):
            print(f"DEBUG: temp_output still exists, conversion likely failed")
        cap.release()
        conn.close()
        raise FileNotFoundError(error_msg)
    
    print(f"DEBUG apform: Successfully created processed video: {final_output}")
    cap.release()

    print('Processing Time:', (time.time() - start), 'sec')
    print('Timer 1:', (t1) / 60.0, 'min')
    print('Timer 2:', (t2) / 60.0, 'min')

def initialize_processing(fn, destination_folder):
    # Strip _original from fn to ensure consistent database naming (same as max_velocity and apsprint)
    # This prevents creating multiple databases for the same video
    fn_for_db = fn.replace('_original', '') if '_original' in fn else fn
    print(f"DEBUG apform: Using fn_for_db={fn_for_db} (original fn={fn})")
    conn = initialize_database(fn_for_db, destination_folder)
    start_time = time.time()
    return conn, start_time

#
def create_empty_dataframe():
    cols = ['time', 'frame']
    joints = [
        'Leye', 'Reye', 'Lshldr', 'Rshldr', 'Lelbow', 'Relbow', 'Lhand', 'Rhand',
        'Lhip', 'Rhip', 'Lknee', 'Rknee', 'Lankle', 'Rankle', 'Lheel', 'Rheel', 'Ltoe', 'Rtoe'
    ]
    for j in joints:
        for suffix in ['x', 'y', 'z', 'v']:
            cols.append(f"{j}_{suffix}")
    return pd.DataFrame(columns=cols)


def _squared_distance_2d(a, b):
    dx = a["x"] - b["x"]
    dy = a["y"] - b["y"]
    return dx * dx + dy * dy


def _enforce_side_consistency(current_landmarks, prev_smoothed_landmarks):
    """
    Enforce left/right side consistency by detecting likely swaps and correcting them.
    Operates in-place on current_landmarks.
    """
    if prev_smoothed_landmarks is None:
        return current_landmarks

    # Precompute previous leg lengths (hip-to-ankle) for left and right, if available
    prev_left_hip = prev_smoothed_landmarks.get(LEFT_HIP)
    prev_right_hip = prev_smoothed_landmarks.get(RIGHT_HIP)
    prev_left_ankle = prev_smoothed_landmarks.get(LEFT_ANKLE)
    prev_right_ankle = prev_smoothed_landmarks.get(RIGHT_ANKLE)

    def _segment_length(a, b):
        if a is None or b is None:
            return None
        return np.sqrt(_squared_distance_2d(a, b))

    prev_left_leg_len = _segment_length(prev_left_hip, prev_left_ankle)
    prev_right_leg_len = _segment_length(prev_right_hip, prev_right_ankle)

    for left_idx, right_idx in SIDE_PAIRS:
        # Leave knees and elbows to MediaPipe's own assignment to avoid over-correcting them.
        # Feet and hips are already well-controlled by the stricter constraints above.
        if (
            (left_idx == LEFT_KNEE and right_idx == RIGHT_KNEE)
            or (left_idx == LEFT_ELBOW and right_idx == RIGHT_ELBOW)
        ):
            continue
        # Use stricter criteria for distal joints (ankles / heels / feet)
        is_distal_pair = (
            left_idx in (LEFT_ANKLE, LEFT_HEEL, LEFT_FOOT_INDEX)
            or right_idx in (RIGHT_ANKLE, RIGHT_HEEL, RIGHT_FOOT_INDEX)
        )
        visibility_threshold = FOOT_VISIBILITY_THRESHOLD if is_distal_pair else VISIBILITY_THRESHOLD
        max_swap_distance_sq = (
            MAX_SWAP_DISTANCE_SQ_DISTAL if is_distal_pair else MAX_SWAP_DISTANCE_SQ
        )

        left_cur = current_landmarks.get(left_idx)
        right_cur = current_landmarks.get(right_idx)
        left_prev = prev_smoothed_landmarks.get(left_idx)
        right_prev = prev_smoothed_landmarks.get(right_idx)

        if (
            left_cur is None
            or right_cur is None
            or left_prev is None
            or right_prev is None
        ):
            continue

        # Skip if visibility is low; we'll rely more on smoothing in that case
        if (
            left_cur["v"] < visibility_threshold
            or right_cur["v"] < visibility_threshold
            or left_prev["v"] < visibility_threshold
            or right_prev["v"] < visibility_threshold
        ):
            continue

        orig_cost = _squared_distance_2d(left_cur, left_prev) + _squared_distance_2d(
            right_cur,
            right_prev,
        )
        swapped_cost = _squared_distance_2d(left_cur, right_prev) + _squared_distance_2d(
            right_cur,
            left_prev,
        )

        # Optional: leg-length based guard that supersedes cost when evaluating ankle/foot swaps.
        # We only apply this when dealing with ankle/foot indices, where leg length is well-defined.
        if is_distal_pair and prev_left_leg_len and prev_right_leg_len:
            # Current assignment leg lengths (no swap)
            cur_left_hip = current_landmarks.get(LEFT_HIP)
            cur_right_hip = current_landmarks.get(RIGHT_HIP)
            cur_left_ankle = current_landmarks.get(LEFT_ANKLE)
            cur_right_ankle = current_landmarks.get(RIGHT_ANKLE)

            cur_left_leg_len = _segment_length(cur_left_hip, cur_left_ankle)
            cur_right_leg_len = _segment_length(cur_right_hip, cur_right_ankle)

            # Hypothetical swap of distal landmarks: left<->right ankle/heel/foot
            swapped_left_ankle = cur_right_ankle
            swapped_right_ankle = cur_left_ankle

            swapped_left_leg_len = _segment_length(cur_left_hip, swapped_left_ankle)
            swapped_right_leg_len = _segment_length(cur_right_hip, swapped_right_ankle)

            def _lengths_reasonable(left_len, right_len, prev_left, prev_right):
                if left_len is None or right_len is None:
                    return False
                # Each leg should not stretch too far relative to its own previous length
                if left_len > prev_left * LEG_LENGTH_MAX_SCALE:
                    return False
                if right_len > prev_right * LEG_LENGTH_MAX_SCALE:
                    return False
                # Legs should not diverge too much from each other
                max_len = max(left_len, right_len)
                min_len = min(left_len, right_len)
                if max_len == 0:
                    return False
                if (max_len - min_len) / max_len > LEG_LENGTH_MAX_DIFF_RATIO:
                    return False
                return True

            # If the swapped assignment produces unreasonable leg lengths, never allow swap
            if not _lengths_reasonable(
                swapped_left_leg_len,
                swapped_right_leg_len,
                prev_left_leg_len,
                prev_right_leg_len,
            ):
                continue

        # Only allow a swap when it clearly reduces the tracking error
        # and both distances are small (very conservative).
        # Made stricter: require swapped cost to be significantly lower AND within strict distance limits
        if (
            swapped_cost + SWAP_MARGIN < orig_cost * SWAP_COST_RATIO
            and swapped_cost < max_swap_distance_sq
            and orig_cost > 0.001  # Require minimum original cost to prevent swaps on near-zero movement
        ):
            # Swap current left/right landmarks for this pair
            current_landmarks[left_idx], current_landmarks[right_idx] = (
                current_landmarks[right_idx],
                current_landmarks[left_idx],
            )

    return current_landmarks


def _smooth_landmarks(current_landmarks, prev_smoothed_landmarks):
    """
    Apply temporal smoothing and velocity limiting to landmarks to reduce jitter.
    """
    if prev_smoothed_landmarks is None:
        # First frame with pose: nothing to smooth against
        return current_landmarks.copy()

    smoothed = {}
    max_delta_sq = MAX_JOINT_DELTA * MAX_JOINT_DELTA

    for idx, cur in current_landmarks.items():
        prev = prev_smoothed_landmarks.get(idx)
        if prev is None:
            smoothed[idx] = cur
            continue

        visibility = cur["v"]

        # Exponential smoothing on position (lighter smoothing, no hard freeze)
        x = SMOOTHING_ALPHA * cur["x"] + (1.0 - SMOOTHING_ALPHA) * prev["x"]
        y = SMOOTHING_ALPHA * cur["y"] + (1.0 - SMOOTHING_ALPHA) * prev["y"]
        z = SMOOTHING_ALPHA * cur["z"] + (1.0 - SMOOTHING_ALPHA) * prev["z"]

        # Clamp movement speed per frame
        dx = x - prev["x"]
        dy = y - prev["y"]
        delta_sq = dx * dx + dy * dy
        if delta_sq > max_delta_sq:
            # Scale back to max allowed movement
            scale = MAX_JOINT_DELTA / float(np.sqrt(delta_sq))
            x = prev["x"] + dx * scale
            y = prev["y"] + dy * scale

        smoothed[idx] = {
            "x": x,
            "y": y,
            "z": z,
            "v": visibility,
        }

    return smoothed


def _stylize_background(image, segmentation_mask):
    """
    Darken the background slightly while keeping the athlete bright and sharp.
    """
    if segmentation_mask is None:
        return image

    h, w = image.shape[:2]
    mask = segmentation_mask
    if mask.shape[0] != h or mask.shape[1] != w:
        mask = cv2.resize(mask, (w, h))

    # Foreground where probability is reasonably high
    condition = mask > 0.1
    condition_3 = np.stack([condition] * 3, axis=-1)

    # Darken background but keep colors
    background = (image * 0.35).astype(np.uint8)
    return np.where(condition_3, image, background)


def _draw_segments(image, landmarks, segments, color, visibility_threshold):
    h, w = image.shape[:2]

    for start_idx, end_idx in segments:
        start = landmarks.get(start_idx)
        end = landmarks.get(end_idx)
        if (
            start is None
            or end is None
            or start["v"] < visibility_threshold
            or end["v"] < visibility_threshold
        ):
            continue

        x1 = int(start["x"] * w)
        y1 = int(start["y"] * h)
        x2 = int(end["x"] * w)
        y2 = int(end["y"] * h)

        # Draw on an overlay, then alpha blend onto the original image for transparency
        overlay = image.copy()
        cv2.line(
            overlay,
            (x1, y1),
            (x2, y2),
            color,
            LINE_THICKNESS,
            cv2.LINE_AA,
        )
        alpha = max(0.0, min(1.0, float(LINE_TRANSPARENCY)))
        cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0, dst=image)


def _draw_joints(image, landmarks, joint_indices, color, visibility_threshold):
    h, w = image.shape[:2]

    for idx in joint_indices:
        lm = landmarks.get(idx)
        if lm is None or lm["v"] < visibility_threshold:
            continue

        x = int(lm["x"] * w)
        y = int(lm["y"] * h)

        # Draw joint on an overlay, then alpha blend for transparency
        overlay = image.copy()
        cv2.circle(
            overlay,
            (x, y),
            JOINT_RADIUS,
            color,
            -1,
            cv2.LINE_AA,
        )
        alpha = max(0.0, min(1.0, float(JOINT_TRANSPARENCY)))
        cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0, dst=image)

        # Draw a small red dot at the exact joint center (fully opaque for clarity)
        cv2.circle(
            image,
            (x, y),
            JOINT_CENTER_RADIUS,
            COLOR_JOINT_CENTER,
            -1,
            cv2.LINE_AA,
        )


def _draw_stylized_pose(image, landmarks):
    """
    Draw a custom, region-colored skeleton: legs, trunk, arms, and face.
    """
    if not landmarks:
        return

    # Define segments for each body region
    face_segments = [
        (NOSE, LEFT_EYE),
        (NOSE, RIGHT_EYE),
    ]
    torso_segments = [
        (LEFT_SHOULDER, RIGHT_SHOULDER),
        (LEFT_SHOULDER, LEFT_HIP),
        (RIGHT_SHOULDER, RIGHT_HIP),
        (LEFT_HIP, RIGHT_HIP),
    ]
    left_arm_segments = [
        (LEFT_SHOULDER, LEFT_ELBOW),
        (LEFT_ELBOW, LEFT_WRIST),
    ]
    right_arm_segments = [
        (RIGHT_SHOULDER, RIGHT_ELBOW),
        (RIGHT_ELBOW, RIGHT_WRIST),
    ]
    left_leg_segments = [
        (LEFT_HIP, LEFT_KNEE),
        (LEFT_KNEE, LEFT_ANKLE),
        (LEFT_ANKLE, LEFT_HEEL),
        (LEFT_HEEL, LEFT_FOOT_INDEX),
    ]
    right_leg_segments = [
        (RIGHT_HIP, RIGHT_KNEE),
        (RIGHT_KNEE, RIGHT_ANKLE),
        (RIGHT_ANKLE, RIGHT_HEEL),
        (RIGHT_HEEL, RIGHT_FOOT_INDEX),
    ]

    # Draw all segments using a single skeleton color
    _draw_segments(image, landmarks, face_segments, COLOR_SKELETON, VISIBILITY_THRESHOLD)
    _draw_segments(image, landmarks, torso_segments, COLOR_SKELETON, VISIBILITY_THRESHOLD)
    _draw_segments(image, landmarks, left_arm_segments, COLOR_SKELETON, VISIBILITY_THRESHOLD)
    _draw_segments(image, landmarks, right_arm_segments, COLOR_SKELETON, VISIBILITY_THRESHOLD)
    _draw_segments(image, landmarks, left_leg_segments, COLOR_SKELETON, VISIBILITY_THRESHOLD)
    _draw_segments(image, landmarks, right_leg_segments, COLOR_SKELETON, VISIBILITY_THRESHOLD)

    # Draw joints for each region
    face_points = [NOSE, LEFT_EYE, RIGHT_EYE]
    torso_points = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
    left_arm_points = [LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST]
    right_arm_points = [RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST]
    left_leg_points = [
        LEFT_HIP,
        LEFT_KNEE,
        LEFT_ANKLE,
        LEFT_HEEL,
        LEFT_FOOT_INDEX,
    ]
    right_leg_points = [
        RIGHT_HIP,
        RIGHT_KNEE,
        RIGHT_ANKLE,
        RIGHT_HEEL,
        RIGHT_FOOT_INDEX,
    ]

    # Draw all joints using the same skeleton color
    _draw_joints(image, landmarks, face_points, COLOR_SKELETON, VISIBILITY_THRESHOLD)
    _draw_joints(image, landmarks, torso_points, COLOR_SKELETON, VISIBILITY_THRESHOLD)
    _draw_joints(image, landmarks, left_arm_points, COLOR_SKELETON, VISIBILITY_THRESHOLD)
    _draw_joints(image, landmarks, right_arm_points, COLOR_SKELETON, VISIBILITY_THRESHOLD)
    _draw_joints(image, landmarks, left_leg_points, COLOR_SKELETON, VISIBILITY_THRESHOLD)
    _draw_joints(image, landmarks, right_leg_points, COLOR_SKELETON, VISIBILITY_THRESHOLD)

def setup_video_io(destination_folder, fn):
    import os
    import glob
    
    # CRITICAL: Reject HTML files and other non-video files immediately
    # If fn contains HTML or other non-video extensions, clean it first
    video_extensions = ['.mp4', '.mov', '.avi', '.m4v', '.MP4', '.MOV', '.AVI', '.M4V']
    fn_lower = fn.lower()
    
    # Check if fn ends with HTML or other non-video extension
    if fn_lower.endswith('.html') or fn_lower.endswith('.htm'):
        # Find the last real video extension in the filename
        last_video_ext_pos = -1
        last_video_ext_len = 0
        for ext in video_extensions:
            pos = fn_lower.rfind(ext.lower())
            if pos > last_video_ext_pos:
                last_video_ext_pos = pos
                last_video_ext_len = len(ext)
        
        if last_video_ext_pos >= 0:
            # Truncate to just the video filename
            fn = fn[:last_video_ext_pos + last_video_ext_len]
            print(f"WARNING: Cleaned HTML suffix from filename, using: {fn}")
        else:
            # No video extension found - search for actual video files
            print(f"WARNING: No video extension in {fn}, searching for actual video files...")
            all_videos = glob.glob(join_paths(destination_folder, '*.mp4')) + glob.glob(join_paths(destination_folder, '*.mov'))
            # First, check if _zoom video exists (prefer it after zoom_preprocess)
            zoom_videos = [v for v in all_videos if '_zoom' in os.path.basename(v) 
                          and '_processed' not in os.path.basename(v) and '_zoom_processed' not in os.path.basename(v)
                          and not os.path.basename(v).endswith('.html')]
            if zoom_videos:
                fn = os.path.basename(zoom_videos[0])
                print(f"Found zoom video file: {fn}")
            else:
                original_videos = [v for v in all_videos if '_processed' not in os.path.basename(v) 
                                  and '_flying_10' not in os.path.basename(v) and '_temp_flying_10' not in os.path.basename(v)
                                  and '_trimmed_flying_10' not in os.path.basename(v) and '_distance_speed' not in os.path.basename(v)
                                  and '_zoom' not in os.path.basename(v) and not os.path.basename(v).endswith('.html')]
                if original_videos:
                    fn = os.path.basename(original_videos[0])
                    print(f"Found actual video file: {fn}")
                else:
                    raise FileNotFoundError(f"No valid video file found. Rejected HTML file: {fn}")
    
    # Try to find the video file - prefer _zoom version if it exists, otherwise use original
    video_base, video_ext = os.path.splitext(fn)
    
    # Validate that the extension is actually a video extension
    if video_ext.lower() not in [e.lower() for e in video_extensions]:
        # Extension is not a video extension - search for actual video files
        print(f"WARNING: Extension {video_ext} is not a video extension, searching for actual video files...")
        all_videos = glob.glob(join_paths(destination_folder, '*.mp4')) + glob.glob(join_paths(destination_folder, '*.mov'))
        # First, check if _zoom video exists (prefer it after zoom_preprocess)
        zoom_videos = [v for v in all_videos if '_zoom' in os.path.basename(v) 
                      and '_processed' not in os.path.basename(v) and '_zoom_processed' not in os.path.basename(v)
                      and not os.path.basename(v).endswith('.html')]
        if zoom_videos:
            fn = os.path.basename(zoom_videos[0])
            video_base, video_ext = os.path.splitext(fn)
            print(f"Found zoom video file: {fn}")
        else:
            original_videos = [v for v in all_videos if '_processed' not in os.path.basename(v) 
                              and '_flying_10' not in os.path.basename(v) and '_temp_flying_10' not in os.path.basename(v)
                              and '_trimmed_flying_10' not in os.path.basename(v) and '_distance_speed' not in os.path.basename(v)
                              and '_zoom' not in os.path.basename(v) and not os.path.basename(v).endswith('.html')]
        if original_videos:
            fn = os.path.basename(original_videos[0])
            video_base, video_ext = os.path.splitext(fn)
            print(f"Found actual video file: {fn}")
        else:
            raise FileNotFoundError(f"No valid video file found. Invalid extension: {video_ext}")
    
    zoom_video = f"{video_base}_zoom{video_ext}"
    zoom_video_path = join_paths(destination_folder, zoom_video)
    original_video_path = join_paths(destination_folder, fn)
    
    # Prefer zoomed video if it exists, otherwise use original
    if path_exists(zoom_video_path):
        video_path = zoom_video_path
        print(f"Using zoomed video: {zoom_video}")
    elif path_exists(original_video_path):
        video_path = original_video_path
        print(f"Using original video: {fn}")
    else:
        # Fallback: try to find any matching video file
        video_pattern = join_paths(destination_folder, f"{video_base}*{video_ext}")
        matches = glob.glob(video_pattern)
        # Exclude processed and other generated videos, AND HTML files
        matches = [m for m in matches if '_processed' not in m and '_flying_10' not in m 
                  and '_temp_flying_10' not in m and '_trimmed_flying_10' not in m
                  and '_distance_speed' not in m and not m.endswith('.html') and not m.endswith('.htm')]
        if matches:
            video_path = matches[0]
            print(f"Using found video: {os.path.basename(video_path)}")
        else:
            video_path = original_video_path
            print(f"Warning: Video not found, trying: {fn}")
    
    # Final safeguard: reject HTML files
    if video_path.endswith('.html') or video_path.endswith('.htm'):
        raise FileNotFoundError(f"HTML file rejected: {video_path}. Please use actual video files only.")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")
    
    duration = cap.get(cv2.CAP_PROP_POS_MSEC)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f'{frame_count} frames, {duration} msec, {fps} fps')

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    temp_output = join_paths(destination_folder, 'temp_output.mp4')
    
    # Output filename: if input was _zoom, output _zoom_processed, otherwise _processed
    if '_zoom' in os.path.basename(video_path):
        final_output = join_paths(destination_folder, f"{video_base}_zoom_processed.mp4")
    else:
        final_output = join_paths(destination_folder, fn + '_processed.mp4')
    out = cv2.VideoWriter(
        temp_output,
        cv2.VideoWriter_fourcc(*'mp4v'),
        int(fps),
        (w, h)
    )

    return cap, frame_count, fps, out, temp_output, final_output

def process_video_frames(out, df, cap, frame_count, fps):
    t1 = 0.0
    t2 = 0.0
    prev_smoothed_landmarks = None

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        static_image_mode=False,
        enable_segmentation=False,  # temporarily disable segmentation
        model_complexity=2,
        smooth_landmarks=False,  # Disabled MediaPipe smoothing to reduce over-smoothing
        min_tracking_confidence=0.5
    ) as pose:
        ic = 0
        ic2 = 0
        while cap.isOpened():
            ic += 1
            if ic > frame_count:
                break

            success, image = cap.read()
            if not success:
                continue

            # Single pass: run pose on the original RGB frame (no segmentation-based refinement)
            s1 = time.time()
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            t1 += time.time() - s1

            if results.pose_landmarks is None:
                continue

            segmentation_mask = None

            # Build current landmarks dictionary from chosen MediaPipe output
            current_landmarks = {}
            for idx, lm in enumerate(results.pose_landmarks.landmark):
                current_landmarks[idx] = {
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "v": lm.visibility
                }

            # Enforce conservative side consistency so left/right don't switch easily
            current_landmarks = _enforce_side_consistency(
                current_landmarks,
                prev_smoothed_landmarks
            )

            # Temporally smooth landmarks
            smoothed_landmarks = _smooth_landmarks(
                current_landmarks,
                prev_smoothed_landmarks
            )
            prev_smoothed_landmarks = smoothed_landmarks

            ic2 += 1
            s2 = time.time()

            # Prepare image for drawing (back to BGR for OpenCV)
            image_draw = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # Background stylization is effectively disabled (segmentation_mask is None)
            image_draw = _stylize_background(image_draw, segmentation_mask)

            # Custom, professional-looking pose rendering
            _draw_stylized_pose(image_draw, smoothed_landmarks)

            out.write(image_draw)
            t2 += time.time() - s2

            # Append smoothed landmarks to df
            row = [ic / fps, ic2 - 1]
            for idx in SELECTED_LANDMARK_INDICES:
                lm = smoothed_landmarks.get(idx)
                if lm is None:
                    row.extend([np.nan, np.nan, np.nan, 0.0])
                else:
                    row.extend([lm["x"], lm["y"], lm["z"], lm["v"]])
            df.loc[ic2 - 1] = row

    return t1, t2

def finalize_output(temp_output, final_output, df, conn):
    print(f"DEBUG finalize_output: Converting {temp_output} to {final_output}")
    try:
        convert_to_web_compatible(temp_output, final_output)
        print(f"DEBUG finalize_output: Conversion successful")
    except Exception as e:
        error_msg = f"ERROR in finalize_output: convert_to_web_compatible failed: {str(e)}"
        print(error_msg)
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")
        raise
    
    # Verify final_output was created
    if not path_exists(final_output):
        error_msg = f"ERROR: final_output was not created after conversion: {final_output}"
        print(error_msg)
        raise FileNotFoundError(error_msg)
    
    if path_exists(temp_output):
        remove(temp_output)
        print(f"DEBUG finalize_output: Removed temp_output")
    
    # Clear existing position_data before inserting new data (for reprocessing)
    # This ensures we don't mix data from different videos
    conn.execute('DELETE FROM position_data')
    conn.commit()
    df.to_sql('position_data', conn, if_exists='append', index=False)
    print(f"DEBUG finalize_output: Saved {len(df)} rows to position_data")
