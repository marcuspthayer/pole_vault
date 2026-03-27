# pv_hip_analysis.py

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

from pv_yolo_utils import PersonDetector

mp_pose = mp.solutions.pose


def compute_hip_time_series(video_path, yolo_model_path="yolo11n.pt", conf=0.25):
    """
    Run YOLO-guided MediaPipe Pose on each frame and return:
      - hip_y_arr: per-frame hip height (normalized to frame height)
      - body_h_arr: per-frame body height (normalized to frame height)
      - fps: frames per second
      - total_frames: number of frames
      - pose_landmarks_list: list of NormalizedLandmarkList (full-frame coords) or None
      - roi_box_list: list of (roi_x1, roi_y1, roi_x2, roi_y2) used for Pose, or None
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not reopen video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Pass 2/3: Running YOLO + Pose on video frames...")

    detector = PersonDetector(model_path=yolo_model_path, conf=conf)

    hip_y_list = []
    body_height_list = []
    pose_landmarks_list = []
    roi_box_list = []

    with mp_pose.Pose(
        min_detection_confidence=0.2,
        min_tracking_confidence=0.2,
        model_complexity=2,
        static_image_mode=False,
        smooth_landmarks=True,
    ) as pose2:
        frame_idx = -1
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            frame_h, frame_w = frame.shape[:2]

            # Progress for Pass 2/3
            if frame_idx % 10 == 0:
                pct = 100.0 * (frame_idx + 1) / max(total_frames, 1)
                print(
                    f"\rPass 2/3: {frame_idx + 1}/{total_frames} frames ({pct:5.1f}%)",
                    end="",
                    flush=True,
                )

            hip_y_norm = np.nan
            body_height_norm = np.nan
            landmark_list_full = None
            roi_box = None

            bbox = detector.detect_largest_person(frame)
            if bbox is not None:
                x1, y1, x2, y2 = bbox

                # Expand the box to avoid cutting off limbs
                margin = 0.3
                bw = x2 - x1
                bh = y2 - y1
                cx = x1 + bw / 2
                cy = y1 + bh / 2

                roi_w = int(bw * (1 + margin))
                roi_h = int(bh * (1 + margin))

                roi_x1 = max(0, int(cx - roi_w / 2))
                roi_y1 = max(0, int(cy - roi_h / 2))
                roi_x2 = min(frame_w, int(cx + roi_w / 2))
                roi_y2 = min(frame_h, int(cy + roi_h / 2))

                if roi_x2 > roi_x1 and roi_y2 > roi_y1:
                    roi_box = (roi_x1, roi_y1, roi_x2, roi_y2)

                    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
                    roi_h_actual, roi_w_actual = roi.shape[:2]

                    if roi_h_actual > 0 and roi_w_actual > 0:
                        image_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                        image_rgb.flags.writeable = False
                        results2 = pose2.process(image_rgb)

                        if results2.pose_landmarks:
                            lm = results2.pose_landmarks.landmark

                            def lm_full_y(landmark):
                                y_roi = landmark.y * roi_h_actual
                                y_full = roi_y1 + y_roi
                                return y_full / frame_h

                            # --- hip + height for metrics ---
                            lhip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
                            rhip = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]
                            hip_y_norm = 0.5 * (lm_full_y(lhip) + lm_full_y(rhip))

                            nose = lm[mp_pose.PoseLandmark.NOSE.value]
                            nose_y_norm = lm_full_y(nose)

                            lankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                            rankle = lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                            ankle_y_norm = max(lm_full_y(lankle), lm_full_y(rankle))

                            body_height_norm = ankle_y_norm - nose_y_norm

                            # --- full-frame landmarks for drawing (no small-skeleton rejection) ---
                            landmarks_full = []
                            for lm_i in lm:
                                x_roi = lm_i.x * roi_w_actual
                                y_roi = lm_i.y * roi_h_actual
                                x_full = roi_x1 + x_roi
                                y_full = roi_y1 + y_roi

                                x_norm_full = x_full / frame_w
                                y_norm_full = y_full / frame_h

                                landmarks_full.append(
                                    landmark_pb2.NormalizedLandmark(
                                        x=x_norm_full,
                                        y=y_norm_full,
                                        z=lm_i.z,
                                        visibility=lm_i.visibility,
                                        presence=lm_i.presence,
                                    )
                                )

                            landmark_list_full = landmark_pb2.NormalizedLandmarkList(
                                landmark=landmarks_full
                            )

            hip_y_list.append(hip_y_norm)
            body_height_list.append(body_height_norm)
            pose_landmarks_list.append(landmark_list_full)
            roi_box_list.append(roi_box)

    cap.release()
    print()  # newline after Pass 2/3 progress

    hip_y_arr = np.array(hip_y_list, dtype=float)
    body_h_arr = np.array(body_height_list, dtype=float)

    return hip_y_arr, body_h_arr, fps, total_frames, pose_landmarks_list, roi_box_list



def compute_hip_drop(hip_y_arr, body_h_arr, first_step_frame=None, last_step_frame=None):
    """
    Compute hip droop metrics using the time series and optional first/last step.

    Window:
        - If both first_step_frame and last_step_frame are provided:
            uses [first_step_frame .. last_step_frame] (inclusive) as the approach.
        - If only last_step_frame is provided: uses [0 .. last_step_frame].
        - If neither is provided: uses the full available range.

    Metric:
        - Baseline: mean hip height over the first ~20% of valid frames.
        - hip_droop_pct: mean of the worst (lowest-hip) 5% of frames
          taken from the *last 50%* of the approach window, expressed
          as % of body height relative to the early baseline.
        - hip_droop_trend_pct: average droop over the last 50% of the
          window minus the early-baseline mean (approx. overall sag).

    Returns:
        hip_droop_pct (float or None)
        hip_droop_trend_pct (float or None)
        n_valid (int): number of valid frames used
        worst_frames_global (list[int]): global frame indices of the worst-droop frames
        start_idx (int): analysis window start index (inclusive)
        end_idx (int): analysis window end index (exclusive)
    """
    n_frames = len(hip_y_arr)
    if n_frames == 0:
        return None, None, 0, [], 0, 0

    # Start index
    if first_step_frame is None:
        start_idx = 0
    else:
        start_idx = int(max(0, min(first_step_frame, n_frames - 1)))

    # End index (exclusive)
    if last_step_frame is None:
        end_idx = n_frames
    else:
        end_idx = int(min(last_step_frame + 1, n_frames))

    if end_idx <= start_idx:
        return None, None, 0, [], start_idx, end_idx

    # Slice the approach window
    hip_seg = hip_y_arr[start_idx:end_idx]
    body_seg = body_h_arr[start_idx:end_idx]
    frame_indices_seg = np.arange(start_idx, end_idx)

    # Valid frames only
    valid_mask = (~np.isnan(hip_seg)) & (~np.isnan(body_seg)) & (body_seg > 1e-4)
    if not np.any(valid_mask):
        return None, None, 0, [], start_idx, end_idx

    hip_y_valid = hip_seg[valid_mask]
    body_h_valid = body_seg[valid_mask]
    frames_valid = frame_indices_seg[valid_mask]
    n_valid = len(hip_y_valid)

    if n_valid < 3:
        return None, None, n_valid, [], start_idx, end_idx

    # --- 1) Baseline from early approach (~first 20%) ---
    k_baseline = max(3, int(0.2 * n_valid))
    k_baseline = min(k_baseline, n_valid)  # safety

    baseline_hip_y_samples = hip_y_valid[:k_baseline].copy()
    baseline_height_samples = body_h_valid[:k_baseline].copy()

    # Helper: IQR-based outlier filter
    def iqr_mask(arr):
        if arr.size < 5:
            # Too few points; don't try to filter
            return np.ones_like(arr, dtype=bool)
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        if iqr < 1e-6:
            # All basically the same; nothing to filter
            return np.ones_like(arr, dtype=bool)
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return (arr >= lower) & (arr <= upper)

    # Apply outlier removal jointly on hip_y and body_height in the baseline region
    mask_baseline = iqr_mask(baseline_hip_y_samples) & iqr_mask(baseline_height_samples)
    if np.any(mask_baseline):
        baseline_hip_y_samples = baseline_hip_y_samples[mask_baseline]
        baseline_height_samples = baseline_height_samples[mask_baseline]

    baseline_hip_y = float(np.mean(baseline_hip_y_samples))
    baseline_height = float(np.mean(baseline_height_samples))

    if baseline_height <= 1e-4:
        return None, None, n_valid, [], start_idx, end_idx

    # Hip droop time series: +% means hips are lower than baseline
    hip_drop_series = (hip_y_valid - baseline_hip_y) / baseline_height * 100.0

    # --- 2) Define the "late approach" region = last 50% of valid frames ---
    half_idx = n_valid // 2
    late_droop = hip_drop_series[half_idx:]
    late_frames = frames_valid[half_idx:]
    if late_droop.size == 0:
        # Fallback: use the whole series if for some reason half_idx == n_valid
        late_droop = hip_drop_series
        late_frames = frames_valid

    # --- 3) Find the worst (lowest-hip) 5% frames in the late approach ---
    # "Lowest hips" => largest positive droop values.
    n_late = late_droop.size
    # At least 1 frame
    n_worst = max(1, int(round(0.05 * n_late)))
    n_worst = min(n_worst, n_late)

    order = np.argsort(late_droop)[::-1]  # largest droop first
    worst_idx = order[:n_worst]
    worst_droops = late_droop[worst_idx]
    worst_frames_global = late_frames[worst_idx]

    hip_droop_pct = float(np.mean(worst_droops))

    # --- 4) Trend metric: overall sag in last 50% vs early baseline ---
    first_mean = float(np.mean(hip_drop_series[:k_baseline]))
    late_mean = float(np.mean(late_droop))
    hip_droop_trend_pct = late_mean - first_mean

    return hip_droop_pct, hip_droop_trend_pct, n_valid, list(worst_frames_global), start_idx, end_idx
