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
      - hip_y_arr, body_h_arr: per-frame hip height and body height
        (both normalized to frame height)
      - fps, total_frames
      - pose_landmarks_list: list of NormalizedLandmarkList (full-frame coords)
        or None per frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not reopen video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    detector = PersonDetector(model_path=yolo_model_path, conf=conf)
    hip_y_list = []
    body_height_list = []
    pose_landmarks_list = []

    print("\nPass 2/3: Extracting hip metrics with YOLO+Pose...")

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
            bbox = detector.detect_largest_person(frame)

            landmark_list_full = None  # default for this frame
            if bbox is None:
                hip_y_norm = np.nan
                body_height_norm = np.nan
            else:
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

                roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
                roi_h_actual, roi_w_actual = roi.shape[:2]

                if roi_h_actual <= 0 or roi_w_actual <= 0:
                    # Degenerate ROI
                    hip_y_norm = np.nan
                    body_height_norm = np.nan
                else:
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

                        # --- full-frame landmarks for drawing ---
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
                    else:
                        hip_y_norm = np.nan
                        body_height_norm = np.nan

            hip_y_list.append(hip_y_norm)
            body_height_list.append(body_height_norm)
            pose_landmarks_list.append(landmark_list_full)
            # --- progress indicator every 30 frames ---
            if frame_idx % 30 == 0:
                pct = 100.0 * (frame_idx + 1) / max(total_frames, 1)
                print(f"\rPass 2/3: {frame_idx + 1}/{total_frames} frames ({pct:5.1f}%)", end="")

    print()  # newline after loop
    cap.release()

    hip_y_arr = np.array(hip_y_list, dtype=float)
    body_h_arr = np.array(body_height_list, dtype=float)

    return hip_y_arr, body_h_arr, fps, total_frames, pose_landmarks_list


def compute_hip_drop(hip_y_arr, body_h_arr, last_step_frame=None):
    """
    Compute hip droop metrics using the time series and optional last_step_frame.

    If last_step_frame is provided:
        - Uses data from frames [0..last_step_frame] as the 'approach region'.
    If last_step_frame is None:
        - Uses the full available range.
    """
    n_frames = len(hip_y_arr)
    if n_frames == 0:
        return None, None, 0

    if last_step_frame is None:
        end_idx = n_frames  # full clip
    else:
        end_idx = min(last_step_frame + 1, n_frames)

    hip_y = hip_y_arr[:end_idx]
    body_h = body_h_arr[:end_idx]

    valid_mask = (~np.isnan(hip_y)) & (~np.isnan(body_h)) & (body_h > 1e-4)
    if not np.any(valid_mask):
        return None, None, 0

    hip_y_valid = hip_y[valid_mask]
    body_h_valid = body_h[valid_mask]
    n_valid = len(hip_y_valid)

    k = max(3, int(0.2 * n_valid))  # ~first/last 20%

    baseline_hip_y = float(np.mean(hip_y_valid[:k]))
    baseline_height = float(np.mean(body_h_valid[:k]))

    hip_drop_series = (hip_y_valid - baseline_hip_y) / baseline_height * 100.0

    hip_droop_pct = float(np.mean(hip_drop_series[-k:]))
    first_mean = float(np.mean(hip_drop_series[:k]))
    last_mean = float(np.mean(hip_drop_series[-k:]))
    hip_droop_trend_pct = last_mean - first_mean

    return hip_droop_pct, hip_droop_trend_pct, n_valid


def roi_pose_landmarks_full_frame(frame, pose, detector, margin=0.3):
    """
    (You may not be using this anymore for rendering, but keeping it here
    as a reusable helper if needed.)
    """
    frame_h, frame_w = frame.shape[:2]
    bbox = detector.detect_largest_person(frame)
    if bbox is None:
        return None, None

    x1, y1, x2, y2 = bbox

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

    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_h_actual, roi_w_actual = roi.shape[:2]

    if roi_h_actual <= 0 or roi_w_actual <= 0:
        return None, (roi_x1, roi_y1, roi_x2, roi_y2)

    image_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        return None, (roi_x1, roi_y1, roi_x2, roi_y2)

    lm_src = results.pose_landmarks.landmark
    landmarks_full = []

    for lm in lm_src:
        x_roi = lm.x * roi_w_actual
        y_roi = lm.y * roi_h_actual
        x_full = roi_x1 + x_roi
        y_full = roi_y1 + y_roi

        x_norm_full = x_full / frame_w
        y_norm_full = y_full / frame_h

        landmarks_full.append(
            landmark_pb2.NormalizedLandmark(
                x=x_norm_full,
                y=y_norm_full,
                z=lm.z,
                visibility=lm.visibility,
                presence=lm.presence,
            )
        )

    landmark_list_full = landmark_pb2.NormalizedLandmarkList(
        landmark=landmarks_full
    )

    return landmark_list_full, (roi_x1, roi_y1, roi_x2, roi_y2)
