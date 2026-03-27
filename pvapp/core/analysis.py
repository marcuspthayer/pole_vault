import cv2
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

# Local import - assuming package structure pvapp.core.detector
from pvapp.core.detector import PersonDetector

mp_pose = mp.solutions.pose

def process_pose_data(pose_results, frame_shape):
    """
    Process the raw MediaPipe Pose results to extract hip height and body height time series.
    
    Args:
        pose_results: List of MediaPipe pose results (from pose_pipeline.py)
        frame_shape: (height, width) of the video frames
        
    Returns:
        hip_y_arr: per-frame hip height (normalized 0.0-1.0)
        body_h_arr: per-frame body height (normalized)
        pose_landmarks_list: list of landmark objects
        roi_box_list: Placeholder (None) for now, as we aren't using ROI tracking in the simple pipeline
    """
    hip_y_list = []
    body_height_list = []
    pose_landmarks_list = []
    roi_box_list = [] # Kept for compatibility if we re-add ROI logic later
    
    mp_pose = mp.solutions.pose
    frame_h, frame_w = frame_shape
    
    for result in pose_results:
        hip_y_norm = np.nan
        body_height_norm = np.nan
        landmark_list_full = None
        
        if result and result.pose_landmarks:
            lm = result.pose_landmarks.landmark
            landmark_list_full = result.pose_landmarks
            
            # Helper to get y from normalized landmark
            def get_y(landmark):
                return landmark.y
                
            lhip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
            rhip = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]
            
            # Hip height (average of y)
            # Note: In image coords, usually Y increases downwards. 
            # If the user wants "Height" as distance from bottom, we might invert.
            # But existing code used raw Y (0=top, 1=bottom). Analysis likely handles expected direction.
            hip_y_norm = 0.5 * (lhip.y + rhip.y)
            
            nose = lm[mp_pose.PoseLandmark.NOSE.value]
            lankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            rankle = lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            
            nose_y = nose.y
            ankle_y = max(lankle.y, rankle.y)
            
            # Body height estimate (pixels)
            body_height_norm = abs(ankle_y - nose_y)
            
        hip_y_list.append(hip_y_norm)
        body_height_list.append(body_height_norm)
        pose_landmarks_list.append(landmark_list_full)
        roi_box_list.append(None)
        
    hip_y_arr = np.array(hip_y_list, dtype=float)
    body_h_arr = np.array(body_height_list, dtype=float)
    
    return hip_y_arr, body_h_arr, pose_landmarks_list, roi_box_list



def compute_hip_drop(hip_y_arr, body_h_arr, first_step_frame=None, last_step_frame=None):
    """
    Compute hip droop metrics using the time series and optional first/last step.

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
