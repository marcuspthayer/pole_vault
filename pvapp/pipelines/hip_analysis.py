import os
import cv2
import numpy as np
import mediapipe as mp

from pvapp.core.analysis import compute_hip_time_series, compute_hip_drop
from pvapp.utils.cv_utils import draw_outlined_text, draw_simple_skeleton

mp_pose = mp.solutions.pose

def run_hip_analysis_pipeline(video_path, 
                              output_path=None, 
                              yolo_model_path="yolo11n.pt", 
                              conf=0.25, 
                              start_frame=None, 
                              plant_frame=None,
                              end_frame=None,
                              progress_callback=None):
    """
    Full pipeline:
    1. Compute Hip Time Series (YOLO + Pose).
    2. Compute Metrics (Hip Drop) based on start_frame -> plant_frame.
    3. Render output video with skeleton and analysis overlay (start_frame -> end_frame).
    
    Args:
        progress_callback: function(float, str) -> None used to report progress percentage and status message.
    """
    if output_path is None:
        # Default output name
        base, ext = os.path.splitext(video_path)
        output_path = f"{base}_hip_analysis{ext}"

    # --- 1. Compute Time Series ---
    if progress_callback:
        progress_callback(0.0, "Running Pose Analysis...")

    def _ts_prog(pct):
        # Time series is roughly 40% of the work? Or 80? Let's say 50% for now.
        if progress_callback:
            progress_callback(pct * 0.5, f"Running Pose Analysis: {int(pct*100)}%")

    hip_y_arr, body_h_arr, fps, total_frames, pose_landmarks_list, roi_box_list = \
        compute_hip_time_series(video_path, yolo_model_path=yolo_model_path, conf=conf, progress_callback=_ts_prog)

    # --- 2. Compute Metrics ---
    if progress_callback:
        progress_callback(0.5, "Computing Metrics...")
        
    (
        hip_droop_pct,
        hip_droop_trend_pct,
        n_valid,
        worst_droop_frames,
        analysis_start_idx,
        analysis_end_idx,
    ) = compute_hip_drop(
        hip_y_arr,
        body_h_arr,
        first_step_frame=start_frame,
        last_step_frame=plant_frame,
    )

    # --- 3. Render Output ---
    if progress_callback:
        progress_callback(0.55, "Rendering Video...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video for reading: {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps_video, (width, height))
    if not out.isOpened():
        raise IOError(f"Could not open video for writing: {output_path}")

    # Text lines to show
    text_lines = []
    if hip_droop_pct is not None:
        text_lines.append(f"Hip droop (worst 5%): {hip_droop_pct:+.1f}% of height")
        text_lines.append(f"Hip trend (sag): {hip_droop_trend_pct:+.1f}% of height")
    else:
        text_lines.append("Hip droop: N/A (Insuff. data)")
    
    if start_frame is not None or end_frame is not None:
        s = start_frame if start_frame is not None else 0
        p = plant_frame if plant_frame is not None else total_frames
        e = end_frame if end_frame is not None else total_frames
        text_lines.append(f"Metrics: {s}-{p}")
        text_lines.append(f"Visuals: {s}-{e}")
    else:
        text_lines.append("Window: Full Video")
        s = 0
        p = total_frames
        e = total_frames

    # Persistent hip path
    hip_points_normal = []
    hip_points_worst = []
    worst_frame_set = set(worst_droop_frames if worst_droop_frames else [])

    frame_idx = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        
        # Progress for rendering
        if progress_callback and frame_idx % 10 == 0:
            # Map 0..total -> 0.55..1.0
            p = 0.55 + 0.45 * (frame_idx / max(total_frames, 1))
            progress_callback(p, f"Rendering: {int(p*100)}%")

        # ROI & Skeleton
        roi_box = None
        if 0 <= frame_idx < len(roi_box_list):
            roi_box = roi_box_list[frame_idx]
        
        landmark_list_full = None
        if 0 <= frame_idx < len(pose_landmarks_list):
            landmark_list_full = pose_landmarks_list[frame_idx]

        if roi_box:
            cv2.rectangle(frame, (roi_box[0], roi_box[1]), (roi_box[2], roi_box[3]), (0, 255, 255), 2)
        
        hip_pt = None
        # Check visual window
        in_visual_window = (frame_idx >= s and frame_idx <= e)

        if landmark_list_full and in_visual_window:
            # Check consistency (simple version)
            # Just draw it for now, can implement the "too big/small" check if needed.
            # But the user asked for modular. I will assume the landmark list is good enough if it exists.
            
            # Draw skeleton
            draw_simple_skeleton(frame, landmark_list_full, mp_pose.POSE_CONNECTIONS, 
                                 point_color=(252, 0, 219), line_color=(0, 255, 0))

            # Hip point for path
            lm = landmark_list_full.landmark
            lhip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
            rhip = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]
            hx = int((lhip.x + rhip.x) * 0.5 * width)
            hy = int((lhip.y + rhip.y) * 0.5 * height)
            hip_pt = (hx, hy)

        # Update path if inside analysis window
        in_window = (frame_idx >= analysis_start_idx and frame_idx < analysis_end_idx)
        if in_window and hip_pt:
            if frame_idx in worst_frame_set:
                hip_points_worst.append(hip_pt)
            else:
                hip_points_normal.append(hip_pt)

        # Draw path
        for pt in hip_points_normal:
            cv2.circle(frame, pt, 5, (255, 255, 0), -1)
        for pt in hip_points_worst:
            cv2.circle(frame, pt, 6, (0, 0, 255), -1)

        # Draw text
        y0 = 60
        for i, txt in enumerate(text_lines):
            draw_outlined_text(frame, txt, (10, y0 + 40*i), cv2.FONT_HERSHEY_SIMPLEX, 
                               1.0, (0,255,0), 2)

        out.write(frame)

    cap.release()
    out.release()
    
    if progress_callback:
        progress_callback(1.0, "Done!")
        
    return output_path
