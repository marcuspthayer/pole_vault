# ap_pv_approach.py

import os
import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk

from pv_step_labeler import choose_video_file, label_steps
from pv_hip_analysis import (
    compute_hip_time_series,
    compute_hip_drop,
    roi_pose_landmarks_full_frame,
)
from pv_yolo_utils import PersonDetector

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def draw_simple_skeleton(frame, landmark_list_full, connections, 
                         point_color=(0, 0, 255), line_color=(0, 255, 0),
                         point_radius=3, line_thickness=2,
                         visibility_threshold=0.1):
    """
    Draw a simple skeleton using OpenCV directly, based on a NormalizedLandmarkList
    and a list of (start_idx, end_idx) connections (e.g. mp_pose.POSE_CONNECTIONS).
    """
    frame_h, frame_w = frame.shape[:2]
    lm = landmark_list_full.landmark

    # Draw lines between connected joints
    for start_idx, end_idx in connections:
        if start_idx >= len(lm) or end_idx >= len(lm):
            continue
        lms = lm[start_idx]
        lme = lm[end_idx]

        # Optionally skip low-visibility landmarks
        if (hasattr(lms, "visibility") and lms.visibility < visibility_threshold) or \
           (hasattr(lme, "visibility") and lme.visibility < visibility_threshold):
            continue

        x1 = int(lms.x * frame_w)
        y1 = int(lms.y * frame_h)
        x2 = int(lme.x * frame_w)
        y2 = int(lme.y * frame_h)

        if 0 <= x1 < frame_w and 0 <= y1 < frame_h and 0 <= x2 < frame_w and 0 <= y2 < frame_h:
            cv2.line(frame, (x1, y1), (x2, y2), line_color, line_thickness)

    # Draw points at each joint
    for lmp in lm:
        px = int(lmp.x * frame_w)
        py = int(lmp.y * frame_h)
        if 0 <= px < frame_w and 0 <= py < frame_h:
            cv2.circle(frame, (px, py), point_radius, point_color, -1)



def analyze_pole_vault_approach(video_path, enable_step_labeling=True):
    destination_folder = os.path.dirname(video_path)
    fn = os.path.basename(video_path)

    # --- STEP 1: optional step labeling ---
    step_frames = []
    last_step_frame = None
    fps = None

    if enable_step_labeling:
        step_frames, last_step_frame, fps = label_steps(video_path)
        if step_frames is None and last_step_frame is None:
            print("Step labeling aborted. Exiting.")
            return None
        print(f"Using last_step_frame={last_step_frame} for approach window.")
    else:
        # No labeling; we still need FPS for reporting
        cap_tmp = cv2.VideoCapture(video_path)
        if not cap_tmp.isOpened():
            raise IOError(f"Could not open video file: {video_path}")
        fps = cap_tmp.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_tmp.release()
        last_step_frame = None  # interpret as: use whole clip

    # --- STEP 2: hip time series + metrics ---
    hip_y_arr, body_h_arr, fps_from_pose, total_frames, pose_landmarks_list = \
    compute_hip_time_series(video_path)

    # If no fps from labeler, use pose's fps
    if fps is None:
        fps = fps_from_pose

    hip_droop_pct, hip_droop_trend_pct, n_valid = compute_hip_drop(
        hip_y_arr, body_h_arr, last_step_frame=last_step_frame
    )

    # --- STEP 3: stride metrics from step_frames ---
    stride_rate = None
    cadence = None

    if step_frames and len(step_frames) >= 2:
        step_times = np.array(step_frames, dtype=float) / fps
        step_intervals = np.diff(step_times)
        valid = step_intervals > 0
        if np.any(valid):
            mean_interval = np.mean(step_intervals[valid])
            stride_rate = 1.0 / mean_interval
            cadence = stride_rate * 60.0

    # --- Print metrics summary ---
    print("\n=== Approach Metrics ===")
    if last_step_frame is not None:
        print(
            f"Approach window: frames 0..{last_step_frame} "
            f"(t_end = {last_step_frame / fps:.3f} s)"
        )
    else:
        print(f"Approach window: full clip (0..{total_frames - 1})")

    if stride_rate is not None:
        print(f"Stride rate (steps/s): {stride_rate:.3f}")
        print(f"Cadence (steps/min): {cadence:.1f}")
    else:
        print("Stride rate: not enough steps labeled to compute (or labeling skipped).")

    if hip_droop_pct is not None:
        print(
            f"Hip droop near takeoff: {hip_droop_pct:+.2f}% of body height "
            f"(positive = hip lower than early approach)."
        )
        print(
            f"Hip droop trend (last vs first 20%): "
            f"{hip_droop_trend_pct:+.2f}% of body height."
        )
        print(f"Valid frames used for hip analysis: {n_valid}")
    else:
        print("Hip droop: insufficient pose data to compute.")

    # --- STEP 4: render processed video with YOLO+stored skeleton ---
    cap3 = cv2.VideoCapture(video_path)
    if not cap3.isOpened():
        raise IOError(f"Could not reopen video file for writing: {video_path}")

    frame_width = int(cap3.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap3.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap3.get(cv2.CAP_PROP_FPS)
    total_frames3 = int(cap3.get(cv2.CAP_PROP_FRAME_COUNT))

    final_output = os.path.join(
        destination_folder,
        os.path.splitext(fn)[0] + "_pv_processed.mp4",
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(final_output, fourcc, fps_video, (frame_width, frame_height))
    if not out.isOpened():
        raise IOError("Could not open VideoWriter for output video.")

    print("Pass 3/3: Writing processed output video...")

    text_lines = []
    if stride_rate is not None:
        text_lines.append(f"Stride rate: {stride_rate:.2f} steps/s ({cadence:.0f} spm)")
    else:
        text_lines.append("Stride rate: N/A")

    if hip_droop_pct is not None:
        text_lines.append(f"Hip droop: {hip_droop_pct:+.1f}% of height")
    else:
        text_lines.append("Hip droop: N/A")

    if last_step_frame is not None:
        text_lines.append("Analysis: up to last labeled step.")
    else:
        text_lines.append("Analysis: entire clip (no step labeling).")

    step_frames_set = set(step_frames)
    detector_draw = PersonDetector(model_path="yolo11n.pt", conf=0.25)

    frame_idx3 = -1
    while True:
        ret, frame = cap3.read()
        if not ret:
            break

        frame_idx3 += 1
        frame_h, frame_w = frame.shape[:2]

        # --- skeleton: use stored landmarks from metrics pass ---
        landmark_list_full = None
        if 0 <= frame_idx3 < len(pose_landmarks_list):
            landmark_list_full = pose_landmarks_list[frame_idx3]

        if landmark_list_full is not None:
            draw_simple_skeleton(
                frame,
                landmark_list_full,
                mp_pose.POSE_CONNECTIONS,
                point_color=(0, 0, 255),
                line_color=(0, 255, 0),
                point_radius=3,
                line_thickness=2,
                visibility_threshold=0.1,
            )

        # --- ROI box: YOLO + margin, just for visualization ---
        roi_box = None
        bbox = detector_draw.detect_largest_person(frame)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
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
            roi_box = (roi_x1, roi_y1, roi_x2, roi_y2)

        if roi_box is not None:
            x1, y1, x2, y2 = roi_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Overlay metrics text
        y0 = 30
        for i, txt in enumerate(text_lines):
            cv2.putText(
                frame,
                txt,
                (10, y0 + 25 * i),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        # Mark step frames
        if frame_idx3 in step_frames_set:
            cv2.putText(
                frame,
                "STEP",
                (frame_width - 120, frame_height - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        # Jump-phase annotation
        if last_step_frame is not None and frame_idx3 > last_step_frame:
            cv2.putText(
                frame,
                "Jump phase (no hip/stride analysis)",
                (10, frame_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

        out.write(frame)

        # --- progress every 30 frames ---
        if frame_idx3 % 30 == 0:
            pct = 100.0 * (frame_idx3 + 1) / max(total_frames3, 1)
            print(f"\rPass 3/3: {frame_idx3 + 1}/{total_frames3} frames ({pct:5.1f}%)", end="")

    print()  # newline after loop

    cap3.release()
    out.release()

    print(f"\nProcessed video saved to: {final_output}")
    return final_output




if __name__ == "__main__":
    video_path = choose_video_file()
    if not video_path:
        print("No video selected. Exiting.")
    else:
        try:
            ans = input(
                "Do you want to label steps (for stride rate / last-step)? [y/N]: "
            ).strip().lower()
            enable_steps = ans in ("y", "yes")
            analyze_pole_vault_approach(video_path, enable_step_labeling=enable_steps)
        except Exception as err:
            print(f"Error during analysis: {err}")
