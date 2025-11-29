# ap_pv_approach.py

import os
import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk

from pv_step_labeler import choose_video_file, label_steps
from pv_hip_analysis import compute_hip_time_series, compute_hip_drop

from pv_yolo_utils import draw_outlined_text

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



def analyze_pole_vault_approach(video_path, enable_step_labeling=True, crop_to_first_step=False):
    destination_folder = os.path.dirname(video_path)
    fn = os.path.basename(video_path)

    # --- STEP 1: optional step labeling ---
    step_frames = []
    last_step_frame = None
    fps = None
    first_step_frame = None

    if enable_step_labeling:
        print("Pass 1/3: Manual step labeling...")
        step_frames, last_step_frame, fps = label_steps(video_path)
        if step_frames is None and last_step_frame is None:
            print("Step labeling aborted. Exiting.")
            return None
        if step_frames:
            first_step_frame = step_frames[0]
        print(f"Using first_step_frame={first_step_frame}, last_step_frame={last_step_frame} for approach window.")
    else:
        # No labeling; we still need FPS for reporting
        cap_tmp = cv2.VideoCapture(video_path)
        if not cap_tmp.isOpened():
            raise IOError(f"Could not open video file: {video_path}")
        fps = cap_tmp.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_tmp.release()
        last_step_frame = None  # interpret as: use whole clip
        first_step_frame = None

    # --- STEP 2: hip time series + metrics ---
    hip_y_arr, body_h_arr, fps_from_pose, total_frames, pose_landmarks_list, roi_box_list = \
    compute_hip_time_series(video_path)

    # If no fps from labeler, use pose's fps
    if fps is None:
        fps = fps_from_pose

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
        first_step_frame=first_step_frame,
        last_step_frame=last_step_frame,
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
        start_idx = analysis_start_idx if "analysis_start_idx" in locals() else (
            first_step_frame if first_step_frame is not None else 0
        )
        print(
            f"Approach window: frames {start_idx}..{last_step_frame} "
            f"(t_start = {start_idx / fps:.3f} s, "
            f"t_end = {last_step_frame / fps:.3f} s)"
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
            f"Hip droop worst-case (lowest 5% in last 50%): "
            f"{hip_droop_pct:+.2f}% of body height "
            f"(positive = hip lower than early approach)."
        )
        print(
            f"Hip droop trend (last 50% vs early baseline): "
            f"{hip_droop_trend_pct:+.2f}% of body height."
        )
        print(f"Valid frames used for hip analysis: {n_valid}")
        print(f"Worst-droop frames (indices): {worst_droop_frames}")
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

    # For 30 fps, we want STEP to show on 3 frames total (step-1, step, step+1).
    # Scale this window with FPS: half-window ~ fps/30.
    if fps_video > 0:
        step_half_window = max(1, int(round(fps_video / 30.0)))
    else:
        step_half_window = 1

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
        text_lines.append(f"Hip droop (worst 5%): {hip_droop_pct:+.1f}% of height")
    else:
        text_lines.append("Hip droop: N/A")

    if last_step_frame is not None:
        text_lines.append("Analysis: up to last labeled step.")
    else:
        text_lines.append("Analysis: entire clip (no step labeling).")

    step_frames_set = set(step_frames)
    worst_frame_set = set(worst_droop_frames if hip_droop_pct is not None else [])
    
    # Precompute all frames where "STEP" should be visible.
    # For each labeled step frame sf, show STEP on frames
    # [sf - step_half_window, ..., sf + step_half_window].
    step_highlight_frames = set()
    if step_frames:
        for sf in step_frames:
            for fidx in range(sf - step_half_window, sf + step_half_window + 1):
                if 0 <= fidx < total_frames3:
                    step_highlight_frames.add(fidx)
                    
    # For drawing a persistent hip path
    hip_points_normal = []  # (x, y) for all analysis-window frames
    hip_points_worst = []   # (x, y) for worst-droop frames

    frame_idx3 = -1

    while True:
        ret, frame = cap3.read()
        if not ret:
            break

        frame_idx3 += 1
        frame_h, frame_w = frame.shape[:2]

        # If requested, skip all frames before the first labeled step
        if crop_to_first_step and first_step_frame is not None and frame_idx3 < first_step_frame:
            continue

        # --- ROI box: reuse the one from hip analysis (Pass 2) ---
        roi_box = None
        if 0 <= frame_idx3 < len(roi_box_list):
            roi_box = roi_box_list[frame_idx3]

        # --- skeleton candidate from stored landmarks ---
        landmark_list_full = None
        if roi_box is not None and 0 <= frame_idx3 < len(pose_landmarks_list):
            landmark_list_full = pose_landmarks_list[frame_idx3]

        # Draw the yellow box from the stored ROI
        if roi_box is not None:
            x1, y1, x2, y2 = roi_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Final sanity check before drawing skeleton:
        #   - If the skeleton's bounding box is way larger than the YOLO ROI
        #     or far from its center, OR if any substantial number of joints
        #     fall outside the ROI, we skip drawing it (and skip hip dot).
        drew_skeleton_this_frame = False
        hip_pt_this_frame = None

        if roi_box is not None and landmark_list_full is not None:
            xs_pix = []
            ys_pix = []

            for lm in landmark_list_full.landmark:
                xs_pix.append(lm.x * frame_w)
                ys_pix.append(lm.y * frame_h)

            skel_min_x = min(xs_pix)
            skel_max_x = max(xs_pix)
            skel_min_y = min(ys_pix)
            skel_max_y = max(ys_pix)
            skel_w = skel_max_x - skel_min_x
            skel_h = skel_max_y - skel_min_y
            skel_cx = 0.5 * (skel_min_x + skel_max_x)
            skel_cy = 0.5 * (skel_min_y + skel_max_y)

            roi_x1, roi_y1, roi_x2, roi_y2 = roi_box
            roi_w = roi_x2 - roi_x1
            roi_h = roi_y2 - roi_y1
            roi_cx = 0.5 * (roi_x1 + roi_x2)
            roi_cy = 0.5 * (roi_y1 + roi_y2)

            # Heuristics: size and center
            too_big = (skel_w > 1.8 * roi_w) or (skel_h > 1.8 * roi_h)
            too_small = (skel_h < 0.25 * roi_h)  # allow fairly small but reject tiny blobs
            far_from_center = (
                abs(skel_cx - roi_cx) > 0.6 * roi_w or
                abs(skel_cy - roi_cy) > 0.6 * roi_h
            )

            # Per-joint inside-ROI check, but with a loose threshold
            outside_points = 0
            tol = 2.0  # small tolerance in pixels
            for x, y in zip(xs_pix, ys_pix):
                if (x < roi_x1 - tol) or (x > roi_x2 + tol) or (y < roi_y1 - tol) or (y > roi_y2 + tol):
                    outside_points += 1
            frac_outside = outside_points / max(len(xs_pix), 1)

            # Always compute hip point when we have landmarks
            lm_list = landmark_list_full.landmark
            lhip = lm_list[mp_pose.PoseLandmark.LEFT_HIP.value]
            rhip = lm_list[mp_pose.PoseLandmark.RIGHT_HIP.value]
            hip_x = 0.5 * (lhip.x + rhip.x) * frame_w
            hip_y_pix = 0.5 * (lhip.y + rhip.y) * frame_h
            hip_pt_this_frame = (int(hip_x), int(hip_y_pix))

            # Require: not too big, not too small, reasonably centered,
            # and at most ~30% of joints outside the box.
            if (not too_big) and (not too_small) and (not far_from_center) and (frac_outside <= 0.30):
                draw_simple_skeleton(
                    frame,
                    landmark_list_full,
                    mp_pose.POSE_CONNECTIONS,
                    point_color=(252, 0, 219),   # joints
                    line_color=(0, 255, 0),      # skeleton
                    point_radius=3,
                    line_thickness=2,
                    visibility_threshold=0.1,
                )
                drew_skeleton_this_frame = True


        # If this frame is within the analysis window and we have a hip point,
        # add it to the persistent hip path lists.
        if (
            last_step_frame is not None
            and "analysis_start_idx" in locals()
            and frame_idx3 >= analysis_start_idx
            and frame_idx3 <= last_step_frame
            and hip_pt_this_frame is not None
        ):
            if frame_idx3 in worst_frame_set:
                hip_points_worst.append(hip_pt_this_frame)
            else:
                hip_points_normal.append(hip_pt_this_frame)
        
        # Draw accumulated hip path dots so they persist over time
        for (hx, hy) in hip_points_normal:
            cv2.circle(frame, (hx, hy), 5, (255, 255, 0), -1)  # normal points: blue

        for (hx, hy) in hip_points_worst:
            cv2.circle(frame, (hx, hy), 6, (0, 0, 255), -1)      # worst-droop: red


        # Overlay metrics text (with black outline) - larger font
        y0 = 60
        line_spacing = 40
        for i, txt in enumerate(text_lines):
            draw_outlined_text(
                frame,
                txt,
                (10, y0 + line_spacing * i),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,          # ~2x bigger than 0.7
                (0, 255, 0),  # same green fill
                3,            # slightly thicker for readability
            )


        # Mark step frames (larger STEP text)
        if frame_idx3 in step_highlight_frames:
            draw_outlined_text(
                frame,
                "STEP",
                (frame_width - 180, frame_height - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,          # ~2x bigger than 1.0
                (0, 0, 255),  # same red fill
                3,
            )


        # Jump-phase annotation (larger text)
        if last_step_frame is not None and frame_idx3 > last_step_frame:
            draw_outlined_text(
                frame,
                "Jump phase (no hip/stride analysis)",
                (10, frame_height - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,           # ~2x bigger
                (255, 255, 0), # same yellow fill
                3,
            )


        out.write(frame)

        # --- progress every 10 frames ---
        if frame_idx3 % 10 == 0:
            pct = 100.0 * (frame_idx3 + 1) / max(total_frames3, 1)
            print(
                f"\rPass 3/3: {frame_idx3 + 1}/{total_frames3} frames ({pct:5.1f}%)",
                end="",
                flush=True,
            )

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

            crop_to_first = False
            if enable_steps:
                ans2 = input(
                    "Crop out all frames before the FIRST labeled step in the output video? [y/N]: "
                ).strip().lower()
                crop_to_first = ans2 in ("y", "yes")

            analyze_pole_vault_approach(
                video_path,
                enable_step_labeling=enable_steps,
                crop_to_first_step=crop_to_first,
            )
        except Exception as err:
            print(f"Error during analysis: {err}")

