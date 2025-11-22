import os
import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import filedialog
from pv_yolo_utils import PersonDetector


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def choose_video_file():
    """Open a file dialog and return selected video path or None."""
    root = tk.Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(
        title="Select pole vault video",
        filetypes=[
            ("Video files", "*.mp4 *.mov *.avi *.mkv"),
            ("All files", "*.*"),
        ],
    )
    root.update_idletasks()
    root.destroy()
    if not video_path:
        return None
    return video_path


def get_screen_size():
    """Return (screen_width, screen_height) using tkinter."""
    root = tk.Tk()
    root.withdraw()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    root.destroy()
    return screen_w, screen_h


def analyze_pole_vault_approach(video_path):
    """
    Analyze the last approach steps of a pole vaulter.

    Workflow:
      1. First pass: manual step labeling with random access.
         - 's' marks a step (ground contact) at the current frame.
         - 'e' marks the LAST visible step before takeoff and ends labeling.
         - 'a' / 'd' skip -10 / +10 frames.
         - 'j' / 'l' skip -50 / +50 frames.
         - 'q' or ESC aborts.
         - Any other key: go to next frame (+1).

      2. Compute:
         - Stride rate (steps/s and steps/min) from marked steps.
         - Hip droop metric: change in hip height (as % of body height)
           between early approach and pre-takeoff.

      3. Second pass: write processed video with:
         - MediaPipe pose wireframe.
         - Metrics text overlay.
         - 'STEP' label on step frames.
         - 'Jump phase (no hip/stride analysis)' after last step.

    Returns:
        final_output (str): Path to the processed video, or None if aborted.
    """
    destination_folder = os.path.dirname(video_path)
    fn = os.path.basename(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Loaded {fn}")
    print(f"Resolution: {frame_width} x {frame_height}, FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")

    # --- figure out how big to display the window (75% of screen, preserving aspect) ---
    screen_w, screen_h = get_screen_size()
    max_disp_w = int(screen_w * 0.75)
    max_disp_h = int(screen_h * 0.75)
    scale_factor = min(max_disp_w / frame_width, max_disp_h / frame_height, 1.0)

    disp_width = int(frame_width * scale_factor)
    disp_height = int(frame_height * scale_factor)

    step_frames = []          # frame indices (0-based) where a step occurs
    last_step_frame = None

    print(
        "\nManual labeling instructions:\n"
        "  - 's' : mark a STEP (ground contact) at the current frame.\n"
        "  - 'e' : mark the LAST visible step (takeoff step) and finish labeling.\n"
        "  - 'a' : jump BACK 10 frames.\n"
        "  - 'd' : jump FORWARD 10 frames.\n"
        "  - 'j' : jump BACK 50 frames.\n"
        "  - 'l' : jump FORWARD 50 frames.\n"
        "  - 'q' or ESC : abort.\n"
        "  - Any other key : go forward 1 frame.\n"
    )

    # Create a resizable window up front (OpenCV)
    window_name = "Pole Vault Approach - Manual Step Labeling"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, disp_width, disp_height)

    current_frame_idx = 0

    # ---------- FIRST PASS: manual labeling only ----------
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=2,
        static_image_mode=False,
        smooth_landmarks=True,
    ) as pose:

        while True:
            # Clamp frame index
            if current_frame_idx < 0:
                current_frame_idx = 0
            if current_frame_idx >= total_frames:
                print("Reached end of video during labeling.")
                break

            # Seek to the desired frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
            ret, frame = cap.read()
            if not ret:
                print("Could not read frame. Ending labeling.")
                break

            # Run MediaPipe pose for visualization (no metrics yet)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = pose.process(image_rgb)

            display_frame = frame.copy()
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    display_frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )

            # Overlay frame index and key help
            cv2.putText(
                display_frame,
                f"Frame {current_frame_idx}/{total_frames-1}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display_frame,
                "s=step  e=last  a/d=-/+10  j/l=-/+50  q=quit",
                (10, frame_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # Draw timeline ticks for labeled steps
            if step_frames:
                denom = max(total_frames - 1, 1)
                for sf in step_frames:
                    x_pos = int(frame_width * sf / denom)
                    cv2.line(
                        display_frame,
                        (x_pos, frame_height - 40),
                        (x_pos, frame_height - 60),
                        (0, 0, 255),
                        2,
                    )

            # Resize for display to fit 75% of screen
            if scale_factor != 1.0:
                disp_frame = cv2.resize(
                    display_frame, (disp_width, disp_height), interpolation=cv2.INTER_AREA
                )
            else:
                disp_frame = display_frame

            cv2.imshow(window_name, disp_frame)
            key = cv2.waitKey(0) & 0xFF

            if key in (ord("q"), 27):  # 'q' or ESC
                print("Aborted by user.")
                cap.release()
                cv2.destroyAllWindows()
                return None

            elif key in (ord("s"), ord("S")):
                if current_frame_idx not in step_frames:
                    step_frames.append(current_frame_idx)
                    step_frames.sort()
                last_step_frame = current_frame_idx
                print(f"Marked STEP at frame {current_frame_idx} (t = {current_frame_idx / fps:.3f} s)")
                # Move forward one frame by default
                current_frame_idx += 1

            elif key in (ord("e"), ord("E")):
                if last_step_frame is None:
                    last_step_frame = current_frame_idx
                    print(
                        f"No step marked yet; treating frame {current_frame_idx} as last step frame."
                    )
                else:
                    print(f"Last step frame confirmed at {last_step_frame}.")
                # End labeling
                break

            elif key in (ord("a"), ord("A")):   # back 10
                current_frame_idx -= 10

            elif key in (ord("d"), ord("D")):   # forward 10
                current_frame_idx += 10

            elif key in (ord("j"), ord("J")):   # back 50
                current_frame_idx -= 50

            elif key in (ord("l"), ord("L")):   # forward 50
                current_frame_idx += 50
            
            elif key in (ord("z"), ord("Z")):   # back 1
                current_frame_idx -= 1

            else:
                # Default: go forward 1 frame
                current_frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    if last_step_frame is None:
        print("No last step frame selected; cannot compute approach metrics.")
        return None

    print(f"Labeling complete. Last step frame: {last_step_frame}")

    # ---------- SECOND PASS: compute metrics (YOLO-guided ROI for Pose) ----------
    cap2 = cv2.VideoCapture(video_path)
    if not cap2.isOpened():
        raise IOError(f"Could not reopen video file: {video_path}")

    # Use YOLO to guide where we run Pose
    detector = PersonDetector(model_path="yolo11n.pt", conf=0.25)

    hip_y_list = []
    body_height_list = []

    with mp_pose.Pose(
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
        model_complexity=2,
        static_image_mode=False,
        smooth_landmarks=True,
    ) as pose2:
        frame_idx2 = -1
        while True:
            ret, frame = cap2.read()
            if not ret:
                break
            frame_idx2 += 1

            frame_h, frame_w = frame.shape[:2]

            # 1) YOLO detection to find the largest "person"
            bbox = detector.detect_largest_person(frame)

            if bbox is None:
                # No person found: mark as NaN for this frame
                hip_y_norm = np.nan
                body_height_norm = np.nan
            else:
                x1, y1, x2, y2 = bbox

                # 2) Expand the box a bit so we don't cut off limbs
                margin = 0.25  # 25% extra around box
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

                # 3) Run MediaPipe Pose on the ROI
                image_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                results2 = pose2.process(image_rgb)

                if results2.pose_landmarks:
                    lm = results2.pose_landmarks.landmark

                    # Helper to convert ROI landmark -> full-frame normalized y
                    def lm_full_y(landmark):
                        y_roi = landmark.y * roi_h_actual
                        y_full = roi_y1 + y_roi
                        return y_full / frame_h

                    # Left/right hip
                    lhip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
                    rhip = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]
                    hip_y_norm = 0.5 * (lm_full_y(lhip) + lm_full_y(rhip))

                    # Nose
                    nose = lm[mp_pose.PoseLandmark.NOSE.value]
                    nose_y_norm = lm_full_y(nose)

                    # Ankles: take the lower (larger y) one
                    lankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                    rankle = lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                    ankle_y_norm = max(lm_full_y(lankle), lm_full_y(rankle))

                    # Approximate body height as ankle-to-nose in normalized coords
                    body_height_norm = ankle_y_norm - nose_y_norm

                else:
                    hip_y_norm = np.nan
                    body_height_norm = np.nan

            hip_y_list.append(hip_y_norm)
            body_height_list.append(body_height_norm)

    cap2.release()


    hip_y_arr = np.array(hip_y_list, dtype=float)
    body_h_arr = np.array(body_height_list, dtype=float)

    # Only use data up to and including last_step_frame for approach metrics
    n_approach_frames = min(last_step_frame + 1, len(hip_y_arr))
    hip_y = hip_y_arr[:n_approach_frames]
    body_height = body_h_arr[:n_approach_frames]

    # ---------- STRIDE RATE ----------
    stride_rate = None  # steps / second
    cadence = None      # steps / minute

    if len(step_frames) >= 2:
        step_times = np.array(step_frames, dtype=float) / fps
        step_intervals = np.diff(step_times)
        valid = step_intervals > 0
        if np.any(valid):
            mean_interval = np.mean(step_intervals[valid])
            stride_rate = 1.0 / mean_interval
            cadence = stride_rate * 60.0

    # ---------- HIP DROOP METRIC ----------
    hip_droop_pct = None
    hip_droop_trend_pct = None

    valid_mask = (~np.isnan(hip_y)) & (~np.isnan(body_height)) & (body_height > 1e-4)
    if np.any(valid_mask):
        hip_y_valid = hip_y[valid_mask]
        body_h_valid = body_height[valid_mask]

        n_valid = len(hip_y_valid)
        k = max(3, int(0.2 * n_valid))  # ~first/last 20%

        baseline_hip_y = float(np.mean(hip_y_valid[:k]))
        baseline_height = float(np.mean(body_h_valid[:k]))

        # Hip drop (positive = hip lower than early approach) as % of body height
        hip_drop_series = (hip_y_valid - baseline_hip_y) / baseline_height * 100.0

        # Average drop over the last ~20% of the approach
        hip_droop_pct = float(np.mean(hip_drop_series[-k:]))

        # Trend: last 20% minus first 20%
        first_mean = float(np.mean(hip_drop_series[:k]))
        last_mean = float(np.mean(hip_drop_series[-k:]))
        hip_droop_trend_pct = last_mean - first_mean

    print("\n=== Approach Metrics ===")
    print(f"Last step frame: {last_step_frame} (t = {last_step_frame / fps:.3f} s)")
    if stride_rate is not None:
        print(f"Stride rate (steps/s): {stride_rate:.3f}")
        print(f"Cadence (steps/min): {cadence:.1f}")
    else:
        print("Stride rate: not enough steps labeled to compute.")

    if hip_droop_pct is not None:
        print(
            f"Hip droop near takeoff: {hip_droop_pct:+.2f}% of body height "
            f"(positive = hip lower than early approach)."
        )
        print(
            f"Hip droop trend (last vs first 20%): {hip_droop_trend_pct:+.2f}% of body height."
        )
    else:
        print("Hip droop: insufficient pose data to compute.")

    # ---------- THIRD PASS: write processed video with overlay ----------
    cap3 = cv2.VideoCapture(video_path)
    if not cap3.isOpened():
        raise IOError(f"Could not reopen video file for writing: {video_path}")

    final_output = os.path.join(
        destination_folder,
        os.path.splitext(fn)[0] + "_pv_processed.mp4",
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(final_output, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        raise IOError("Could not open VideoWriter for output video.")

    text_lines = []
    if stride_rate is not None:
        text_lines.append(f"Stride rate: {stride_rate:.2f} steps/s ({cadence:.0f} spm)")
    else:
        text_lines.append("Stride rate: N/A")

    if hip_droop_pct is not None:
        text_lines.append(f"Hip droop: {hip_droop_pct:+.1f}% of height")
    else:
        text_lines.append("Hip droop: N/A")

    text_lines.append("Analysis: up to last ground step only.")

    step_frames_set = set(step_frames)

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=2,
        static_image_mode=False,
        smooth_landmarks=True,
    ) as pose3:
        frame_idx3 = -1
        while True:
            ret, frame = cap3.read()
            if not ret:
                break

            frame_idx3 += 1

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results3 = pose3.process(image_rgb)

            if results3.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results3.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )

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

            # Mark jump phase after last step
            if frame_idx3 > last_step_frame:
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
            analyze_pole_vault_approach(video_path)
        except Exception as err:
            print(f"Error during analysis: {err}")
