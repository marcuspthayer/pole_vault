import numpy as np
import cv2
import mediapipe as mp
from pv_yolo_utils import draw_outlined_text, pick_video_file, PersonDetector


def choose_frame_range(video_path):
    """
    Interactive frame selector.

    Controls:
        A / D : -1 / +1 frame
        J / L : -10 / +10 frames
        Z / C : -50 / +50 frames
        S     : set start frame
        E     : set end frame
        Enter : confirm selection
        Esc   : cancel
    """
    print("\n=== Frame Selection ===")
    print("Use these keys in the OpenCV window:")
    print("  A / D : -1 / +1 frame")
    print("  J / L : -10 / +10 frames")
    print("  Z / C : -50 / +50 frames")
    print("  S     : set start frame")
    print("  E     : set end frame")
    print("  Enter : confirm selection")
    print("  Esc   : cancel\n")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return None, None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print("Video has no frames?")
        cap.release()
        return None, None

    current_idx = 0
    start_frame = None
    end_frame = None

    window_name = "Select start/end frames"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    font = cv2.FONT_HERSHEY_SIMPLEX
    window_resized = False

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {current_idx}")
            break

        display = frame.copy()
        h, w = display.shape[:2]

        # Make the window bigger (only once)
        if not window_resized:
            # You can tweak these dimensions if you want even bigger/smaller
            cv2.resizeWindow(window_name, 1600, 900)
            window_resized = True

        # Frame index text
        draw_outlined_text(
            display,
            f"Frame {current_idx}/{total_frames-1}",
            (20, 40),
            font,
            0.8,
            (0, 255, 0),
            2,
        )

        # Show chosen start/end
        y_txt = 80
        if start_frame is not None:
            draw_outlined_text(
                display,
                f"Start: {start_frame}",
                (20, y_txt),
                font,
                0.7,
                (255, 255, 0),
                2,
            )
            y_txt += 30

        if end_frame is not None:
            draw_outlined_text(
                display,
                f"End:   {end_frame}",
                (20, y_txt),
                font,
                0.7,
                (255, 255, 0),
                2,
            )

        # Controls hint (small font at bottom)
        controls_text = "A/D: ±1  J/L: ±10  Z/C: ±50  S: set start  E: set end  Enter: confirm  Esc: cancel"
        cv2.putText(
            display,
            controls_text,
            (20, h - 20),
            font,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow(window_name, display)
        key = cv2.waitKey(0) & 0xFF

        if key == 27:  # Esc
            print("Frame selection cancelled.")
            start_frame, end_frame = None, None
            break

        elif key in (13, 10):  # Enter
            if start_frame is not None and end_frame is not None:
                if start_frame > end_frame:
                    start_frame, end_frame = end_frame, start_frame
                print(f"Selected frames: start={start_frame}, end={end_frame}")
                break
            else:
                print("Please set both start and end frames before confirming.")

        elif key in (ord('a'), ord('A')):
            current_idx = max(0, current_idx - 1)
        elif key in (ord('d'), ord('D')):
            current_idx = min(total_frames - 1, current_idx + 1)
        elif key in (ord('j'), ord('J')):
            current_idx = max(0, current_idx - 10)
        elif key in (ord('l'), ord('L')):
            current_idx = min(total_frames - 1, current_idx + 10)
        elif key in (ord('z'), ord('Z')):
            current_idx = max(0, current_idx - 50)
        elif key in (ord('c'), ord('C')):
            current_idx = min(total_frames - 1, current_idx + 50)
        elif key in (ord('s'), ord('S')):
            start_frame = current_idx
            print(f"Start frame set to {start_frame}")
        elif key in (ord('e'), ord('E')):
            end_frame = current_idx
            print(f"End frame set to {end_frame}")

    cap.release()
    cv2.destroyWindow(window_name)
    return start_frame, end_frame


def _interpolate_nans(y):
    """
    Linearly interpolate NaNs in a 1D array.
    """
    y = np.asarray(y, dtype=np.float32)
    n = len(y)
    x = np.arange(n)
    mask = ~np.isnan(y)

    if mask.sum() == 0:
        return y  # all NaNs, nothing we can do

    y_interp = y.copy()
    y_interp[~mask] = np.interp(x[~mask], x[mask], y[mask])
    return y_interp


def _moving_average(y, window=5):
    """
    Simple moving average with edge padding.
    """
    if window <= 1:
        return y
    y = np.asarray(y, dtype=np.float32)
    pad = window // 2
    y_pad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    y_smooth = np.convolve(y_pad, kernel, mode="same")[pad:-pad]
    return y_smooth


def _clean_contact_signal(contact, min_contact_frames=3):
    """
    Remove very short 'contact' segments (1-2 frames) as likely noise.
    """
    contact = contact.astype(bool)
    n = len(contact)
    i = 0
    while i < n:
        if contact[i]:
            j = i + 1
            while j < n and contact[j]:
                j += 1
            if j - i < min_contact_frames:
                contact[i:j] = False
            i = j
        else:
            i += 1
    return contact


def run_ground_contact_analysis(video_path, start_frame, end_frame):
    """
    Run Option A (rule-based) ground contact detection using MediaPipe,
    but only inside the person bounding box from PersonDetector, and map
    BOTH feet's coordinates back to full-frame normalized coordinates.

    We return:
      - right_foot_x/y (raw + smooth), right_contact, right step metrics
      - left_foot_x/y (raw + smooth), left_contact, left step metrics
      - per-frame YOLO bbox and hip/knee/ankle joint positions for overlay
    """
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # fallback
    dt = 1.0 / fps

    if start_frame is None or end_frame is None:
        print("start_frame or end_frame is None, skipping analysis.")
        cap.release()
        return

    start_frame = max(0, int(start_frame))
    end_frame = min(total_frames - 1, int(end_frame))
    if start_frame > end_frame:
        start_frame, end_frame = end_frame, start_frame

    n_frames = end_frame - start_frame + 1

    # Separate arrays for each foot
    right_foot_x = []
    right_foot_y = []
    left_foot_x = []
    left_foot_y = []
    timestamps = []

    # Bboxes and joints for overlay
    bboxes = []  # (x1, y1, x2, y2) in pixels, or (-1, -1, -1, -1) if invalid

    right_hip_x = []
    right_hip_y = []
    right_knee_x = []
    right_knee_y = []
    right_ankle_x = []
    right_ankle_y = []

    left_hip_x = []
    left_hip_y = []
    left_knee_x = []
    left_knee_y = []
    left_ankle_x = []
    left_ankle_y = []

    print(
        f"Running ground contact analysis on frames {start_frame} to {end_frame} "
        f"({n_frames} frames, fps={fps:.2f})"
    )

    # Use your YOLO-based person detector
    detector = PersonDetector()
    last_bbox = None  # for simple tracking when YOLO misses occasionally

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for i in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                print(f"\nFailed to read frame at index {start_frame + i}, stopping.")
                break

            frame_idx = start_frame + i
            t = frame_idx * dt

            h_full, w_full = frame.shape[:2]

            # --- YOLO person detection ---
            bbox = detector.detect_largest_person(frame)
            if bbox is None:
                # fall back to previous bbox if we have one
                if last_bbox is not None:
                    bbox = last_bbox
                else:
                    # No detection and no history: both feet and all joints are NaN, bbox invalid
                    right_foot_x.append(np.nan)
                    right_foot_y.append(np.nan)
                    left_foot_x.append(np.nan)
                    left_foot_y.append(np.nan)
                    timestamps.append(t)
                    bboxes.append((-1, -1, -1, -1))

                    for arr in [
                        right_hip_x, right_hip_y, right_knee_x, right_knee_y,
                        right_ankle_x, right_ankle_y, left_hip_x, left_hip_y,
                        left_knee_x, left_knee_y, left_ankle_x, left_ankle_y
                    ]:
                        arr.append(np.nan)

                    progress = 100.0 * (i + 1) / float(n_frames)
                    print(
                        f"\rProcessing frames: {i+1}/{n_frames} "
                        f"({progress:5.1f}%)",
                        end="",
                        flush=True,
                    )
                    continue
            last_bbox = bbox

            x1, y1, x2, y2 = bbox
            # Safety check on bbox
            x1 = max(0, min(w_full - 1, x1))
            x2 = max(0, min(w_full, x2))
            y1 = max(0, min(h_full - 1, y1))
            y2 = max(0, min(h_full, y2))
            if x2 <= x1 or y2 <= y1:
                # Bad bbox, record NaNs and invalid bbox
                right_foot_x.append(np.nan)
                right_foot_y.append(np.nan)
                left_foot_x.append(np.nan)
                left_foot_y.append(np.nan)
                timestamps.append(t)
                bboxes.append((-1, -1, -1, -1))

                for arr in [
                    right_hip_x, right_hip_y, right_knee_x, right_knee_y,
                    right_ankle_x, right_ankle_y, left_hip_x, left_hip_y,
                    left_knee_x, left_knee_y, left_ankle_x, left_ankle_y
                ]:
                    arr.append(np.nan)

                progress = 100.0 * (i + 1) / float(n_frames)
                print(
                    f"\rProcessing frames: {i+1}/{n_frames} "
                    f"({progress:5.1f}%)",
                    end="",
                    flush=True,
                )
                continue

            person_roi = frame[y1:y2, x1:x2]
            if person_roi.size == 0:
                right_foot_x.append(np.nan)
                right_foot_y.append(np.nan)
                left_foot_x.append(np.nan)
                left_foot_y.append(np.nan)
                timestamps.append(t)
                bboxes.append((-1, -1, -1, -1))

                for arr in [
                    right_hip_x, right_hip_y, right_knee_x, right_knee_y,
                    right_ankle_x, right_ankle_y, left_hip_x, left_hip_y,
                    left_knee_x, left_knee_y, left_ankle_x, left_ankle_y
                ]:
                    arr.append(np.nan)

                progress = 100.0 * (i + 1) / float(n_frames)
                print(
                    f"\rProcessing frames: {i+1}/{n_frames} "
                    f"({progress:5.1f}%)",
                    end="",
                    flush=True,
                )
                continue

            # Convert cropped ROI to RGB for MediaPipe
            roi_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
            result = pose.process(roi_rgb)

            # Defaults: NaNs
            rx_val = ry_val = np.nan
            lx_val = ly_val = np.nan

            rhip_x = rhip_y = np.nan
            rknee_x = rknee_y = np.nan
            rankle_x = rankle_y = np.nan
            lhip_x = lhip_y = np.nan
            lknee_x = lknee_y = np.nan
            lankle_x = lankle_y = np.nan

            box_w = float(x2 - x1)
            box_h = float(y2 - y1)

            if result.pose_landmarks and box_w > 0 and box_h > 0:
                lm = result.pose_landmarks.landmark

                # Helper: map normalized ROI coords to full-frame normalized coords
                def roi_to_full(norm_x, norm_y):
                    x_full = x1 + norm_x * box_w
                    y_full = y1 + norm_y * box_h
                    return x_full / float(w_full), y_full / float(h_full)

                # Right foot
                r_idx = mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value
                r_lm = lm[r_idx]
                if r_lm.visibility > 0.5:
                    rx_val, ry_val = roi_to_full(r_lm.x, r_lm.y)

                # Left foot
                l_idx = mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value
                l_lm = lm[l_idx]
                if l_lm.visibility > 0.5:
                    lx_val, ly_val = roi_to_full(l_lm.x, l_lm.y)

                # Right hip, knee, ankle
                rh_idx = mp_pose.PoseLandmark.RIGHT_HIP.value
                rk_idx = mp_pose.PoseLandmark.RIGHT_KNEE.value
                ra_idx = mp_pose.PoseLandmark.RIGHT_ANKLE.value

                rh_lm = lm[rh_idx]
                rk_lm = lm[rk_idx]
                ra_lm = lm[ra_idx]

                if rh_lm.visibility > 0.5:
                    rhip_x, rhip_y = roi_to_full(rh_lm.x, rh_lm.y)
                if rk_lm.visibility > 0.5:
                    rknee_x, rknee_y = roi_to_full(rk_lm.x, rk_lm.y)
                if ra_lm.visibility > 0.5:
                    rankle_x, rankle_y = roi_to_full(ra_lm.x, ra_lm.y)

                # Left hip, knee, ankle
                lh_idx = mp_pose.PoseLandmark.LEFT_HIP.value
                lk_idx = mp_pose.PoseLandmark.LEFT_KNEE.value
                la_idx = mp_pose.PoseLandmark.LEFT_ANKLE.value

                lh_lm = lm[lh_idx]
                lk_lm = lm[lk_idx]
                la_lm = lm[la_idx]

                if lh_lm.visibility > 0.5:
                    lhip_x, lhip_y = roi_to_full(lh_lm.x, lh_lm.y)
                if lk_lm.visibility > 0.5:
                    lknee_x, lknee_y = roi_to_full(lk_lm.x, lk_lm.y)
                if la_lm.visibility > 0.5:
                    lankle_x, lankle_y = roi_to_full(la_lm.x, la_lm.y)

            # Append per-frame data
            right_foot_x.append(rx_val)
            right_foot_y.append(ry_val)
            left_foot_x.append(lx_val)
            left_foot_y.append(ly_val)
            timestamps.append(t)

            bboxes.append((x1, y1, x2, y2))

            right_hip_x.append(rhip_x)
            right_hip_y.append(rhip_y)
            right_knee_x.append(rknee_x)
            right_knee_y.append(rknee_y)
            right_ankle_x.append(rankle_x)
            right_ankle_y.append(rankle_y)

            left_hip_x.append(lhip_x)
            left_hip_y.append(lhip_y)
            left_knee_x.append(lknee_x)
            left_knee_y.append(lknee_y)
            left_ankle_x.append(lankle_x)
            left_ankle_y.append(lankle_y)

            # Progress tracker
            progress = 100.0 * (i + 1) / float(n_frames)
            print(
                f"\rProcessing frames: {i+1}/{n_frames} "
                f"({progress:5.1f}%)",
                end="",
                flush=True,
            )

    cap.release()
    print()  # newline after progress bar

    # Convert to arrays
    right_foot_x = np.array(right_foot_x, dtype=np.float32)
    right_foot_y = np.array(right_foot_y, dtype=np.float32)
    left_foot_x = np.array(left_foot_x, dtype=np.float32)
    left_foot_y = np.array(left_foot_y, dtype=np.float32)
    timestamps = np.array(timestamps, dtype=np.float32)
    bboxes = np.array(bboxes, dtype=np.int32)

    right_hip_x = np.array(right_hip_x, dtype=np.float32)
    right_hip_y = np.array(right_hip_y, dtype=np.float32)
    right_knee_x = np.array(right_knee_x, dtype=np.float32)
    right_knee_y = np.array(right_knee_y, dtype=np.float32)
    right_ankle_x = np.array(right_ankle_x, dtype=np.float32)
    right_ankle_y = np.array(right_ankle_y, dtype=np.float32)

    left_hip_x = np.array(left_hip_x, dtype=np.float32)
    left_hip_y = np.array(left_hip_y, dtype=np.float32)
    left_knee_x = np.array(left_knee_x, dtype=np.float32)
    left_knee_y = np.array(left_knee_y, dtype=np.float32)
    left_ankle_x = np.array(left_ankle_x, dtype=np.float32)
    left_ankle_y = np.array(left_ankle_y, dtype=np.float32)

    if np.all(np.isnan(right_foot_y)) and np.all(np.isnan(left_foot_y)):
        print("All foot positions are NaN – detections failed on this segment.")
        return

    # Helper: process a single foot's vertical trajectory
    def analyze_single_foot(foot_x, foot_y, label):
        foot_x = _interpolate_nans(foot_x)
        foot_y = _interpolate_nans(foot_y)

        foot_x_smooth = _moving_average(foot_x, window=5)
        foot_y_smooth = _moving_average(foot_y, window=5)

        # Vertical velocity dy/dt
        dy = np.diff(foot_y_smooth) / dt
        dy = np.concatenate(([dy[0]], dy))  # same length as foot_y_smooth
        vel_mag = np.abs(dy)

        # Heuristic thresholds (per-foot):
        # - Contact when foot is relatively low (big y value) AND velocity is small.
        y_thr = np.percentile(foot_y_smooth, 80)
        vel_thr = np.percentile(vel_mag, 30)

        contact = (foot_y_smooth >= y_thr) & (vel_mag <= vel_thr)
        contact = _clean_contact_signal(contact, min_contact_frames=3)

        # Step events = rising edges in contact (0 -> 1)
        step_indices = np.where((contact[1:] == True) & (contact[:-1] == False))[0] + 1
        step_times = timestamps[step_indices]

        if len(timestamps) >= 2:
            total_time = timestamps[-1] - timestamps[0]
        else:
            total_time = 0.0

        num_steps = len(step_indices)
        if total_time > 0:
            step_frequency = num_steps / total_time  # steps per second
        else:
            step_frequency = 0.0

        # Optional: average step interval
        if len(step_times) >= 2:
            intervals = np.diff(step_times)
            mean_interval = float(np.mean(intervals))
        else:
            mean_interval = float("nan")

        print(f"--- {label} foot ---")
        print(f"  Steps detected: {num_steps}")
        print(f"  Step frequency: {step_frequency:.3f} steps/s")
        if not np.isnan(mean_interval):
            print(f"  Mean step interval: {mean_interval:.3f} s")

        return {
            "x_raw": foot_x,
            "y_raw": foot_y,
            "x_smooth": foot_x_smooth,
            "y_smooth": foot_y_smooth,
            "dy": dy,
            "contact": contact,
            "step_indices": step_indices,
            "step_times": step_times,
            "step_frequency": step_frequency,
            "mean_step_interval": mean_interval,
        }

    print("---- Ground contact summary ----")
    print(f"Frames analyzed: {n_frames} (index {start_frame} to {end_frame})")
    if len(timestamps) >= 2:
        print(f"Total time window: {timestamps[-1] - timestamps[0]:.3f} s")

    right_data = analyze_single_foot(right_foot_x, right_foot_y, label="Right")
    left_data = analyze_single_foot(left_foot_x, left_foot_y, label="Left")

    return {
        "timestamps": timestamps,
        "right_foot_x_raw": right_data["x_raw"],
        "right_foot_y_raw": right_data["y_raw"],
        "right_foot_x_smooth": right_data["x_smooth"],
        "right_foot_y_smooth": right_data["y_smooth"],
        "right_dy": right_data["dy"],
        "right_contact": right_data["contact"],
        "right_step_indices": right_data["step_indices"],
        "right_step_times": right_data["step_times"],
        "right_step_frequency": right_data["step_frequency"],
        "right_mean_step_interval": right_data["mean_step_interval"],
        "left_foot_x_raw": left_data["x_raw"],
        "left_foot_y_raw": left_data["y_raw"],
        "left_foot_x_smooth": left_data["x_smooth"],
        "left_foot_y_smooth": left_data["y_smooth"],
        "left_dy": left_data["dy"],
        "left_contact": left_data["contact"],
        "left_step_indices": left_data["step_indices"],
        "left_step_times": left_data["step_times"],
        "left_step_frequency": left_data["step_frequency"],
        "left_mean_step_interval": left_data["mean_step_interval"],
        "start_frame": start_frame,
        "end_frame": end_frame,
        "fps": fps,
        "bboxes": bboxes,
        "right_hip_x": right_hip_x,
        "right_hip_y": right_hip_y,
        "right_knee_x": right_knee_x,
        "right_knee_y": right_knee_y,
        "right_ankle_x": right_ankle_x,
        "right_ankle_y": right_ankle_y,
        "left_hip_x": left_hip_x,
        "left_hip_y": left_hip_y,
        "left_knee_x": left_knee_x,
        "left_knee_y": left_knee_y,
        "left_ankle_x": left_ankle_x,
        "left_ankle_y": left_ankle_y,
    }


def save_contact_overlay_video(video_path, results, which_foot="right", output_path=None):
    """
    Create a video overlay that:
      - Draws the YOLO person bounding box
      - Draws hips and legs of the skeleton
      - Draws separate path dots for each foot (permanent trail), with
        contact vs swing colors.
    """
    if results is None:
        print("No results to visualize.")
        return

    start_frame = int(results["start_frame"])
    end_frame = int(results["end_frame"])
    fps = results["fps"]

    # Foot trajectories & contact (already smoothed)
    right_x = results["right_foot_x_smooth"]
    right_y = results["right_foot_y_smooth"]
    right_contact = results["right_contact"]

    left_x = results["left_foot_x_smooth"]
    left_y = results["left_foot_y_smooth"]
    left_contact = results["left_contact"]

    # Bboxes & joints
    bboxes = results["bboxes"]  # shape (n_frames, 4)
    rhip_x = results["right_hip_x"]
    rhip_y = results["right_hip_y"]
    rknee_x = results["right_knee_x"]
    rknee_y = results["right_knee_y"]
    rankle_x = results["right_ankle_x"]
    rankle_y = results["right_ankle_y"]

    lhip_x = results["left_hip_x"]
    lhip_y = results["left_hip_y"]
    lknee_x = results["left_knee_x"]
    lknee_y = results["left_knee_y"]
    lankle_x = results["left_ankle_x"]
    lankle_y = results["left_ankle_y"]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video for overlay: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = max(0, start_frame)
    end_frame = min(total_frames - 1, end_frame)
    n_frames = end_frame - start_frame + 1

    # Get frame size
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, frame = cap.read()
    if not ret:
        print("Failed to read first frame for overlay.")
        cap.release()
        return
    h, w = frame.shape[:2]

    # Prepare VideoWriter
    if output_path is None:
        import os
        base, ext = os.path.splitext(video_path)
        output_path = f"{base}_foot_contact_overlay.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Path points (in pixel space) for each foot
    right_path_points = []  # (x_px, y_px, is_contact)
    left_path_points = []

    print(f"Saving overlay video to: {output_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    def norm_to_px(xn, yn):
        if np.isnan(xn) or np.isnan(yn):
            return None
        x_clamp = float(np.clip(xn, 0.0, 1.0))
        y_clamp = float(np.clip(yn, 0.0, 1.0))
        return int(x_clamp * w), int(y_clamp * h)

    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"\nFailed to read frame at index {start_frame + i}, stopping overlay.")
            break

        idx = i

        # --- Draw YOLO bbox ---
        x1, y1, x2, y2 = [int(v) for v in bboxes[idx]]
        if x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # --- Skeleton: hips and legs ---
        # Convert normalized joints to pixels
        rhip = norm_to_px(rhip_x[idx], rhip_y[idx])
        rknee = norm_to_px(rknee_x[idx], rknee_y[idx])
        rankle = norm_to_px(rankle_x[idx], rankle_y[idx])

        lhip = norm_to_px(lhip_x[idx], lhip_y[idx])
        lknee = norm_to_px(lknee_x[idx], lknee_y[idx])
        lankle = norm_to_px(lankle_x[idx], lankle_y[idx])

        # Line between hips
        if rhip is not None and lhip is not None:
            cv2.line(frame, rhip, lhip, (255, 255, 255), 2)

        # Right leg: hip-knee-ankle
        if rhip is not None and rknee is not None:
            cv2.line(frame, rhip, rknee, (255, 255, 255), 2)
        if rknee is not None and rankle is not None:
            cv2.line(frame, rknee, rankle, (255, 255, 255), 2)

        # Left leg: hip-knee-ankle
        if lhip is not None and lknee is not None:
            cv2.line(frame, lhip, lknee, (255, 255, 255), 2)
        if lknee is not None and lankle is not None:
            cv2.line(frame, lknee, lankle, (255, 255, 255), 2)

        # --- Foot positions & permanent path dots ---

        # Right foot
        r_pos = norm_to_px(right_x[idx], right_y[idx])
        if r_pos is not None:
            rx_px, ry_px = r_pos
            right_path_points.append((rx_px, ry_px, bool(right_contact[idx])))

        # Left foot
        l_pos = norm_to_px(left_x[idx], left_y[idx])
        if l_pos is not None:
            lx_px, ly_px = l_pos
            left_path_points.append((lx_px, ly_px, bool(left_contact[idx])))

        # Draw right foot path
        for (px, py, is_contact) in right_path_points:
            # Right: swing = blue, contact = red
            color = (255, 0, 0) if not is_contact else (0, 0, 255)
            cv2.circle(frame, (px, py), 3, color, -1)

        # Draw left foot path
        for (px, py, is_contact) in left_path_points:
            # Left: swing = yellow, contact = magenta
            color = (0, 255, 255) if not is_contact else (255, 0, 255)
            cv2.circle(frame, (px, py), 3, color, -1)

        # Highlight current frame's foot positions a bit larger
        if r_pos is not None:
            rx_px, ry_px = r_pos
            current_color_r = (0, 255, 0) if right_contact[idx] else (255, 255, 0)
            cv2.circle(frame, (rx_px, ry_px), 6, current_color_r, -1)

        if l_pos is not None:
            lx_px, ly_px = l_pos
            current_color_l = (0, 128, 255) if left_contact[idx] else (128, 0, 255)
            cv2.circle(frame, (lx_px, ly_px), 6, current_color_l, -1)

        # --- Text overlays: per-foot contact state + frame index ---
        right_state_text = "CONTACT" if right_contact[idx] else "SWING"
        right_state_color = (0, 255, 0) if right_contact[idx] else (0, 255, 255)

        left_state_text = "CONTACT" if left_contact[idx] else "SWING"
        left_state_color = (0, 255, 0) if left_contact[idx] else (0, 255, 255)

        draw_outlined_text(
            frame,
            f"Right foot: {right_state_text}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            right_state_color,
            2,
        )

        draw_outlined_text(
            frame,
            f"Left foot:  {left_state_text}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            left_state_color,
            2,
        )

        draw_outlined_text(
            frame,
            f"Frame {start_frame + i}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        writer.write(frame)

        # Progress
        progress = 100.0 * (i + 1) / float(n_frames)
        print(
            f"\rWriting overlay video frames: {i+1}/{n_frames} ({progress:5.1f}%)",
            end="",
            flush=True,
        )

    print()  # newline
    writer.release()
    cap.release()
    print("Overlay video saved.")



def main():
    video_path = pick_video_file()
    if not video_path:
        print("No video selected.")
        return

    print(f"Selected video: {video_path}")
    start_frame, end_frame = choose_frame_range(video_path)
    if start_frame is None or end_frame is None:
        print("No valid frame range selected, exiting.")
        return

    results = run_ground_contact_analysis(
        video_path,
        start_frame,
        end_frame,
    )

    if results is not None:
        print("Right step times (s):", results["right_step_times"])
        print("Left step times  (s):", results["left_step_times"])

        # Overlay video with bbox, skeleton, and both feet
        save_contact_overlay_video(
            video_path,
            results,
            which_foot="right",
            output_path=None,
        )


if __name__ == "__main__":
    main()
