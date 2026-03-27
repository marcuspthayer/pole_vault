# pv_step_labeler.py

import cv2
import tkinter as tk
from tkinter import filedialog


def get_screen_size():
    """Return (screen_width, screen_height) using tkinter."""
    root = tk.Tk()
    root.withdraw()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    root.destroy()
    return screen_w, screen_h


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


def label_steps(video_path):
    """
    Interactive step-labeling UI.

    Controls:
      - 's' : mark a STEP (ground contact) at the current frame.
      - 'e' : end labeling and confirm that the last step labeled was the takeoff step.
      - 'a' : jump BACK 10 frames.
      - 'd' : jump FORWARD 10 frames.
      - 'j' : jump BACK 50 frames.
      - 'l' : jump FORWARD 50 frames.
      - 'z' : jump BACK 1 frame.
      - 'q' or ESC : abort labeling and return (None, None, None).
      - Any other key : go forward 1 frame.

    Returns:
        step_frames (list[int]): sorted list of frame indices where 's' was pressed.
        last_step_frame (int or None): the frame index chosen as the last step.
        fps (float or None): video FPS.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Loaded {video_path}")
    print(f"Resolution: {frame_width} x {frame_height}, FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")

    # Scale window to ~75% of screen
    screen_w, screen_h = get_screen_size()
    max_disp_w = int(screen_w * 0.75)
    max_disp_h = int(screen_h * 0.75)
    scale_factor = min(max_disp_w / frame_width, max_disp_h / frame_height, 1.0)

    disp_width = int(frame_width * scale_factor)
    disp_height = int(frame_height * scale_factor)

    step_frames = []
    last_step_frame = None

    print(
        "\nManual labeling instructions:\n"
        "  - 's' : mark a STEP (ground contact) at the current frame.\n"
        "  - 'e' : mark the LAST visible step (takeoff step) and finish labeling.\n"
        "  - 'a' : jump BACK 10 frames.\n"
        "  - 'd' : jump FORWARD 10 frames.\n"
        "  - 'j' : jump BACK 50 frames.\n"
        "  - 'l' : jump FORWARD 50 frames.\n"
        "  - 'z' : jump BACK 1 frame.\n"
        "  - 'q' or ESC : abort.\n"
        "  - Any other key : go forward 1 frame.\n"
    )

    window_name = "Pole Vault Approach - Manual Step Labeling"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, disp_width, disp_height)

    current_frame_idx = 0

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

        display_frame = frame.copy()

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
            "s=step  e=last  a/d=-/+10  j/l=-/+50  z=-1  q=quit",
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

        # Resize for display
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
            return None, None, None

        elif key in (ord("s"), ord("S")):
            if current_frame_idx not in step_frames:
                step_frames.append(current_frame_idx)
                step_frames.sort()
            last_step_frame = current_frame_idx
            print(
                f"Marked STEP at frame {current_frame_idx} "
                f"(t = {current_frame_idx / fps:.3f} s)"
            )
            current_frame_idx += 1

        elif key in (ord("e"), ord("E")):
            if last_step_frame is None:
                last_step_frame = current_frame_idx
                print(
                    f"No step marked yet; treating frame {current_frame_idx} "
                    f"as last step frame."
                )
            else:
                print(f"Last step frame confirmed at {last_step_frame}.")
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
            current_frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    print(f"Step labeling complete. Steps: {step_frames}, last step: {last_step_frame}")
    return step_frames, last_step_frame, fps
