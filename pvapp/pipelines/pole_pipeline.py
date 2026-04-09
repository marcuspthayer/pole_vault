import cv2
import numpy as np
from pvapp.core.pole_manager import PoleDetector


class LightPoleResult:
    """Lightweight wrapper holding only the polygon contours from YOLO masks.

    Downstream code accesses `result.masks.xy` — this object mimics that
    interface without holding the full-resolution mask arrays or the
    original frame, cutting memory from ~22 MB/frame to ~1 KB/frame.
    """
    __slots__ = ("masks",)

    def __init__(self, xy_list):
        self.masks = _LightMasks(xy_list)


class _LightMasks:
    __slots__ = ("xy",)

    def __init__(self, xy_list):
        self.xy = xy_list


def extract_pole_data(video_path: str, model_path="pole_detector_v3.pt", conf=0.25, start_frame=0, end_frame=None, progress_callback=None, device=None):
    """
    Run Pole Detection on the video range.

    Returns:
        list: LightPoleResult per frame (with .masks.xy polygon contours)
              or None if nothing detected. length == total_frames.
    """

    detector = PoleDetector(model_path=model_path, conf=conf, device=device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame is None:
        end_frame = total_frames - 1

    start_frame = max(0, start_frame) if start_frame is not None else 0
    end_frame = min(end_frame, total_frames - 1)

    # Pre-fill with None
    pole_results = [None] * start_frame

    # Seek
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame
    process_count = 0
    total_to_process = end_frame - start_frame + 1

    while frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection
        result = detector.detect(frame)

        # Strip heavy YOLO Result down to just polygon contours
        if result is not None and result.masks is not None:
            xy_list = [poly.copy() for poly in result.masks.xy]
            pole_results.append(LightPoleResult(xy_list))
        else:
            pole_results.append(None)

        frame_idx += 1
        process_count += 1
        if progress_callback and process_count % 10 == 0:
            pct = process_count / max(total_to_process, 1)
            progress_callback(pct, f"Extracting Pole Data: {int(pct*100)}%")

    cap.release()

    # Fill remaining
    while len(pole_results) < total_frames:
        pole_results.append(None)

    if progress_callback:
        progress_callback(1.0, "Pole Extraction Complete")

    return pole_results
