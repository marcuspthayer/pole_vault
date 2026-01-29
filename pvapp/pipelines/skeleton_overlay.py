import os
import time
import logging
import cv2
import mediapipe as mp
import subprocess

from pv_yolo_utils import PersonDetector, draw_outlined_text
from pvapp.pose import detect_person_roi, run_pose_on_frame
from pvapp.render import draw_simple_skeleton
from pvapp.logging_utils import setup_logger

mp_pose = mp.solutions.pose

logger = logging.getLogger("pv.pipeline")

def process_video_skeleton_overlay(
    video_path: str,
    output_path: str | None = None,
    *,
    log_file: str | None = None,
    max_frames: int | None = None,
    yolo_model_path: str = "yolo11n.pt",
    conf: float = 0.25,
    margin: float = 0.30,
    draw_roi_box: bool = True,
    draw_metrics_header: bool = True,
) -> str:
    """Create an output video with YOLO ROI + MediaPipe skeleton overlay."""
    setup_logger("pv", log_file=log_file)
    logger.info(f"Starting skeleton overlay | video={video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Video props | fps={fps:.3f} size={w}x{h} frames={total}")

    if output_path is None:
        folder = os.path.dirname(video_path)
        stem = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(folder, stem + "_skeleton.mp4")

    logger.info(f"Output path | {output_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, float(fps), (w, h))
    if not out.isOpened():
        cap.release()
        raise IOError(f"Could not open VideoWriter for output: {output_path}")

    detector = PersonDetector(model_path=yolo_model_path, conf=conf)

    t0 = time.time()
    last_log = time.time()
    frames_written = 0

    with mp_pose.Pose(
        min_detection_confidence=0.2,
        min_tracking_confidence=0.2,
        model_complexity=2,
        static_image_mode=False,
        smooth_landmarks=True,
    ) as pose:
        idx = -1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1

            roi_box = detect_person_roi(frame, detector, margin=margin)
            lms_full = run_pose_on_frame(frame, pose, roi_box)

            if draw_roi_box and roi_box is not None:
                x1, y1, x2, y2 = roi_box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            if lms_full is not None:
                draw_simple_skeleton(frame, lms_full, mp_pose.POSE_CONNECTIONS)

            if draw_metrics_header:
                draw_outlined_text(
                    frame,
                    "Debug: YOLO-person + MediaPipe pose",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                )
                draw_outlined_text(
                    frame,
                    f"frame {idx+1}/{max(total,1)}",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                )

            out.write(frame)
            frames_written += 1

            # Debug option: stop early
            if max_frames is not None and frames_written >= max_frames:
                logger.info(f"Reached max_frames={max_frames}; stopping early")
                break

            # Progress log every ~2s
            now = time.time()
            if now - last_log >= 2.0:
                elapsed = now - t0
                rate = frames_written / max(elapsed, 1e-6)
                remaining = max(total - (idx + 1), 0)
                eta = remaining / max(rate, 1e-6)
                logger.info(f"Progress | frame={idx+1}/{max(total,1)} | fps_proc={rate:.2f} | eta~{eta:.1f}s")
                last_log = now

    cap.release()
    out.release()
    logger.info(f"Finished skeleton overlay | wrote={frames_written} | out={output_path}")
    
    # Make it playable in Streamlit/browser
    try:
        output_h264 = reencode_h264(output_path, logger=logger)
        logger.info(f"H.264 re-encode complete | out={output_h264}")
        return output_h264
    except Exception as e:
        logger.warning(f"ffmpeg re-encode failed; returning original mp4v. err={e}")
        return output_path



def reencode_h264(in_path: str, logger=None) -> str:
    """
    Re-encode a video to H.264 (browser-friendly) MP4.
    Returns the new output path.
    """
    out_path = os.path.splitext(in_path)[0] + "_h264.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-i", in_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        out_path,
    ]
    if logger:
        logger.info("Re-encoding to H.264 for Streamlit playback…")
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return out_path
