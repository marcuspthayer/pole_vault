import os
import time
import logging
import cv2
import subprocess

from pv_hip_analysis import compute_hip_time_series, compute_hip_drop
from pv_yolo_utils import draw_outlined_text
from pvapp.render import draw_simple_skeleton
import mediapipe as mp

mp_pose = mp.solutions.pose
logger = logging.getLogger("pv.pipeline.hip")


def reencode_h264(in_path: str) -> str:
    out_path = os.path.splitext(in_path)[0] + "_h264.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-i", in_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        out_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return out_path


def process_video_hip_overlay(
    video_path: str,
    output_path: str | None = None,
    *,
    yolo_model_path: str = "yolo11n.pt",
    conf: float = 0.25,
    draw_roi_box: bool = True,
    draw_metrics_header: bool = True,
    log_every_sec: float = 2.0,
) -> tuple[str, dict]:
    """
    Returns:
      (output_video_path, metrics_dict)
    """

    # ---------- PASS 1: compute hip series + store landmarks/roi ----------
    t0 = time.time()
    hip_y_arr, body_h_arr, fps, total_frames, pose_landmarks_list, roi_box_list = \
        compute_hip_time_series(video_path, yolo_model_path=yolo_model_path, conf=conf)

    hip_droop_pct, hip_droop_trend_pct, n_valid, worst_frames, start_idx, end_idx = \
        compute_hip_drop(hip_y_arr, body_h_arr)

    metrics = {
        "hip_droop_pct": hip_droop_pct,
        "hip_droop_trend_pct": hip_droop_trend_pct,
        "n_valid": int(n_valid),
        "start_idx": int(start_idx),
        "end_idx": int(end_idx),
        "worst_frames": list(worst_frames),
        "fps": float(fps) if fps else None,
        "total_frames": int(total_frames),
    }

    # ---------- PASS 2: render output video ----------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv2.CAP_PROP_FPS) or (fps or 30.0)
    total2 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if output_path is None:
        folder = os.path.dirname(video_path)
        stem = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(folder, stem + "_hip.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, float(fps_video), (w, h))
    if not out.isOpened():
        cap.release()
        raise IOError(f"Could not open VideoWriter for output: {output_path}")

    worst_set = set(worst_frames) if hip_droop_pct is not None else set()
    hip_points_normal = []
    hip_points_worst = []

    last_log = time.time()
    frames_written = 0
    frame_idx = -1

    # Text for overlay
    text_lines = []
    if hip_droop_pct is not None:
        text_lines.append(f"Hip droop (worst 5% late): {hip_droop_pct:+.1f}% height")
        text_lines.append(f"Hip trend (late vs early): {hip_droop_trend_pct:+.1f}% height")
        text_lines.append(f"Valid frames: {n_valid}")
    else:
        text_lines.append("Hip droop: N/A (insufficient pose)")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        roi_box = roi_box_list[frame_idx] if 0 <= frame_idx < len(roi_box_list) else None
        lms_full = pose_landmarks_list[frame_idx] if 0 <= frame_idx < len(pose_landmarks_list) else None

        # ROI box
        if draw_roi_box and roi_box is not None:
            x1, y1, x2, y2 = roi_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Skeleton + hip point
        hip_pt = None
        if lms_full is not None:
            draw_simple_skeleton(frame, lms_full, mp_pose.POSE_CONNECTIONS)

            lm = lms_full.landmark
            lhip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
            rhip = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]
            hx = int(0.5 * (lhip.x + rhip.x) * w)
            hy = int(0.5 * (lhip.y + rhip.y) * h)
            hip_pt = (hx, hy)

        # Persist hip path only inside analysis window
        if hip_pt is not None and start_idx <= frame_idx < end_idx:
            if frame_idx in worst_set:
                hip_points_worst.append(hip_pt)
            else:
                hip_points_normal.append(hip_pt)

        for (hx, hy) in hip_points_normal:
            cv2.circle(frame, (hx, hy), 4, (255, 255, 0), -1)   # cyan-ish
        for (hx, hy) in hip_points_worst:
            cv2.circle(frame, (hx, hy), 6, (0, 0, 255), -1)     # red

        # Metrics header text
        if draw_metrics_header:
            y0 = 50
            for i, txt in enumerate(text_lines):
                draw_outlined_text(
                    frame, txt, (10, y0 + i * 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (0, 255, 0), 3
                )

        out.write(frame)
        frames_written += 1

        now = time.time()
        if now - last_log >= log_every_sec:
            elapsed = now - t0
            rate = frames_written / max(elapsed, 1e-6)
            eta = (total2 - frames_written) / max(rate, 1e-6)
            logger.info(f"Progress | frame={frame_idx+1}/{total2} | fps_proc={rate:.2f} | eta~{eta:.1f}s")
            last_log = now

    cap.release()
    out.release()

    # Re-encode to browser-friendly H264 so Streamlit always displays it
    try:
        output_h264 = reencode_h264(output_path)
        return output_h264, metrics
    except Exception as e:
        logger.warning(f"ffmpeg re-encode failed; returning mp4v. err={e}")
        return output_path, metrics
