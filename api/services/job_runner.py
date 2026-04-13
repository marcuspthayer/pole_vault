"""
Background job runner for VaultSense video analysis.

Wraps run_unified_pipeline() in a ProcessPoolExecutor so FastAPI stays
responsive while heavy video processing runs in a subprocess.
"""

import asyncio
import json
import os
import pickle
import subprocess
import time
import uuid
import logging
from concurrent.futures import ProcessPoolExecutor, BrokenExecutor
from pathlib import Path
from typing import Optional

logger = logging.getLogger("vaultsense.job_runner")

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
JOBS_DIR = DATA_DIR / "jobs"
JOB_TTL_SECONDS = 1800  # 30 minutes — enough time for both passes + viewing results

MAX_FPS = 60  # Downsample anything above this at upload time

def _find_ffmpeg() -> str:
    """Return path to ffmpeg binary, preferring imageio_ffmpeg then system."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        import shutil
        path = shutil.which("ffmpeg")
        if path:
            return path
        raise FileNotFoundError("No ffmpeg binary found (imageio_ffmpeg or system)")


def _downsample_if_needed(input_path: Path) -> Optional[float]:
    """If video fps > MAX_FPS, re-encode to MAX_FPS using ffmpeg.

    Returns the original fps if downsampling occurred, else None.
    The input file is replaced in-place on success.
    """
    try:
        import cv2
        cap = cv2.VideoCapture(str(input_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        cap.release()

        if fps <= MAX_FPS:
            return None

        logger.info("Video is %.1ffps — downsampling to %dfps", fps, MAX_FPS)
        ffmpeg_exe = _find_ffmpeg()
        temp_path = input_path.with_name("input_60fps.mp4")

        cmd = [
            ffmpeg_exe, "-y",
            "-i", str(input_path),
            "-vf", f"fps={MAX_FPS}",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(temp_path),
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, timeout=120)

        if result.returncode == 0 and temp_path.exists():
            os.remove(input_path)
            os.rename(temp_path, input_path)
            logger.info("Downsampled %.1ffps → %dfps: %s", fps, MAX_FPS, input_path.name)
            return fps
        else:
            logger.warning("ffmpeg downsample failed (rc=%d): %s",
                           result.returncode, result.stderr[:500])
            if temp_path.exists():
                os.remove(temp_path)
            return None
    except Exception as e:
        logger.warning("Downsample skipped: %s", e)
        return None


def jobs_dir() -> Path:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    return JOBS_DIR


def job_dir(job_id: str) -> Path:
    d = jobs_dir() / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_job(job_id: str) -> Optional[dict]:
    p = jobs_dir() / job_id / "job.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def write_job(job_id: str, data: dict) -> None:
    p = job_dir(job_id) / "job.json"
    with open(p, "w") as f:
        json.dump(data, f)


def write_progress(job_id: str, progress: float, message: str) -> None:
    p = job_dir(job_id) / "progress.json"
    with open(p, "w") as f:
        json.dump({"progress": progress, "message": message}, f)


def read_progress(job_id: str) -> dict:
    p = jobs_dir() / job_id / "progress.json"
    if not p.exists():
        return {"progress": 0.0, "message": ""}
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError):
        return {"progress": 0.0, "message": ""}


def _run_pipeline_subprocess(job_id: str, data_dir: str) -> dict:
    """
    Runs inside the ProcessPoolExecutor subprocess.
    Returns a dict of result metrics and file flags.
    """
    import sys
    import os as _os
    # Ensure the repo root is on the path inside the subprocess
    repo_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from pathlib import Path as _Path
    import json as _json
    import pickle as _pickle

    jobs_dir_ = _Path(data_dir) / "jobs"
    jdir = jobs_dir_ / job_id
    job_file = jdir / "job.json"
    progress_file = jdir / "progress.json"

    def _write_progress(p, msg):
        with open(progress_file, "w") as f:
            _json.dump({"progress": p, "message": msg}, f)

    def _write_job(d):
        with open(job_file, "w") as f:
            _json.dump(d, f)

    with open(job_file) as f:
        job = _json.load(f)

    config = job["config"]
    job["status"] = "running"
    _write_job(job)
    _write_progress(0.0, "Starting...")

    try:
        from pvapp.pipelines.unified_pipeline import run_unified_pipeline

        model_dir = _Path(repo_root)
        pole_model = str(model_dir / "pole_detect_v3.pt")

        input_path = str(jdir / "input.mp4")
        output_path = str(jdir / "output.mp4")
        debug_dir = str(jdir / "debug")
        _Path(debug_dir).mkdir(exist_ok=True)

        precomputed_pose = None
        precomputed_pole = None
        pkl_path = jdir / "precomputed.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                precomputed = _pickle.load(f)
            precomputed_pose = precomputed.get("pose")
            precomputed_pole = precomputed.get("pole")

        manual_pole_frames = None
        if "pass2_config" in job:
            p2 = job["pass2_config"]
            manual_pole_frames = {
                "phase1": p2["phase1_frame"],
                "phase2": p2["phase2_frame"],
                "plant": p2["plant_frame"],
                "max_bend": (p2["bend_start_frame"], p2["bend_end_frame"]),
            }

        skip_pole_metrics = config.get("enable_manual_pole_frames", False) and precomputed_pose is None

        result = run_unified_pipeline(
            video_path=input_path,
            output_path=output_path,
            pole_model_path=pole_model,
            pole_conf=config.get("pole_conf", 0.25),
            enable_pose=config.get("enable_skeleton", True),
            enable_hip=config.get("enable_hip", False),
            enable_step=config.get("enable_step", True),
            enable_max_hip_height=config.get("enable_max_hip_height", True),
            enable_pole=config.get("enable_pole", True),
            start_frame=config.get("start_frame"),
            plant_frame=config.get("plant_frame"),
            end_frame=config.get("end_frame"),
            athlete_height_m=config.get("athlete_height_m", 1.70),
            progress_callback=_write_progress,
            pole_length_m=config.get("pole_length_m"),
            skip_pole_metrics=skip_pole_metrics,
            manual_pole_frames=manual_pole_frames,
            precomputed_pose=precomputed_pose,
            precomputed_pole=precomputed_pole,
            debug_dir=debug_dir,
            crop_before_start=config.get("crop_before_start", False),
        )

        # result is an 11-tuple:
        # (output_path, stride_data_list, calib_px, ankles, bend_data,
        #  height_scale, max_hip_height_data, pose_results, pole_results, velocity_data, cadence_val)
        (out_vid, stride_data, calib_px, ankles, bend_data,
         height_scale, max_hip_height_data, pose_results, pole_results, velocity_data, cadence_val) = result

        # --- Pass 1 complete: save precomputed results for Pass 2 ---
        if skip_pole_metrics:
            with open(pkl_path, "wb") as f:
                _pickle.dump({"pose": pose_results, "pole": pole_results}, f)
            job["status"] = "pass1_done"
            _write_job(job)
            _write_progress(1.0, "Pass 1 complete — select pole frames")
            return {"status": "pass1_done"}

        # --- Export stride data CSV ---
        result_files = {"video": False, "gait_csv": False, "velocity_csv": False, "debug_images": []}

        if _Path(output_path).exists():
            result_files["video"] = True

        if stride_data:
            import pandas as pd
            gait_path = str(jdir / "gait_data.csv")
            pd.DataFrame(stride_data).to_csv(gait_path, index=False)
            result_files["gait_csv"] = True

        if velocity_data and len(velocity_data) > 0:
            import pandas as pd
            vel_path = str(jdir / "velocity_data.csv")
            pd.DataFrame(velocity_data).to_csv(vel_path, index=False)
            result_files["velocity_csv"] = True

        # Collect debug images
        debug_p = _Path(debug_dir)
        if debug_p.exists():
            result_files["debug_images"] = [f.name for f in debug_p.glob("*.jpg")]

        # --- Extract summary metrics ---
        metrics = {}
        if cadence_val is not None:
            metrics["cadence_spm"] = round(cadence_val, 1)

        if max_hip_height_data:
            mhh = max_hip_height_data.get("height_m")
            if mhh:
                metrics["max_hip_height_m"] = round(mhh, 3)
                pc = max_hip_height_data.get("predicted_clear_m")
                if pc:
                    metrics["predicted_clear_m"] = round(pc, 3)
                    metrics["predicted_clear_in"] = round(pc * 39.3701, 1)
            peak_frame = max_hip_height_data.get("peak_frame")
            p_frame = config.get("plant_frame")
            job_fps = job.get("fps", 30.0)
            if peak_frame is not None and p_frame is not None and job_fps > 0:
                metrics["plant_to_peak_s"] = round((peak_frame - p_frame) / job_fps, 2)

        if bend_data and isinstance(bend_data, dict):
            # bend_data is a dict with 'max_bend' (smoothed %), 'poly_max_bend', 'bend_series', etc.
            max_bend_val = bend_data.get("poly_max_bend") or bend_data.get("max_bend")
            if max_bend_val is not None:
                metrics["max_pole_bend_pct"] = round(max_bend_val, 1)

        if velocity_data and len(velocity_data) > 0:
            vels = [v.get("velocity_m_s", 0) for v in velocity_data if v.get("velocity_m_s")]
            if vels:
                metrics["peak_velocity_ms"] = round(max(vels), 2)
                metrics["avg_velocity_ms"] = round(sum(vels) / len(vels), 2)
                metrics["takeoff_velocity_ms"] = round(vels[-1], 2) if vels else None

        job["status"] = "complete"
        job["metrics"] = metrics
        job["result_files"] = result_files
        _write_job(job)
        _write_progress(1.0, "Analysis complete")

        # Clean up large intermediate files immediately — keep output.mp4 for user to view/download
        for large_file in ["input.mp4", "precomputed.pkl"]:
            p = _Path(str(jdir / large_file))
            if p.exists():
                p.unlink()

        return {"status": "complete", "metrics": metrics, "result_files": result_files}

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        job["status"] = "failed"
        job["error"] = str(e)
        _write_job(job)
        _write_progress(0.0, f"Failed: {e}")
        return {"status": "failed", "error": str(e), "traceback": err}


def _ensure_disk_space() -> None:
    """Delete ALL finished job dirs if disk is critically low (<100 MB free)."""
    import shutil
    try:
        usage = shutil.disk_usage(str(JOBS_DIR))
        free_mb = usage.free / (1024 * 1024)
        if free_mb < 100:
            logger.warning("Disk critically low (%.0f MB free) — purging all finished jobs", free_mb)
            for jdir in JOBS_DIR.iterdir():
                if not jdir.is_dir():
                    continue
                job_file = jdir / "job.json"
                if not job_file.exists():
                    shutil.rmtree(jdir, ignore_errors=True)
                    continue
                try:
                    with open(job_file) as f:
                        job = json.load(f)
                    if job.get("status") in ("complete", "failed", "pass1_done"):
                        shutil.rmtree(jdir, ignore_errors=True)
                except Exception:
                    shutil.rmtree(jdir, ignore_errors=True)
    except Exception as e:
        logger.warning("Disk space check failed: %s", e)


def _detect_start_frame(video_path: str, max_check: int = 50, stride: int = 10) -> Optional[int]:
    """Scan early frames to find the first one containing a person (YOLO class 0)."""
    try:
        import cv2 as _cv2
        from ultralytics import YOLO
        model = YOLO("yolo11n.pt")
        cap = _cv2.VideoCapture(video_path)
        total = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))

        for i in range(0, min(total, max_check * stride), stride):
            cap.set(_cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(source=frame, conf=0.15, imgsz=320, verbose=False)
            if len(results) > 0 and len(results[0].boxes) > 0:
                import numpy as np
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                if 0 in classes:
                    cap.release()
                    del model
                    return max(0, i)
        cap.release()
        del model
        return None
    except Exception as e:
        logger.warning("Auto start-frame detection failed: %s", e)
        return None


async def create_job(video_bytes: bytes, filename: str, config: dict, status: str = "queued") -> dict:
    """
    Save uploaded video and create job record. Returns job metadata dict.
    """
    _ensure_disk_space()

    job_id = str(uuid.uuid4())
    jdir = job_dir(job_id)

    # Save video
    input_path = jdir / "input.mp4"
    with open(input_path, "wb") as f:
        f.write(video_bytes)

    # Downsample high-fps videos (e.g. 120/240fps slow-mo) to 60fps
    original_fps = _downsample_if_needed(input_path)

    # Read video metadata (reflects downsampled file if applicable)
    import cv2
    cap = cv2.VideoCapture(str(input_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Quick YOLO scan to suggest a start frame (first frame with a person)
    suggested_start = _detect_start_frame(str(input_path))

    job = {
        "job_id": job_id,
        "status": status,
        "progress": 0.0,
        "message": "",
        "config": config,
        "total_frames": total_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "original_fps": original_fps,
        "suggested_start_frame": suggested_start,
        "created_at": time.time(),
        "metrics": None,
        "result_files": None,
        "error": None,
    }
    write_job(job_id, job)
    return job


async def submit_job(job_id: str) -> None:
    """Enqueue the job for background processing.

    Uses a fresh ProcessPoolExecutor per job and shuts it down afterward
    so the subprocess (and its ~3.5 GB of ML models) is freed from memory.
    """
    loop = asyncio.get_event_loop()
    executor = ProcessPoolExecutor(max_workers=1)
    try:
        await loop.run_in_executor(
            executor,
            _run_pipeline_subprocess,
            job_id,
            str(DATA_DIR),
        )
    except BrokenExecutor:
        # Subprocess was killed (likely OOM). Mark the job as failed so the
        # user sees a clear error instead of a raw 500.
        logger.error("Job %s: subprocess killed (possible OOM)", job_id)
        job = read_job(job_id)
        if job and job["status"] not in ("completed", "failed"):
            job["status"] = "failed"
            job["error"] = "Processing failed: out of memory. Try a shorter or lower-resolution video."
            write_job(job_id, job)
    finally:
        executor.shutdown(wait=True)


def _cleanup_once() -> None:
    """Delete entire job directories that are expired or orphaned."""
    import shutil
    now = time.time()
    if not JOBS_DIR.exists():
        return
    for jdir in JOBS_DIR.iterdir():
        if not jdir.is_dir():
            continue
        job_file = jdir / "job.json"
        if not job_file.exists():
            # Orphaned directory with no job.json — delete it
            shutil.rmtree(jdir, ignore_errors=True)
            logger.info("Deleted orphaned job dir %s", jdir.name)
            continue
        try:
            with open(job_file) as f:
                job = json.load(f)
        except (json.JSONDecodeError, ValueError):
            shutil.rmtree(jdir, ignore_errors=True)
            logger.info("Deleted corrupt job dir %s", jdir.name)
            continue
        status = job.get("status", "")
        created = job.get("created_at", now)
        # Mark stuck running jobs as failed after TTL
        if status in ("running", "queued") and now - created > JOB_TTL_SECONDS:
            job["status"] = "failed"
            job["error"] = "Processing timed out — video may be too large or high-FPS for this server"
            with open(job_file, "w") as wf:
                json.dump(job, wf)
            status = "failed"
        # Only clean up completed or failed jobs — never in-progress ones
        if status not in ("complete", "failed"):
            continue
        if now - created > JOB_TTL_SECONDS:
            shutil.rmtree(jdir, ignore_errors=True)
            logger.info("Deleted expired job dir %s", jdir.name)


async def cleanup_old_jobs() -> None:
    """Run cleanup on startup, then periodically."""
    # Immediate cleanup on startup to reclaim disk from prior crashes
    try:
        _cleanup_once()
    except Exception as e:
        logger.warning("Startup cleanup error: %s", e)
    while True:
        await asyncio.sleep(120)  # check every 2 minutes
        try:
            _cleanup_once()
        except Exception as e:
            logger.warning("Cleanup error: %s", e)
