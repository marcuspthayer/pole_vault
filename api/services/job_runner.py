"""
Background job runner for VaultSense video analysis.

Wraps run_unified_pipeline() in a ProcessPoolExecutor so FastAPI stays
responsive while heavy video processing runs in a subprocess.
"""

import asyncio
import json
import os
import pickle
import time
import uuid
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

logger = logging.getLogger("vaultsense.job_runner")

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
JOBS_DIR = DATA_DIR / "jobs"
JOB_TTL_SECONDS = 7200  # 2 hours

# Single global executor — 1 worker to stay within RAM budget
_executor = ProcessPoolExecutor(max_workers=1)


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
    with open(p) as f:
        return json.load(f)


def _run_pipeline_subprocess(job_id: str, data_dir: str) -> dict:
    """
    Runs inside the ProcessPoolExecutor subprocess.
    Returns a dict of result metrics and file flags.
    """
    import sys
    import os as _os
    # Ensure the repo root is on the path inside the subprocess
    repo_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
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
                "bend_start": p2["bend_start_frame"],
                "bend_end": p2["bend_end_frame"],
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
            step_min_lift=config.get("step_min_lift", 0.015),
            step_min_dist=config.get("step_min_dist"),
            pole_length_m=config.get("pole_length_m"),
            enable_ml_steps=config.get("enable_ml_steps", False),
            skip_pole_metrics=skip_pole_metrics,
            manual_pole_frames=manual_pole_frames,
            precomputed_pose=precomputed_pose,
            precomputed_pole=precomputed_pole,
        )

        # result is a 10-tuple:
        # (output_path, stride_data_list, calib_px, ankles, bend_data,
        #  height_scale, max_hip_height_data, pose_results, pole_results, velocity_data)
        (out_vid, stride_data, calib_px, ankles, bend_data,
         height_scale, max_hip_height_data, pose_results, pole_results, velocity_data) = result

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
        if stride_data:
            cadences = [s.get("cadence_spm") for s in stride_data if s.get("cadence_spm")]
            if cadences:
                metrics["cadence_spm"] = round(sum(cadences) / len(cadences), 1)

        if max_hip_height_data:
            mhh = max_hip_height_data.get("max_hip_height_m")
            if mhh:
                metrics["max_hip_height_m"] = round(mhh, 3)

        if bend_data:
            max_bend = max((b.get("chord_ratio", 0) for b in bend_data), default=None)
            if max_bend:
                metrics["max_pole_bend_pct"] = round(max_bend * 100, 1)

        if velocity_data and len(velocity_data) > 0:
            vels = [v.get("velocity_ms", 0) for v in velocity_data if v.get("velocity_ms")]
            if vels:
                metrics["peak_velocity_ms"] = round(max(vels), 2)
                metrics["avg_velocity_ms"] = round(sum(vels) / len(vels), 2)
                metrics["takeoff_velocity_ms"] = round(vels[-1], 2) if vels else None

        job["status"] = "complete"
        job["metrics"] = metrics
        job["result_files"] = result_files
        _write_job(job)
        _write_progress(1.0, "Analysis complete")

        # Clean up input video to save disk space
        input_p = _Path(input_path)
        if input_p.exists():
            input_p.unlink()

        return {"status": "complete", "metrics": metrics, "result_files": result_files}

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        job["status"] = "failed"
        job["error"] = str(e)
        _write_job(job)
        _write_progress(0.0, f"Failed: {e}")
        return {"status": "failed", "error": str(e), "traceback": err}


async def create_job(video_bytes: bytes, filename: str, config: dict) -> dict:
    """
    Save uploaded video and create job record. Returns job metadata dict.
    """
    job_id = str(uuid.uuid4())
    jdir = job_dir(job_id)

    # Save video
    input_path = jdir / "input.mp4"
    with open(input_path, "wb") as f:
        f.write(video_bytes)

    # Read video metadata
    import cv2
    cap = cv2.VideoCapture(str(input_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    job = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0.0,
        "message": "",
        "config": config,
        "total_frames": total_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "created_at": time.time(),
        "metrics": None,
        "result_files": None,
        "error": None,
    }
    write_job(job_id, job)
    return job


async def submit_job(job_id: str) -> None:
    """Enqueue the job for background processing."""
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        _executor,
        _run_pipeline_subprocess,
        job_id,
        str(DATA_DIR),
    )


async def cleanup_old_jobs() -> None:
    """Periodic task: delete jobs older than JOB_TTL_SECONDS."""
    while True:
        await asyncio.sleep(600)  # check every 10 minutes
        now = time.time()
        try:
            for jdir in JOBS_DIR.iterdir():
                job_file = jdir / "job.json"
                if not job_file.exists():
                    continue
                with open(job_file) as f:
                    job = json.load(f)
                created = job.get("created_at", now)
                if now - created > JOB_TTL_SECONDS:
                    import shutil
                    shutil.rmtree(jdir, ignore_errors=True)
                    logger.info(f"Cleaned up expired job {jdir.name}")
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")
