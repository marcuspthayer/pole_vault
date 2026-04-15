"""
Job routes for VaultSense API.
"""

import asyncio
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse

from api.models.job import JobConfig, JobResponse, Pass2Config, StartConfig
from api.services.job_runner import (
    create_job, submit_job, read_job, write_job,
    read_progress, jobs_dir, job_dir, DATA_DIR
)

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB


@router.post("", status_code=202)
async def create_analysis_job(
    video: UploadFile = File(...),
    config: str = Form("{}"),
    auto_start: str = Form("true"),
):
    """Upload a video and queue an analysis job."""
    video_bytes = await video.read()
    if len(video_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Video file too large (max 500MB)")

    try:
        config_dict = json.loads(config)
        JobConfig(**config_dict)  # validate
    except Exception as e:
        raise HTTPException(422, f"Invalid config: {e}")

    should_start = auto_start.lower() not in ("false", "0", "no")
    job = await create_job(video_bytes, video.filename or "input.mp4", config_dict,
                           status="queued" if should_start else "created")
    if should_start:
        await submit_job(job["job_id"])
        job = read_job(job["job_id"])
        if job and job.get("status") == "failed":
            raise HTTPException(503, job.get("error", "Processing failed"))
    return job


@router.post("/{job_id}/start", status_code=202)
async def start_analysis(job_id: str, start_config: StartConfig):
    """Set frame selections and start processing."""
    job = read_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job["status"] != "created":
        raise HTTPException(409, f"Job is not in created state (current: {job['status']})")

    # Merge frame selections into config
    if start_config.start_frame is not None:
        job["config"]["start_frame"] = start_config.start_frame
    if start_config.plant_frame is not None:
        job["config"]["plant_frame"] = start_config.plant_frame
    if start_config.end_frame is not None:
        job["config"]["end_frame"] = start_config.end_frame
    if start_config.crop_before_start:
        job["config"]["crop_before_start"] = True
    if start_config.crop_after_end:
        job["config"]["crop_after_end"] = True

    job["status"] = "queued"
    write_job(job_id, job)
    await submit_job(job_id)
    # Re-read in case submit_job marked it as failed
    job = read_job(job_id)
    if job and job.get("status") == "failed":
        raise HTTPException(503, job.get("error", "Processing failed"))
    return job


@router.get("/{job_id}")
async def get_job_status(job_id: str):
    """Poll job status and results."""
    job = read_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")

    # Merge live progress for running jobs
    if job["status"] in ("queued", "running"):
        prog = read_progress(job_id)
        job["progress"] = prog.get("progress", 0.0)
        job["message"] = prog.get("message", "")

    return job


@router.get("/{job_id}/stream")
async def stream_job_progress(job_id: str):
    """Server-Sent Events stream of job progress."""
    if read_job(job_id) is None:
        raise HTTPException(404, "Job not found")

    async def event_generator():
        last_progress = -1.0
        while True:
            job = read_job(job_id)
            if job is None:
                break

            prog = read_progress(job_id)
            progress = prog.get("progress", 0.0)
            message = prog.get("message", "")

            if progress != last_progress:
                data = json.dumps({"progress": progress, "message": message, "status": job["status"]})
                yield f"data: {data}\n\n"
                last_progress = progress

            if job["status"] in ("complete", "failed", "pass1_done"):
                # Send final event with full job data
                yield f"data: {json.dumps(job)}\n\n"
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{job_id}/frame")
async def get_frame(job_id: str, frame_idx: int = 0, source: str = "input"):
    """Extract a single frame from the video as JPEG. source=input|output."""
    jdir = jobs_dir() / job_id
    video_file = "output.mp4" if source == "output" else "input.mp4"
    input_path = jdir / video_file
    if not input_path.exists():
        raise HTTPException(404, f"Video '{video_file}' not found (may have been cleaned up)")

    from api.services.job_runner import _find_ffmpeg
    ffmpeg_exe = _find_ffmpeg()
    frame_idx = max(0, frame_idx)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name

    cmd = [
        ffmpeg_exe, "-y",
        "-loglevel", "error",
        "-i", str(input_path),
        "-vf", f"select=eq(n\\,{frame_idx}),scale='min(1280,iw)':-2",
        "-vframes", "1",
        "-q:v", "4",
        tmp_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    if result.returncode != 0 or not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise HTTPException(500, f"Could not read frame {frame_idx}")

    return FileResponse(tmp_path, media_type="image/jpeg")


@router.post("/{job_id}/pass2", status_code=202)
async def submit_pass2(job_id: str, pass2: Pass2Config):
    """Submit manual pole frame selections and kick off Pass 2."""
    job = read_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job["status"] != "pass1_done":
        raise HTTPException(409, f"Job is not in pass1_done state (current: {job['status']})")

    job["pass2_config"] = pass2.model_dump()
    job["status"] = "queued"
    write_job(job_id, job)
    await submit_job(job_id)
    # Re-read in case submit_job marked it as failed
    job = read_job(job_id)
    if job and job.get("status") == "failed":
        raise HTTPException(503, job.get("error", "Processing failed"))
    return job


@router.get("/{job_id}/results/{filename}")
async def download_result(job_id: str, filename: str):
    """Download a result file (output.mp4, gait_data.csv, velocity_data.csv, debug/*.jpg)."""
    # Sanitize filename to prevent path traversal
    if ".." in filename or filename.startswith("/"):
        raise HTTPException(400, "Invalid filename")

    jdir = jobs_dir() / job_id
    candidates = [
        jdir / filename,
        jdir / "debug" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            media_type = "video/mp4" if filename.endswith(".mp4") else (
                "text/csv" if filename.endswith(".csv") else "image/jpeg"
            )
            return FileResponse(str(candidate), media_type=media_type, filename=filename)

    raise HTTPException(404, "Result file not found")


@router.delete("/{job_id}", status_code=204)
async def delete_job(job_id: str):
    """Delete a job and all its files."""
    jdir = jobs_dir() / job_id
    if not jdir.exists():
        raise HTTPException(404, "Job not found")
    import shutil
    shutil.rmtree(jdir, ignore_errors=True)
    return None


@router.delete("", status_code=200)
async def cleanup_all_jobs():
    """Delete ALL jobs and free disk space."""
    import shutil
    jdir = jobs_dir()
    count = 0
    for d in jdir.iterdir():
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
            count += 1
    return {"deleted": count}


@router.get("/health", tags=["health"])
async def health():
    return {"status": "ok"}
