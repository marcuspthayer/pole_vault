"""
Job routes for VaultSense API.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Optional

import cv2
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse

from api.models.job import JobConfig, JobResponse, Pass2Config
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

    job = await create_job(video_bytes, video.filename or "input.mp4", config_dict)
    await submit_job(job["job_id"])
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
async def get_frame(job_id: str, frame_idx: int = 0):
    """Extract a single frame from the uploaded video as JPEG."""
    jdir = jobs_dir() / job_id
    input_path = jdir / "input.mp4"
    if not input_path.exists():
        raise HTTPException(404, "Video not found (may have been cleaned up)")

    cap = cv2.VideoCapture(str(input_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = max(0, min(frame_idx, total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise HTTPException(500, "Could not read frame")

    # Scale down for UI if large
    h, w = frame.shape[:2]
    if w > 1280:
        scale = 1280 / w
        frame = cv2.resize(frame, (1280, int(h * scale)))

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        tmp_path = tmp.name

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


@router.get("/health", tags=["health"])
async def health():
    return {"status": "ok"}
