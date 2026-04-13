from pydantic import BaseModel
from typing import Optional
from enum import Enum


class JobStatus(str, Enum):
    created = "created"
    queued = "queued"
    running = "running"
    pass1_done = "pass1_done"
    complete = "complete"
    failed = "failed"


class StartConfig(BaseModel):
    start_frame: Optional[int] = None
    plant_frame: Optional[int] = None
    end_frame: Optional[int] = None
    crop_before_start: bool = False


class JobConfig(BaseModel):
    athlete_height_m: float = 1.70
    start_frame: Optional[int] = None
    plant_frame: Optional[int] = None
    end_frame: Optional[int] = None
    enable_skeleton: bool = True
    enable_hip: bool = False
    enable_step: bool = True
    enable_max_hip_height: bool = True
    enable_pole: bool = True
    enable_manual_pole_frames: bool = False
    pole_conf: float = 0.25
    pole_length_m: Optional[float] = None


class Pass2Config(BaseModel):
    phase1_frame: int
    phase2_frame: int
    plant_frame: int
    bend_start_frame: int
    bend_end_frame: int


class JobMetrics(BaseModel):
    cadence_spm: Optional[float] = None
    max_hip_height_m: Optional[float] = None
    max_pole_bend_pct: Optional[float] = None
    peak_velocity_ms: Optional[float] = None
    avg_velocity_ms: Optional[float] = None
    takeoff_velocity_ms: Optional[float] = None
    predicted_clear_m: Optional[float] = None
    predicted_clear_in: Optional[float] = None
    plant_to_peak_s: Optional[float] = None


class JobResultFiles(BaseModel):
    video: bool = False
    gait_csv: bool = False
    velocity_csv: bool = False
    debug_images: list[str] = []


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: float = 0.0
    message: str = ""
    error: Optional[str] = None
    total_frames: Optional[int] = None
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    metrics: Optional[JobMetrics] = None
    result_files: Optional[JobResultFiles] = None
