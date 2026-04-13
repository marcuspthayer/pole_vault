// Mirrors api/models/job.py — keep in sync

export type JobStatus = 'created' | 'queued' | 'running' | 'pass1_done' | 'complete' | 'failed';

export interface StartConfig {
  start_frame?: number;
  plant_frame?: number;
  end_frame?: number;
  crop_before_start?: boolean;
}

export interface JobConfig {
  athlete_height_m?: number;
  start_frame?: number;
  plant_frame?: number;
  end_frame?: number;
  enable_skeleton?: boolean;
  enable_pole?: boolean;
  enable_step?: boolean;
  enable_max_hip_height?: boolean;
  enable_manual_pole_frames?: boolean;
  pole_length_m?: number;
  pole_conf?: number;
}

export interface Pass2Config {
  phase1_frame: number;
  phase2_frame: number;
  plant_frame: number;
  bend_start_frame: number;
  bend_end_frame: number;
}

export interface JobMetrics {
  cadence_spm?: number;
  max_hip_height_m?: number;
  max_pole_bend_pct?: number;
  peak_velocity_ms?: number;
  avg_velocity_ms?: number;
  takeoff_velocity_ms?: number;
  predicted_clear_m?: number;
  predicted_clear_in?: number;
  plant_to_peak_s?: number;
}

// Data rows from CSV downloads
export interface VelocityRow {
  frame: number;
  time_s: number;
  velocity_m_s: number;
  velocity_mph: number;
  hip_x_px: number;
}

export interface StrideRow {
  frame: number;
  stride_norm: number;
  frames_dur: number;
  side: 'left' | 'right';
  cadence_spm?: number;
  stride_cm?: number;
  stride_in?: number;
}

export interface JobResultFiles {
  video: boolean;
  gait_csv: boolean;
  velocity_csv: boolean;
  debug_images: string[];
}

export interface JobResponse {
  job_id: string;
  status: JobStatus;
  progress: number;
  message: string;
  error?: string;
  total_frames?: number;
  fps?: number;
  width?: number;
  height?: number;
  suggested_start_frame?: number;
  metrics?: JobMetrics;
  result_files?: JobResultFiles;
}

// Supabase DB types
export interface Profile {
  id: string;
  full_name: string | null;
  role: 'coach' | 'athlete';
  coach_id: string | null;
  created_at: string;
}

export interface Session {
  id: string;
  athlete_id: string;
  label: string | null;
  job_id: string | null;
  metrics: JobMetrics;
  config: JobConfig | null;
  created_at: string;
}
