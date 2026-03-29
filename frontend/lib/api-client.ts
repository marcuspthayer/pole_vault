import type { JobConfig, JobResponse, Pass2Config } from './types';

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? 'https://api.vaultsense.app';

export const apiClient = {
  async createJob(video: File, config: JobConfig): Promise<JobResponse> {
    const form = new FormData();
    form.append('video', video);
    form.append('config', JSON.stringify(config));
    const res = await fetch(`${API_BASE}/api/jobs`, { method: 'POST', body: form });
    if (!res.ok) throw new Error(`Failed to create job: ${res.statusText}`);
    return res.json();
  },

  async getJob(jobId: string): Promise<JobResponse> {
    const res = await fetch(`${API_BASE}/api/jobs/${jobId}`);
    if (!res.ok) throw new Error(`Failed to get job: ${res.statusText}`);
    return res.json();
  },

  async submitPass2(jobId: string, config: Pass2Config): Promise<JobResponse> {
    const res = await fetch(`${API_BASE}/api/jobs/${jobId}/pass2`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    if (!res.ok) throw new Error(`Failed to submit pass2: ${res.statusText}`);
    return res.json();
  },

  async deleteJob(jobId: string): Promise<void> {
    await fetch(`${API_BASE}/api/jobs/${jobId}`, { method: 'DELETE' });
  },

  getFrameUrl(jobId: string, frameIdx: number): string {
    return `${API_BASE}/api/jobs/${jobId}/frame?frame_idx=${frameIdx}`;
  },

  getResultFileUrl(jobId: string, filename: string): string {
    return `${API_BASE}/api/jobs/${jobId}/results/${filename}`;
  },

  getStreamUrl(jobId: string): string {
    return `${API_BASE}/api/jobs/${jobId}/stream`;
  },
};
