import type { JobConfig, JobResponse, Pass2Config, StartConfig } from './types';

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? 'https://api.vaultsense.app';

export const apiClient = {
  async createJob(video: File, config: JobConfig, autoStart = true): Promise<JobResponse> {
    const form = new FormData();
    form.append('video', video);
    form.append('config', JSON.stringify(config));
    form.append('auto_start', autoStart ? 'true' : 'false');
    const res = await fetch(`${API_BASE}/api/jobs`, { method: 'POST', body: form });
    if (!res.ok) throw new Error(`Failed to create job: ${res.statusText}`);
    return res.json();
  },

  async startJob(jobId: string, config: StartConfig): Promise<JobResponse> {
    const res = await fetch(`${API_BASE}/api/jobs/${jobId}/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    if (!res.ok) throw new Error(`Failed to start job: ${res.statusText}`);
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

  async fetchCsv<T>(jobId: string, filename: string): Promise<T[]> {
    const res = await fetch(`${API_BASE}/api/jobs/${jobId}/results/${filename}`);
    if (!res.ok) return [];
    const text = await res.text();
    const lines = text.trim().split('\n');
    if (lines.length < 2) return [];
    const headers = lines[0].split(',');
    return lines.slice(1).map(line => {
      const vals = line.split(',');
      const obj: Record<string, unknown> = {};
      headers.forEach((h, i) => {
        const v = vals[i];
        obj[h] = v === '' ? null : isNaN(Number(v)) ? v : Number(v);
      });
      return obj as T;
    });
  },

  getFrameUrl(jobId: string, frameIdx: number, source: 'input' | 'output' = 'input'): string {
    return `${API_BASE}/api/jobs/${jobId}/frame?frame_idx=${frameIdx}&source=${source}`;
  },

  getResultFileUrl(jobId: string, filename: string): string {
    return `${API_BASE}/api/jobs/${jobId}/results/${filename}`;
  },

  getStreamUrl(jobId: string): string {
    return `${API_BASE}/api/jobs/${jobId}/stream`;
  },
};
