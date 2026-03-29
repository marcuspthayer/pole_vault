'use client';

import { useState, useReducer } from 'react';
import { VideoUploader } from '@/components/analyze/VideoUploader';
import { AnalysisProgress } from '@/components/analyze/AnalysisProgress';
import { FrameSelector } from '@/components/analyze/FrameSelector';
import { MetricsGrid } from '@/components/results/MetricsGrid';
import { useJobPoll } from '@/hooks/useJobPoll';
import { apiClient } from '@/lib/api-client';
import type { JobConfig, JobResponse, Pass2Config } from '@/lib/types';
import Link from 'next/link';

type Stage = 'idle' | 'uploading' | 'pass1' | 'awaiting_frames' | 'pass2' | 'complete' | 'failed';

interface State {
  stage: Stage;
  jobId: string | null;
  error: string | null;
}

type Action =
  | { type: 'UPLOAD_START' }
  | { type: 'JOB_CREATED'; jobId: string }
  | { type: 'PASS1_DONE' }
  | { type: 'PASS2_START' }
  | { type: 'COMPLETE' }
  | { type: 'FAIL'; error: string }
  | { type: 'RESET' };

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case 'UPLOAD_START':   return { ...state, stage: 'uploading', error: null };
    case 'JOB_CREATED':   return { ...state, stage: 'pass1', jobId: action.jobId };
    case 'PASS1_DONE':    return { ...state, stage: 'awaiting_frames' };
    case 'PASS2_START':   return { ...state, stage: 'pass2' };
    case 'COMPLETE':      return { ...state, stage: 'complete' };
    case 'FAIL':          return { ...state, stage: 'failed', error: action.error };
    case 'RESET':         return { stage: 'idle', jobId: null, error: null };
    default:              return state;
  }
}

export default function AnalyzePage() {
  const [{ stage, jobId, error }, dispatch] = useReducer(reducer, {
    stage: 'idle', jobId: null, error: null,
  });

  const pollingActive = stage === 'pass1' || stage === 'pass2';
  const job = useJobPoll(jobId, pollingActive);

  // React to job status changes
  const [lastStatus, setLastStatus] = useState<string | null>(null);
  if (job && job.status !== lastStatus) {
    setLastStatus(job.status);
    if (job.status === 'pass1_done' && stage === 'pass1') dispatch({ type: 'PASS1_DONE' });
    if (job.status === 'complete' && stage === 'pass2') dispatch({ type: 'COMPLETE' });
    if (job.status === 'failed') dispatch({ type: 'FAIL', error: job.error ?? 'Analysis failed' });
  }

  async function handleUpload(file: File, config: JobConfig) {
    dispatch({ type: 'UPLOAD_START' });
    try {
      const res = await apiClient.createJob(file, config);
      dispatch({ type: 'JOB_CREATED', jobId: res.job_id });
    } catch (e) {
      dispatch({ type: 'FAIL', error: String(e) });
    }
  }

  async function handlePass2(config: Pass2Config) {
    if (!jobId) return;
    dispatch({ type: 'PASS2_START' });
    try {
      await apiClient.submitPass2(jobId, config);
    } catch (e) {
      dispatch({ type: 'FAIL', error: String(e) });
    }
  }

  const currentJob: JobResponse | null = job;

  return (
    <main className="min-h-screen bg-gray-950 text-white">
      <nav className="px-8 py-4 flex justify-between items-center border-b border-gray-800">
        <Link href="/dashboard" className="text-xl font-bold hover:text-gray-300">VaultSense</Link>
        <Link href="/dashboard" className="text-gray-400 hover:text-white text-sm">← Dashboard</Link>
      </nav>

      <div className="max-w-2xl mx-auto px-6 py-10 space-y-8">
        <h1 className="text-2xl font-bold">Analyze a vault</h1>

        {stage === 'idle' && (
          <VideoUploader onSubmit={handleUpload} loading={false} />
        )}

        {stage === 'uploading' && (
          <div className="text-center py-12 text-gray-400">Uploading video…</div>
        )}

        {(stage === 'pass1' || stage === 'pass2') && currentJob && (
          <AnalysisProgress
            progress={currentJob.progress}
            message={currentJob.message}
          />
        )}

        {stage === 'awaiting_frames' && currentJob && (
          <FrameSelector
            jobId={currentJob.job_id}
            totalFrames={currentJob.total_frames ?? 300}
            fps={currentJob.fps ?? 30}
            onSubmit={handlePass2}
            loading={false}
          />
        )}

        {stage === 'complete' && currentJob?.metrics && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold">Results</h2>
              <button
                onClick={() => dispatch({ type: 'RESET' })}
                className="text-sm text-gray-400 hover:text-white"
              >
                Analyze another
              </button>
            </div>
            <MetricsGrid metrics={currentJob.metrics} />
            {currentJob.result_files?.video && (
              <div className="space-y-2">
                <h3 className="text-sm font-medium text-gray-400">Annotated video</h3>
                <video
                  src={apiClient.getResultFileUrl(currentJob.job_id, 'output.mp4')}
                  controls
                  className="w-full rounded-xl"
                />
              </div>
            )}
            <a
              href={apiClient.getResultFileUrl(currentJob.job_id, 'velocity_data.csv')}
              className="inline-block text-sm text-blue-400 hover:underline"
              download
            >
              Download velocity CSV
            </a>
          </div>
        )}

        {stage === 'failed' && (
          <div className="space-y-4">
            <p className="text-red-400">{error}</p>
            <button
              onClick={() => dispatch({ type: 'RESET' })}
              className="px-4 py-2 rounded-lg bg-gray-800 hover:bg-gray-700 text-sm"
            >
              Try again
            </button>
          </div>
        )}
      </div>
    </main>
  );
}
