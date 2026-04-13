'use client';

import { useState, useReducer } from 'react';
import { VideoUploader } from '@/components/analyze/VideoUploader';
import { InitialFrameSelector } from '@/components/analyze/InitialFrameSelector';
import { AnalysisProgress } from '@/components/analyze/AnalysisProgress';
import { FrameSelector } from '@/components/analyze/FrameSelector';
import { ResultsPanel } from '@/components/results/ResultsPanel';
import { useJobPoll } from '@/hooks/useJobPoll';
import { apiClient } from '@/lib/api-client';
import type { JobConfig, JobResponse, Pass2Config, StartConfig } from '@/lib/types';
import Link from 'next/link';

type Stage =
  | 'idle'
  | 'uploading'
  | 'select_frames'
  | 'pass1'
  | 'awaiting_pole_frames'
  | 'pass2'
  | 'complete'
  | 'failed';

interface State {
  stage: Stage;
  jobId: string | null;
  jobData: JobResponse | null;
  error: string | null;
  startFrame: number;
  plantFrame: number;
  endFrame: number;
}

type Action =
  | { type: 'UPLOAD_START' }
  | { type: 'JOB_CREATED'; job: JobResponse }
  | { type: 'FRAMES_SELECTED'; start: number; plant: number; end: number }
  | { type: 'PASS1_DONE' }
  | { type: 'PASS2_START' }
  | { type: 'COMPLETE' }
  | { type: 'FAIL'; error: string }
  | { type: 'RESET' };

const INITIAL_STATE: State = {
  stage: 'idle',
  jobId: null,
  jobData: null,
  error: null,
  startFrame: 0,
  plantFrame: 0,
  endFrame: 0,
};

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case 'UPLOAD_START':
      return { ...state, stage: 'uploading', error: null };
    case 'JOB_CREATED':
      return { ...state, stage: 'select_frames', jobId: action.job.job_id, jobData: action.job };
    case 'FRAMES_SELECTED':
      return { ...state, stage: 'pass1', startFrame: action.start, plantFrame: action.plant, endFrame: action.end };
    case 'PASS1_DONE':
      return { ...state, stage: 'awaiting_pole_frames' };
    case 'PASS2_START':
      return { ...state, stage: 'pass2' };
    case 'COMPLETE':
      return { ...state, stage: 'complete' };
    case 'FAIL':
      return { ...state, stage: 'failed', error: action.error };
    case 'RESET':
      return INITIAL_STATE;
    default:
      return state;
  }
}

export default function AnalyzePage() {
  const [state, dispatch] = useReducer(reducer, INITIAL_STATE);
  const { stage, jobId, jobData, error, startFrame, plantFrame, endFrame } = state;

  const pollingActive = stage === 'pass1' || stage === 'pass2';
  const polledJob = useJobPoll(jobId, pollingActive);

  // Use polled data when available, fall back to stored jobData
  const currentJob = polledJob ?? jobData;

  // React to job status changes
  const [lastStatus, setLastStatus] = useState<string | null>(null);
  if (polledJob && polledJob.status !== lastStatus) {
    setLastStatus(polledJob.status);
    if (polledJob.status === 'pass1_done' && stage === 'pass1') dispatch({ type: 'PASS1_DONE' });
    if (polledJob.status === 'complete' && (stage === 'pass2' || stage === 'pass1')) dispatch({ type: 'COMPLETE' });
    if (polledJob.status === 'failed') dispatch({ type: 'FAIL', error: polledJob.error ?? 'Analysis failed' });
  }

  function friendlyError(e: unknown): string {
    const msg = String(e);
    if (msg.includes('Failed to fetch')) {
      return 'Could not reach the analysis server. It may be restarting — please wait a moment and try again.';
    }
    return msg;
  }

  async function handleUpload(file: File, config: JobConfig) {
    dispatch({ type: 'UPLOAD_START' });
    try {
      const res = await apiClient.createJob(file, config, false);
      dispatch({ type: 'JOB_CREATED', job: res });
    } catch (e) {
      dispatch({ type: 'FAIL', error: friendlyError(e) });
    }
  }

  async function handleFramesSelected(config: StartConfig) {
    if (!jobId) return;
    const start = config.start_frame ?? 0;
    const plant = config.plant_frame ?? 0;
    const end = config.end_frame ?? 0;
    dispatch({ type: 'FRAMES_SELECTED', start, plant, end });
    try {
      await apiClient.startJob(jobId, config);
    } catch (e) {
      dispatch({ type: 'FAIL', error: friendlyError(e) });
    }
  }

  async function handlePass2(config: Pass2Config) {
    if (!jobId) return;
    dispatch({ type: 'PASS2_START' });
    try {
      await apiClient.submitPass2(jobId, config);
    } catch (e) {
      dispatch({ type: 'FAIL', error: friendlyError(e) });
    }
  }

  return (
    <main className="min-h-screen bg-gray-950 text-white">
      <nav className="px-8 py-4 flex justify-between items-center border-b border-gray-800">
        <Link href="/dashboard" className="text-xl font-bold hover:text-gray-300">VaultSense</Link>
        <Link href="/dashboard" className="text-gray-400 hover:text-white text-sm">Dashboard</Link>
      </nav>

      <div className="max-w-3xl mx-auto px-6 py-10 space-y-8">
        <h1 className="text-2xl font-bold">Analyze a vault</h1>

        {stage === 'idle' && (
          <VideoUploader onSubmit={handleUpload} loading={false} />
        )}

        {stage === 'uploading' && (
          <div className="text-center py-12 text-gray-400">Uploading video…</div>
        )}

        {stage === 'select_frames' && currentJob && (
          <InitialFrameSelector
            jobId={currentJob.job_id}
            totalFrames={currentJob.total_frames ?? 300}
            fps={currentJob.fps ?? 30}
            suggestedStartFrame={currentJob.suggested_start_frame}
            onSubmit={handleFramesSelected}
            loading={false}
          />
        )}

        {(stage === 'pass1' || stage === 'pass2') && currentJob && (
          <AnalysisProgress
            progress={currentJob.progress}
            message={currentJob.message}
          />
        )}

        {stage === 'awaiting_pole_frames' && currentJob && (
          <FrameSelector
            jobId={currentJob.job_id}
            totalFrames={currentJob.total_frames ?? 300}
            fps={currentJob.fps ?? 30}
            startFrame={startFrame}
            plantFrame={plantFrame}
            endFrame={endFrame}
            onSubmit={handlePass2}
            loading={false}
          />
        )}

        {stage === 'complete' && currentJob && (
          <ResultsPanel
            job={currentJob}
            onReset={() => dispatch({ type: 'RESET' })}
          />
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
