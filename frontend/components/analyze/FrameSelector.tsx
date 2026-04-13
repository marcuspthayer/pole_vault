'use client';

import { useState } from 'react';
import { apiClient } from '@/lib/api-client';
import type { Pass2Config } from '@/lib/types';

interface Props {
  jobId: string;
  totalFrames: number;
  fps: number;
  startFrame: number;
  plantFrame: number;
  endFrame: number;
  onSubmit: (config: Pass2Config) => void;
  loading: boolean;
}

interface FrameDef {
  key: keyof Pass2Config;
  label: string;
  description: string;
}

const FRAME_DEFS: FrameDef[] = [
  { key: 'phase1_frame',     label: 'Phase 1 Frame',  description: 'Tip-to-bottom-hand, pole should be straight' },
  { key: 'phase2_frame',     label: 'Phase 2 Frame',  description: 'Bottom hand to top of pole, around plant' },
  { key: 'plant_frame',      label: 'Plant Frame',    description: 'For tip reconstruction, needs solid mask' },
  { key: 'bend_start_frame', label: 'Max Bend Start', description: 'Start of search window for maximum pole bend' },
  { key: 'bend_end_frame',   label: 'Max Bend End',   description: 'End of search window for maximum pole bend' },
];

export function FrameSelector({
  jobId, totalFrames, fps,
  startFrame, plantFrame, endFrame,
  onSubmit, loading,
}: Props) {
  const [frames, setFrames] = useState<Record<string, number>>({
    phase1_frame: Math.min(startFrame + 5, totalFrames - 1),
    phase2_frame: plantFrame,
    plant_frame: plantFrame,
    bend_start_frame: Math.min(plantFrame + Math.round(0.1 * fps), endFrame),
    bend_end_frame: Math.min(plantFrame + Math.round(0.25 * fps), endFrame),
  });

  function setFrame(key: string, value: number) {
    setFrames(prev => ({ ...prev, [key]: Math.max(0, Math.min(totalFrames - 1, value)) }));
  }

  const allSet = FRAME_DEFS.every(f => frames[f.key] !== undefined);

  function handleSubmit() {
    if (!allSet) return;
    onSubmit(frames as unknown as Pass2Config);
  }

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-xl font-semibold">Select pole frames</h2>
        <p className="text-gray-400 text-sm mt-1">
          Review the annotated frames below. Select frames where the pole mask is clean and accurate.
        </p>
      </div>

      {/* Pass 1 output video */}
      <div className="space-y-2">
        <h3 className="text-sm font-medium text-gray-400">Pass 1 output (reference)</h3>
        <video
          src={apiClient.getResultFileUrl(jobId, 'output.mp4')}
          controls
          className="w-full rounded-xl"
        />
      </div>

      {FRAME_DEFS.map(({ key, label, description }) => {
        const val = frames[key];
        const timeSec = (val / fps).toFixed(2);
        return (
          <div key={key} className="space-y-3">
            <div className="flex justify-between items-baseline">
              <div>
                <span className="font-medium">{label}</span>
                <span className="text-gray-400 text-sm ml-2">{description}</span>
              </div>
              <span className="text-xs text-gray-500">
                Frame {val} · {timeSec}s
              </span>
            </div>

            {/* Annotated frame preview from output.mp4 */}
            <div className="relative bg-black rounded-lg overflow-hidden aspect-video">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                key={val}
                src={apiClient.getFrameUrl(jobId, val, 'output')}
                alt={`${label}: frame ${val}`}
                className="w-full h-full object-contain"
              />
            </div>

            {/* Controls: -1, slider, +1 */}
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setFrame(key, val - 1)}
                className="px-3 py-1.5 rounded-lg bg-gray-800 hover:bg-gray-700 text-sm font-medium shrink-0"
              >
                -1
              </button>
              <input
                type="range"
                min={0}
                max={totalFrames - 1}
                value={val}
                onChange={e => setFrame(key, Number(e.target.value))}
                className="flex-1 accent-blue-500"
              />
              <button
                type="button"
                onClick={() => setFrame(key, val + 1)}
                className="px-3 py-1.5 rounded-lg bg-gray-800 hover:bg-gray-700 text-sm font-medium shrink-0"
              >
                +1
              </button>
            </div>
          </div>
        );
      })}

      <button
        onClick={handleSubmit}
        disabled={!allSet || loading}
        className="w-full py-3 rounded-xl bg-green-600 hover:bg-green-700 text-white font-semibold disabled:opacity-40"
      >
        {loading ? 'Running final analysis…' : 'Finalize Analysis'}
      </button>
    </div>
  );
}
