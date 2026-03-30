'use client';

import { useState } from 'react';
import { apiClient } from '@/lib/api-client';
import type { StartConfig } from '@/lib/types';

interface Props {
  jobId: string;
  totalFrames: number;
  fps: number;
  onSubmit: (config: StartConfig) => void;
  loading: boolean;
}

interface FrameDef {
  key: keyof StartConfig;
  label: string;
  description: string;
}

const FRAMES: FrameDef[] = [
  { key: 'start_frame', label: '1. Start Frame', description: 'First step of the approach run' },
  { key: 'plant_frame', label: '2. Plant Frame', description: 'Pole tip enters the box' },
  { key: 'end_frame',   label: '3. End Frame',   description: 'Athlete lands or leaves frame' },
];

export function InitialFrameSelector({ jobId, totalFrames, fps, onSubmit, loading }: Props) {
  const [frames, setFrames] = useState<Record<string, number>>({
    start_frame: 0,
    plant_frame: Math.round(totalFrames * 0.8),
    end_frame: totalFrames - 1,
  });
  const [fullVideo, setFullVideo] = useState(false);

  function setFrame(key: string, value: number) {
    setFrames(prev => ({ ...prev, [key]: Math.max(0, Math.min(totalFrames - 1, value)) }));
  }

  function handleSubmit() {
    if (fullVideo) {
      onSubmit({});
    } else {
      onSubmit({
        start_frame: frames.start_frame,
        plant_frame: frames.plant_frame,
        end_frame: frames.end_frame,
      });
    }
  }

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-xl font-semibold">Select key frames</h2>
        <p className="text-gray-400 text-sm mt-1">
          Identify the start of the approach, the pole plant, and end of the vault.
        </p>
      </div>

      {FRAMES.map(({ key, label, description }) => {
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

            {/* Frame preview */}
            <div className="relative bg-black rounded-lg overflow-hidden aspect-video">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                key={val}
                src={apiClient.getFrameUrl(jobId, val)}
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

      <div className="flex items-center gap-2 text-sm text-gray-400">
        <input
          type="checkbox"
          id="fullVideo"
          checked={fullVideo}
          onChange={e => setFullVideo(e.target.checked)}
          className="accent-blue-500"
        />
        <label htmlFor="fullVideo">Ignore sliders (process full video)</label>
      </div>

      <p className="text-xs text-gray-500">
        Analysis window: frame {frames.start_frame} → {frames.end_frame}
        {' '}({((frames.end_frame - frames.start_frame) / fps).toFixed(1)}s)
      </p>

      <button
        onClick={handleSubmit}
        disabled={loading}
        className="w-full py-3 rounded-xl bg-blue-600 hover:bg-blue-700 text-white font-semibold disabled:opacity-40"
      >
        {loading ? 'Starting…' : 'Run Analysis'}
      </button>
    </div>
  );
}
