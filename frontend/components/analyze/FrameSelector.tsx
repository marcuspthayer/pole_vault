'use client';

import { useState, useCallback } from 'react';
import { apiClient } from '@/lib/api-client';
import type { Pass2Config } from '@/lib/types';

interface Props {
  jobId: string;
  totalFrames: number;
  fps: number;
  onSubmit: (config: Pass2Config) => void;
  loading: boolean;
}

const FRAME_LABELS: { key: keyof Pass2Config; label: string; description: string }[] = [
  { key: 'plant_frame',      label: 'Plant',      description: 'Pole tip touches the box' },
  { key: 'phase1_frame',     label: 'Phase 1',    description: 'Bottom hand grip point' },
  { key: 'phase2_frame',     label: 'Phase 2',    description: 'Top hand grip point' },
  { key: 'bend_start_frame', label: 'Bend Start', description: 'Pole begins to bend' },
  { key: 'bend_end_frame',   label: 'Bend End',   description: 'Maximum pole bend' },
];

export function FrameSelector({ jobId, totalFrames, fps, onSubmit, loading }: Props) {
  const [frameIdx, setFrameIdx] = useState(0);
  const [selections, setSelections] = useState<Partial<Pass2Config>>({});

  const frameUrl = apiClient.getFrameUrl(jobId, frameIdx);
  const timeStr = (frameIdx / fps).toFixed(2);

  const stamp = useCallback((key: keyof Pass2Config) => {
    setSelections(prev => ({ ...prev, [key]: frameIdx }));
  }, [frameIdx]);

  const allSelected = FRAME_LABELS.every(f => selections[f.key] !== undefined);

  function handleSubmit() {
    if (allSelected) onSubmit(selections as Pass2Config);
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold">Select pole frames</h2>
        <p className="text-gray-400 text-sm mt-1">
          Scrub to each moment and stamp the frame. All 5 are required.
        </p>
      </div>

      {/* Frame viewer */}
      <div className="relative bg-black rounded-xl overflow-hidden aspect-video">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={frameUrl} alt={`Frame ${frameIdx}`} className="w-full h-full object-contain" />
        <div className="absolute bottom-2 right-2 bg-black/60 text-white text-xs px-2 py-1 rounded">
          Frame {frameIdx} / {totalFrames - 1} &nbsp;·&nbsp; {timeStr}s
        </div>
      </div>

      {/* Scrubber */}
      <input
        type="range"
        min={0}
        max={totalFrames - 1}
        value={frameIdx}
        onChange={e => setFrameIdx(Number(e.target.value))}
        className="w-full accent-blue-500"
      />

      {/* Stamp buttons */}
      <div className="grid grid-cols-1 gap-2">
        {FRAME_LABELS.map(({ key, label, description }) => (
          <div key={key} className="flex items-center justify-between p-3 rounded-lg bg-gray-800">
            <div>
              <span className="font-medium text-sm">{label}</span>
              <span className="text-gray-400 text-xs ml-2">{description}</span>
            </div>
            <div className="flex items-center gap-3">
              {selections[key] !== undefined && (
                <span className="text-xs text-blue-400">frame {selections[key]}</span>
              )}
              <button
                type="button"
                onClick={() => stamp(key)}
                className="px-3 py-1 rounded-lg bg-blue-600 hover:bg-blue-700 text-sm font-medium"
              >
                Stamp
              </button>
            </div>
          </div>
        ))}
      </div>

      <button
        onClick={handleSubmit}
        disabled={!allSelected || loading}
        className="w-full py-3 rounded-xl bg-green-600 hover:bg-green-700 text-white font-semibold disabled:opacity-40"
      >
        {loading ? 'Running final analysis…' : 'Run final analysis'}
      </button>
    </div>
  );
}
