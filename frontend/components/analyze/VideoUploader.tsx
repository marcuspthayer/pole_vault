'use client';

import { useRef, useState } from 'react';
import type { JobConfig } from '@/lib/types';

interface Props {
  onSubmit: (file: File, config: JobConfig) => void;
  loading: boolean;
}

function feetInchesToM(feet: string, inches: string): number | undefined {
  const ft = parseFloat(feet) || 0;
  const inc = parseFloat(inches) || 0;
  if (!ft && !inc) return undefined;
  return (ft * 12 + inc) * 0.0254;
}

function feetToM(feet: string): number | undefined {
  const ft = parseFloat(feet);
  if (isNaN(ft)) return undefined;
  return ft * 0.3048;
}

export function VideoUploader({ onSubmit, loading }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [units, setUnits] = useState<'metric' | 'imperial'>('imperial');

  // Metric inputs
  const [heightM, setHeightM] = useState('');
  const [poleLengthM, setPoleLengthM] = useState('');

  // Imperial inputs
  const [heightFt, setHeightFt] = useState('');
  const [heightIn, setHeightIn] = useState('');
  const [poleLengthFt, setPoleLengthFt] = useState('');

  const inputRef = useRef<HTMLInputElement>(null);

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith('video/')) setFile(f);
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!file) return;

    let athleteHeight: number | undefined;
    let poleLength: number | undefined;

    if (units === 'metric') {
      athleteHeight = heightM ? parseFloat(heightM) : undefined;
      poleLength = poleLengthM ? parseFloat(poleLengthM) : undefined;
    } else {
      athleteHeight = feetInchesToM(heightFt, heightIn);
      poleLength = feetToM(poleLengthFt);
    }

    const config: JobConfig = {
      enable_skeleton: true,
      enable_pole: true,
      enable_step: true,
      enable_max_hip_height: true,
      enable_manual_pole_frames: true,
      ...(athleteHeight ? { athlete_height_m: athleteHeight } : {}),
      ...(poleLength ? { pole_length_m: poleLength } : {}),
    };
    onSubmit(file, config);
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* Drop zone */}
      <div
        onDrop={handleDrop}
        onDragOver={e => e.preventDefault()}
        onClick={() => inputRef.current?.click()}
        className="border-2 border-dashed border-gray-700 rounded-xl p-12 text-center cursor-pointer hover:border-blue-500 transition-colors"
      >
        <input
          ref={inputRef}
          type="file"
          accept="video/*"
          className="hidden"
          onChange={e => setFile(e.target.files?.[0] ?? null)}
        />
        {file ? (
          <div className="space-y-1">
            <p className="text-white font-medium">{file.name}</p>
            <p className="text-gray-400 text-sm">{(file.size / 1024 / 1024).toFixed(1)} MB</p>
          </div>
        ) : (
          <div className="space-y-2">
            <p className="text-gray-300">Drop a video here or click to browse</p>
            <p className="text-gray-500 text-sm">MP4, MOV — any resolution</p>
          </div>
        )}
      </div>

      {/* Unit toggle */}
      <div className="flex items-center gap-3">
        <span className="text-sm text-gray-400">Units:</span>
        <div className="flex rounded-lg overflow-hidden border border-gray-700">
          {(['imperial', 'metric'] as const).map(u => (
            <button
              key={u}
              type="button"
              onClick={() => setUnits(u)}
              className={`px-4 py-1.5 text-sm font-medium capitalize transition-colors ${
                units === u
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:text-white'
              }`}
            >
              {u}
            </button>
          ))}
        </div>
      </div>

      {/* Inputs */}
      {units === 'imperial' ? (
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-1">
            <label className="text-sm text-gray-400">Athlete height</label>
            <div className="flex gap-2">
              <div className="relative flex-1">
                <input
                  type="number"
                  min="4" max="8" step="1"
                  placeholder="6"
                  value={heightFt}
                  onChange={e => setHeightFt(e.target.value)}
                  className="w-full px-3 py-2 pr-8 rounded-lg bg-gray-800 text-white border border-gray-700 focus:outline-none focus:border-blue-500"
                />
                <span className="absolute right-2.5 top-2.5 text-gray-500 text-sm">ft</span>
              </div>
              <div className="relative flex-1">
                <input
                  type="number"
                  min="0" max="11" step="1"
                  placeholder="0"
                  value={heightIn}
                  onChange={e => setHeightIn(e.target.value)}
                  className="w-full px-3 py-2 pr-8 rounded-lg bg-gray-800 text-white border border-gray-700 focus:outline-none focus:border-blue-500"
                />
                <span className="absolute right-2.5 top-2.5 text-gray-500 text-sm">in</span>
              </div>
            </div>
          </div>
          <div className="space-y-1">
            <label className="text-sm text-gray-400">Pole length</label>
            <div className="relative">
              <input
                type="number"
                min="10" max="20" step="0.5"
                placeholder="16"
                value={poleLengthFt}
                onChange={e => setPoleLengthFt(e.target.value)}
                className="w-full px-3 py-2 pr-10 rounded-lg bg-gray-800 text-white border border-gray-700 focus:outline-none focus:border-blue-500"
              />
              <span className="absolute right-2.5 top-2.5 text-gray-500 text-sm">ft</span>
            </div>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-1">
            <label className="text-sm text-gray-400">Athlete height (m)</label>
            <input
              type="number"
              step="0.01" min="1.0" max="2.5"
              placeholder="1.83"
              value={heightM}
              onChange={e => setHeightM(e.target.value)}
              className="w-full px-3 py-2 rounded-lg bg-gray-800 text-white border border-gray-700 focus:outline-none focus:border-blue-500"
            />
          </div>
          <div className="space-y-1">
            <label className="text-sm text-gray-400">Pole length (m)</label>
            <input
              type="number"
              step="0.01" min="3.0" max="6.0"
              placeholder="4.87"
              value={poleLengthM}
              onChange={e => setPoleLengthM(e.target.value)}
              className="w-full px-3 py-2 rounded-lg bg-gray-800 text-white border border-gray-700 focus:outline-none focus:border-blue-500"
            />
          </div>
        </div>
      )}

      <button
        type="submit"
        disabled={!file || loading}
        className="w-full py-3 rounded-xl bg-blue-600 hover:bg-blue-700 text-white font-semibold disabled:opacity-40"
      >
        {loading ? 'Uploading…' : 'Start analysis'}
      </button>
    </form>
  );
}
