'use client';

import { useState } from 'react';
import { MetricsGrid } from './MetricsGrid';
import { DebugImages } from './DebugImages';
import { VelocityChart } from './VelocityChart';
import { StrideChart } from './StrideChart';
import { CollapsibleSection } from './CollapsibleSection';
import { LLMPanel } from './LLMPanel';
import { apiClient } from '@/lib/api-client';
import type { JobResponse } from '@/lib/types';

interface Props {
  job: JobResponse;
  onReset: () => void;
}

function metersToFeetInches(m: number): string {
  const totalInches = m * 39.3701;
  const feet = Math.floor(totalInches / 12);
  const inches = totalInches % 12;
  return `${feet}'${inches.toFixed(1)}"`;
}

export function ResultsPanel({ job, onReset }: Props) {
  const [imperial, setImperial] = useState(true);
  const [gender, setGender] = useState<'men' | 'women'>('men');
  const metrics = job.metrics;
  const resultFiles = job.result_files;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">Analysis Results</h2>
        <button onClick={onReset} className="text-sm text-gray-400 hover:text-white">
          Analyze another
        </button>
      </div>

      {/* Toggles */}
      <div className="flex gap-4">
        <div className="flex rounded-lg overflow-hidden border border-gray-700">
          {(['imperial', 'metric'] as const).map(u => (
            <button
              key={u}
              onClick={() => setImperial(u === 'imperial')}
              className={`px-3 py-1 text-xs font-medium capitalize ${
                (u === 'imperial') === imperial
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:text-white'
              }`}
            >
              {u}
            </button>
          ))}
        </div>
        <div className="flex rounded-lg overflow-hidden border border-gray-700">
          {(['men', 'women'] as const).map(g => (
            <button
              key={g}
              onClick={() => setGender(g)}
              className={`px-3 py-1 text-xs font-medium capitalize ${
                gender === g
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:text-white'
              }`}
            >
              {g}
            </button>
          ))}
        </div>
      </div>

      {/* Headline: Predicted Clearance */}
      {metrics?.predicted_clear_m != null && (
        <div className="p-6 rounded-xl bg-gradient-to-br from-blue-900/40 to-purple-900/40 border border-blue-800/50 text-center">
          <p className="text-gray-400 text-sm mb-1">Predicted Bar Clearance</p>
          <p className="text-4xl font-bold">
            {imperial
              ? metersToFeetInches(metrics.predicted_clear_m)
              : `${metrics.predicted_clear_m.toFixed(2)} m`
            }
          </p>
          <p className="text-gray-400 text-sm mt-1">
            {imperial
              ? `${metrics.predicted_clear_m.toFixed(2)} m`
              : metersToFeetInches(metrics.predicted_clear_m)
            }
          </p>
        </div>
      )}

      {/* Summary Report Card */}
      {metrics && <MetricsGrid metrics={metrics} gender={gender} />}

      {/* Annotated Video */}
      {resultFiles?.video && (
        <div className="space-y-2">
          <h3 className="text-sm font-medium text-gray-400">Annotated video</h3>
          <video
            src={`${apiClient.getResultFileUrl(job.job_id, 'output.mp4')}?t=${Date.now()}`}
            controls
            className="w-full rounded-xl"
          />
          <a
            href={`${apiClient.getResultFileUrl(job.job_id, 'output.mp4')}?t=${Date.now()}`}
            download="analysis_output.mp4"
            className="inline-block text-xs text-blue-400 hover:underline"
          >
            Download video
          </a>
        </div>
      )}

      {/* Approach Run */}
      {resultFiles?.velocity_csv && (
        <CollapsibleSection title="Approach Run">
          <VelocityChart jobId={job.job_id} imperial={imperial} />
          {resultFiles.gait_csv && (
            <StrideChart jobId={job.job_id} imperial={imperial} />
          )}
        </CollapsibleSection>
      )}

      {/* Pole Mechanics */}
      {resultFiles && resultFiles.debug_images.length > 0 && (
        <CollapsibleSection title="Pole Mechanics">
          <DebugImages jobId={job.job_id} images={resultFiles.debug_images} />
        </CollapsibleSection>
      )}

      {/* Jump Height */}
      {metrics?.max_hip_height_m != null && (
        <CollapsibleSection title="Jump Height">
          <div className="grid grid-cols-2 gap-4">
            <div className="p-4 rounded-lg bg-gray-900 border border-gray-800">
              <p className="text-xs text-gray-400 mb-1">Peak Hip Height</p>
              <p className="text-lg font-semibold">
                {imperial
                  ? metersToFeetInches(metrics.max_hip_height_m)
                  : `${metrics.max_hip_height_m.toFixed(2)} m`
                }
              </p>
              <p className="text-xs text-gray-500 mt-0.5">
                {imperial
                  ? `${metrics.max_hip_height_m.toFixed(2)} m`
                  : metersToFeetInches(metrics.max_hip_height_m)
                }
              </p>
            </div>
            {metrics.predicted_clear_m != null && (
              <div className="p-4 rounded-lg bg-gray-900 border border-gray-800">
                <p className="text-xs text-gray-400 mb-1">Predicted Clearance</p>
                <p className="text-lg font-semibold">
                  {imperial
                    ? metersToFeetInches(metrics.predicted_clear_m)
                    : `${metrics.predicted_clear_m.toFixed(2)} m`
                  }
                </p>
                <p className="text-xs text-gray-500 mt-0.5">
                  {imperial
                    ? `${metrics.predicted_clear_m.toFixed(2)} m`
                    : metersToFeetInches(metrics.predicted_clear_m)
                  }
                </p>
              </div>
            )}
          </div>
        </CollapsibleSection>
      )}

      {/* AI Coach */}
      {metrics && <LLMPanel metrics={metrics} />}
    </div>
  );
}
