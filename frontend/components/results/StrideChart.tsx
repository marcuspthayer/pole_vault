'use client';

import { useEffect, useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Cell,
} from 'recharts';
import { apiClient } from '@/lib/api-client';
import type { StrideRow } from '@/lib/types';

interface Props {
  jobId: string;
  imperial: boolean;
}

export function StrideChart({ jobId, imperial }: Props) {
  const [data, setData] = useState<StrideRow[]>([]);

  useEffect(() => {
    apiClient.fetchCsv<StrideRow>(jobId, 'gait_data.csv').then(setData);
  }, [jobId]);

  if (!data.length) return null;

  const hasRealUnits = data[0].stride_cm != null;
  const chartData = data.map((row, i) => ({
    step: i + 1,
    length: hasRealUnits
      ? (imperial ? Number((row.stride_in ?? 0).toFixed(1)) : Number((row.stride_cm ?? 0).toFixed(0)))
      : Number(row.stride_norm.toFixed(3)),
    side: row.side,
  }));

  const unit = hasRealUnits ? (imperial ? 'in' : 'cm') : 'norm';
  const avg = chartData.reduce((sum, d) => sum + d.length, 0) / chartData.length;

  return (
    <div className="space-y-3">
      <div className="flex justify-between items-baseline">
        <h4 className="text-sm font-medium text-gray-300">Stride length per step</h4>
        <span className="text-xs text-gray-500">
          Avg: {avg.toFixed(hasRealUnits ? 1 : 3)} {unit}
        </span>
      </div>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="step" stroke="#9CA3AF" fontSize={11} label={{ value: 'Step #', position: 'insideBottom', offset: -2, style: { fill: '#9CA3AF', fontSize: 11 } }} />
          <YAxis stroke="#9CA3AF" fontSize={11} label={{ value: unit, angle: -90, position: 'insideLeft', style: { fill: '#9CA3AF', fontSize: 11 } }} />
          <Tooltip contentStyle={{ background: '#1F2937', border: '1px solid #374151', borderRadius: 8, fontSize: 12 }} />
          <ReferenceLine y={avg} stroke="#F97316" strokeDasharray="4 4" strokeOpacity={0.5} />
          <Bar dataKey="length" radius={[4, 4, 0, 0]}>
            {chartData.map((entry, i) => (
              <Cell key={i} fill={entry.side === 'left' ? '#3B82F6' : '#EF4444'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="flex justify-between items-center">
        <div className="flex gap-4 text-xs text-gray-500">
          <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-blue-500 inline-block" /> Left</span>
          <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-red-500 inline-block" /> Right</span>
        </div>
        <a
          href={apiClient.getResultFileUrl(jobId, 'gait_data.csv')}
          download
          className="text-xs text-blue-400 hover:underline"
        >
          Download CSV
        </a>
      </div>
    </div>
  );
}
