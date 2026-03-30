'use client';

import { useEffect, useState } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from 'recharts';
import { apiClient } from '@/lib/api-client';
import type { VelocityRow } from '@/lib/types';

interface Props {
  jobId: string;
  imperial: boolean;
}

export function VelocityChart({ jobId, imperial }: Props) {
  const [data, setData] = useState<VelocityRow[]>([]);

  useEffect(() => {
    apiClient.fetchCsv<VelocityRow>(jobId, 'velocity_data.csv').then(setData);
  }, [jobId]);

  if (!data.length) return null;

  const chartData = data.map(row => ({
    time: Number(row.time_s.toFixed(2)),
    velocity: imperial ? Number(row.velocity_mph.toFixed(1)) : Number(row.velocity_m_s.toFixed(2)),
  }));

  const unit = imperial ? 'mph' : 'm/s';
  const peak = Math.max(...chartData.map(d => d.velocity));

  return (
    <div className="space-y-3">
      <div className="flex justify-between items-baseline">
        <h4 className="text-sm font-medium text-gray-300">Velocity over time</h4>
        <span className="text-xs text-gray-500">Peak: {peak.toFixed(imperial ? 1 : 2)} {unit}</span>
      </div>
      <ResponsiveContainer width="100%" height={250}>
        <LineChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="time" stroke="#9CA3AF" fontSize={11} label={{ value: 'Time (s)', position: 'insideBottom', offset: -2, style: { fill: '#9CA3AF', fontSize: 11 } }} />
          <YAxis stroke="#9CA3AF" fontSize={11} label={{ value: unit, angle: -90, position: 'insideLeft', style: { fill: '#9CA3AF', fontSize: 11 } }} />
          <Tooltip contentStyle={{ background: '#1F2937', border: '1px solid #374151', borderRadius: 8, fontSize: 12 }} />
          <ReferenceLine y={peak} stroke="#3B82F6" strokeDasharray="4 4" strokeOpacity={0.5} />
          <Line type="monotone" dataKey="velocity" stroke="#3B82F6" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
      <div className="flex justify-end">
        <a
          href={apiClient.getResultFileUrl(jobId, 'velocity_data.csv')}
          download
          className="text-xs text-blue-400 hover:underline"
        >
          Download CSV
        </a>
      </div>
    </div>
  );
}
