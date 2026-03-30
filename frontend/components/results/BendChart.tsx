'use client';

import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from 'recharts';

interface BendPoint {
  frame: number;
  chord_ratio: number;
  smoothed?: number;
}

interface Props {
  data: BendPoint[];
}

export function BendChart({ data }: Props) {
  if (!data.length) return null;

  const chartData = data.map(d => ({
    frame: d.frame,
    raw: Number((d.chord_ratio * 100).toFixed(1)),
    smoothed: d.smoothed != null ? Number((d.smoothed * 100).toFixed(1)) : undefined,
  }));

  return (
    <div className="space-y-3">
      <h4 className="text-sm font-medium text-gray-300">Pole bend progression</h4>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="frame" stroke="#9CA3AF" fontSize={11} label={{ value: 'Frame', position: 'insideBottom', offset: -2, style: { fill: '#9CA3AF', fontSize: 11 } }} />
          <YAxis stroke="#9CA3AF" fontSize={11} domain={[60, 100]} label={{ value: '% chord', angle: -90, position: 'insideLeft', style: { fill: '#9CA3AF', fontSize: 11 } }} />
          <Tooltip contentStyle={{ background: '#1F2937', border: '1px solid #374151', borderRadius: 8, fontSize: 12 }} />
          <ReferenceLine y={75} stroke="#22C55E" strokeDasharray="4 4" label={{ value: 'Elite target (75%)', position: 'right', style: { fill: '#22C55E', fontSize: 10 } }} />
          <Line type="monotone" dataKey="raw" stroke="#22C55E" strokeWidth={1} dot={false} opacity={0.4} name="Raw" />
          {chartData[0]?.smoothed !== undefined && (
            <Line type="monotone" dataKey="smoothed" stroke="#22C55E" strokeWidth={2} dot={false} name="Smoothed" />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
