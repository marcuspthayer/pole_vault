'use client';

interface Props {
  progress: number; // 0.0 - 1.0
  message: string;
}

export function AnalysisProgress({ progress, message }: Props) {
  const pct = Math.round(progress * 100);
  return (
    <div className="space-y-4">
      <div className="flex justify-between text-sm text-gray-400">
        <span>{message}</span>
        <span>{pct}%</span>
      </div>
      <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
        <div
          className="h-full bg-blue-500 rounded-full transition-all duration-500"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
