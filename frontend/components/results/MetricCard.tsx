'use client';

interface Props {
  label: string;
  value: string | number | null | undefined;
  unit?: string;
  badge?: string;
  badgeColor?: string;
  description?: string;
}

export function MetricCard({ label, value, unit, badge, badgeColor = 'text-gray-400', description }: Props) {
  return (
    <div className="p-4 rounded-xl bg-gray-900 border border-gray-800 space-y-1">
      <p className="text-gray-400 text-xs uppercase tracking-wide">{label}</p>
      <div className="flex items-baseline gap-1.5">
        <span className="text-2xl font-bold text-white">
          {value != null ? value : '—'}
        </span>
        {unit && <span className="text-gray-400 text-sm">{unit}</span>}
      </div>
      {badge && <span className={`text-xs font-medium ${badgeColor}`}>{badge}</span>}
      {description && <p className="text-gray-500 text-xs">{description}</p>}
    </div>
  );
}
