'use client';

import { MetricCard } from './MetricCard';
import { benchmarks } from '@/lib/benchmarks';
import type { JobMetrics } from '@/lib/types';

interface Props {
  metrics: JobMetrics;
  gender?: 'men' | 'women';
}

export function MetricsGrid({ metrics, gender = 'men' }: Props) {
  const velocityClass = metrics.peak_velocity_ms != null
    ? benchmarks.peakVelocity(metrics.peak_velocity_ms, gender)
    : null;

  const retention = metrics.peak_velocity_ms && metrics.takeoff_velocity_ms
    ? benchmarks.velocityRetention(metrics.peak_velocity_ms, metrics.takeoff_velocity_ms)
    : null;

  const cadenceClass = metrics.cadence_spm != null
    ? benchmarks.cadence(metrics.cadence_spm)
    : null;

  const bendClass = metrics.max_pole_bend_pct != null
    ? benchmarks.poleBend(metrics.max_pole_bend_pct)
    : null;

  const clearanceClass = metrics.predicted_clear_m != null
    ? benchmarks.predictedClearance(metrics.predicted_clear_m, gender)
    : null;

  const plantToPeakClass = metrics.plant_to_peak_s != null
    ? benchmarks.plantToPeak(metrics.plant_to_peak_s)
    : null;

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
      <MetricCard
        label="Peak velocity"
        value={metrics.peak_velocity_ms?.toFixed(2)}
        unit="m/s"
        badge={velocityClass?.label}
        badgeColor={velocityClass?.color}
      />
      <MetricCard
        label="Takeoff velocity"
        value={metrics.takeoff_velocity_ms?.toFixed(2)}
        unit="m/s"
        badge={retention ? `${(retention.ratio * 100).toFixed(0)}% retention` : undefined}
        badgeColor={retention?.classification?.color}
      />
      <MetricCard
        label="Avg velocity"
        value={metrics.avg_velocity_ms?.toFixed(2)}
        unit="m/s"
      />
      <MetricCard
        label="Cadence"
        value={metrics.cadence_spm?.toFixed(0)}
        unit="spm"
        badge={cadenceClass?.label}
        badgeColor={cadenceClass?.color}
      />
      <MetricCard
        label="Pole bend"
        value={metrics.max_pole_bend_pct?.toFixed(1)}
        unit="% chord"
        badge={bendClass?.label}
        badgeColor={bendClass?.color}
        description="Lower = more bent. Elite: 70–75%"
      />
      <MetricCard
        label="Peak hip height"
        value={metrics.max_hip_height_m?.toFixed(2)}
        unit="m"
      />
      <MetricCard
        label="Plant to peak"
        value={metrics.plant_to_peak_s?.toFixed(2)}
        unit="s"
        badge={plantToPeakClass?.label}
        badgeColor={plantToPeakClass?.color}
      />
      <MetricCard
        label="Predicted clearance"
        value={metrics.predicted_clear_m?.toFixed(2)}
        unit={`m (${metrics.predicted_clear_in?.toFixed(1)}")`}
        badge={clearanceClass?.label}
        badgeColor={clearanceClass?.color}
      />
    </div>
  );
}
