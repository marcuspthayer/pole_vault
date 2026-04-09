const metrics = [
  { label: 'Peak Velocity', value: '8.7', unit: 'm/s', badge: 'Competitive', badgeColor: 'text-blue-400' },
  { label: 'Takeoff Velocity', value: '8.1', unit: 'm/s', badge: 'Competitive', badgeColor: 'text-blue-400' },
  { label: 'Velocity Retention', value: '93', unit: '%', badge: 'Elite', badgeColor: 'text-green-400' },
  { label: 'Cadence', value: '288', unit: 'spm', badge: 'Competitive', badgeColor: 'text-blue-400' },
  { label: 'Pole Bend', value: '74', unit: '% chord', badge: 'Elite', badgeColor: 'text-green-400' },
  { label: 'Peak Hip Height', value: '4.8', unit: 'm', badge: null, badgeColor: '' },
  { label: 'Predicted Clearance', value: '4.6', unit: 'm', badge: 'Competitive', badgeColor: 'text-blue-400' },
  { label: 'Plant-to-Peak', value: '1.28', unit: 's', badge: 'Competitive', badgeColor: 'text-blue-400' },
  { label: 'Stride Length', value: '2.1', unit: 'm', badge: null, badgeColor: '' },
];

export function MetricsShowcase() {
  return (
    <section className="bg-gray-900/40 border-y border-gray-800">
      <div className="max-w-6xl mx-auto px-6 py-16 md:py-20">
        <h2 className="text-3xl md:text-4xl font-bold text-center">
          Performance metrics, computed automatically
        </h2>
        <p className="text-gray-400 text-lg text-center max-w-2xl mx-auto mt-3">
          Every metric is benchmarked against elite standards
        </p>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 mt-12">
          {metrics.map((m) => (
            <div
              key={m.label}
              className="p-4 rounded-xl bg-gray-900 border border-gray-800 space-y-1"
            >
              <p className="text-gray-400 text-xs uppercase tracking-wide">
                {m.label}
              </p>
              <div className="flex items-baseline gap-1.5">
                <span className="text-2xl font-bold text-white">{m.value}</span>
                <span className="text-gray-400 text-sm">{m.unit}</span>
              </div>
              {m.badge && (
                <span className={`text-xs font-medium ${m.badgeColor}`}>
                  {m.badge}
                </span>
              )}
            </div>
          ))}
        </div>
        <p className="text-gray-600 text-xs text-center mt-6">
          Sample values shown for illustration
        </p>
      </div>
    </section>
  );
}
