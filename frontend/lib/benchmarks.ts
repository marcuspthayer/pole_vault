export type BenchmarkLevel = 'beginner' | 'developing' | 'competitive' | 'elite';

export interface BenchmarkRange {
  level: BenchmarkLevel;
  label: string;
  color: string; // Tailwind text color class
  min: number;
  max: number;
}

function classify(
  value: number,
  ranges: BenchmarkRange[],
  invert = false
): BenchmarkRange | null {
  const sorted = invert ? [...ranges].reverse() : ranges;
  for (const range of sorted) {
    if (value >= range.min && value < range.max) return range;
  }
  return ranges[ranges.length - 1];
}

// Peak approach velocity (m/s) — higher is better
const VELOCITY_RANGES_MEN: BenchmarkRange[] = [
  { level: 'beginner',    label: 'Beginner',    color: 'text-gray-400',   min: 0,    max: 6.5  },
  { level: 'developing',  label: 'Developing',  color: 'text-yellow-400', min: 6.5,  max: 8.0  },
  { level: 'competitive', label: 'Competitive', color: 'text-blue-400',   min: 8.0,  max: 9.2  },
  { level: 'elite',       label: 'Elite',       color: 'text-green-400',  min: 9.2,  max: 99   },
];
const VELOCITY_RANGES_WOMEN: BenchmarkRange[] = [
  { level: 'beginner',    label: 'Beginner',    color: 'text-gray-400',   min: 0,    max: 5.8  },
  { level: 'developing',  label: 'Developing',  color: 'text-yellow-400', min: 5.8,  max: 7.0  },
  { level: 'competitive', label: 'Competitive', color: 'text-blue-400',   min: 7.0,  max: 7.8  },
  { level: 'elite',       label: 'Elite',       color: 'text-green-400',  min: 7.8,  max: 99   },
];

// Velocity retention (takeoff / peak) — higher is better
const RETENTION_RANGES: BenchmarkRange[] = [
  { level: 'beginner',    label: 'Needs Work',  color: 'text-red-400',    min: 0,    max: 0.80 },
  { level: 'developing',  label: 'Developing',  color: 'text-yellow-400', min: 0.80, max: 0.87 },
  { level: 'competitive', label: 'Competitive', color: 'text-blue-400',   min: 0.87, max: 0.93 },
  { level: 'elite',       label: 'Elite',       color: 'text-green-400',  min: 0.93, max: 99   },
];

// Cadence (steps/min) — higher is better
const CADENCE_RANGES: BenchmarkRange[] = [
  { level: 'beginner',    label: 'Low',         color: 'text-gray-400',   min: 0,    max: 230  },
  { level: 'developing',  label: 'Developing',  color: 'text-yellow-400', min: 230,  max: 265  },
  { level: 'competitive', label: 'Competitive', color: 'text-blue-400',   min: 265,  max: 295  },
  { level: 'elite',       label: 'Elite',       color: 'text-green-400',  min: 295,  max: 9999 },
];

// Pole bend chord ratio % — LOWER is more bent = better (down to ~70%)
// We classify based on how close to elite target (70-75%) the athlete is
const POLE_BEND_RANGES: BenchmarkRange[] = [
  { level: 'elite',       label: 'Elite',       color: 'text-green-400',  min: 70,  max: 76   },
  { level: 'competitive', label: 'Competitive', color: 'text-blue-400',   min: 76,  max: 82   },
  { level: 'developing',  label: 'Developing',  color: 'text-yellow-400', min: 82,  max: 90   },
  { level: 'beginner',    label: 'Under-loaded',color: 'text-gray-400',   min: 90,  max: 9999 },
];

// Predicted bar clearance (m) — higher is better
const CLEARANCE_RANGES_MEN: BenchmarkRange[] = [
  { level: 'beginner',    label: 'Beginner',    color: 'text-gray-400',   min: 0,    max: 3.5  },
  { level: 'developing',  label: 'Developing',  color: 'text-yellow-400', min: 3.5,  max: 4.5  },
  { level: 'competitive', label: 'Competitive', color: 'text-blue-400',   min: 4.5,  max: 5.2  },
  { level: 'elite',       label: 'Elite',       color: 'text-green-400',  min: 5.2,  max: 99   },
];
const CLEARANCE_RANGES_WOMEN: BenchmarkRange[] = [
  { level: 'beginner',    label: 'Beginner',    color: 'text-gray-400',   min: 0,    max: 2.8  },
  { level: 'developing',  label: 'Developing',  color: 'text-yellow-400', min: 2.8,  max: 3.6  },
  { level: 'competitive', label: 'Competitive', color: 'text-blue-400',   min: 3.6,  max: 4.4  },
  { level: 'elite',       label: 'Elite',       color: 'text-green-400',  min: 4.4,  max: 99   },
];

export const benchmarks = {
  peakVelocity(value: number, gender: 'men' | 'women' = 'men') {
    return classify(value, gender === 'men' ? VELOCITY_RANGES_MEN : VELOCITY_RANGES_WOMEN);
  },
  velocityRetention(peakMs: number, takeoffMs: number) {
    const ratio = peakMs > 0 ? takeoffMs / peakMs : 0;
    return { ratio, classification: classify(ratio, RETENTION_RANGES) };
  },
  cadence(value: number) {
    return classify(value, CADENCE_RANGES);
  },
  poleBend(value: number) {
    // value is chord ratio * 100 (lower = more bent)
    return classify(value, POLE_BEND_RANGES);
  },
  predictedClearance(value: number, gender: 'men' | 'women' = 'men') {
    return classify(value, gender === 'men' ? CLEARANCE_RANGES_MEN : CLEARANCE_RANGES_WOMEN);
  },
};
