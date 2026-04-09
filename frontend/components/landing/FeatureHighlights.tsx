import { MediaPlaceholder } from './MediaPlaceholder';

const features = [
  {
    title: '33-point skeleton overlay',
    description:
      'MediaPipe pose estimation tracks 33 landmarks frame-by-frame — shoulders, hips, knees, ankles, and more. See exactly how your body moves through the approach, plant, swing, and clearance.',
    mediaLabel: 'Skeleton tracking preview',
    mediaSublabel: 'Coming soon',
  },
  {
    title: 'Custom pole segmentation',
    description:
      'A purpose-trained YOLO model segments the pole in every frame. We measure chord-to-arc-length ratio to quantify bend progression from plant through release.',
    mediaLabel: 'Pole bend analysis preview',
    mediaSublabel: 'Coming soon',
  },
  {
    title: 'AI coaching insights',
    description:
      'After analysis, get personalized coaching feedback powered by AI. Your metrics are compared to elite benchmarks to identify what you\'re doing well and where to focus your training.',
    mediaLabel: '',
    mediaSublabel: '',
    mockCoaching: true,
  },
];

function CoachingMock() {
  return (
    <div className="w-full rounded-xl bg-gray-900 border border-gray-800 p-5 space-y-3">
      <div className="flex items-center gap-2">
        <div className="w-6 h-6 rounded-full bg-blue-600 flex items-center justify-center">
          <svg className="w-3.5 h-3.5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M8.625 12a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Zm0 0H8.25m4.125 0a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Zm0 0H12m4.125 0a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Zm0 0h-.375M21 12c0 4.556-4.03 8.25-9 8.25a9.764 9.764 0 0 1-2.555-.337A5.972 5.972 0 0 1 5.41 20.97a5.969 5.969 0 0 1-.474-.065 4.48 4.48 0 0 0 .978-2.025c.09-.457-.133-.901-.467-1.226C3.93 16.178 3 14.189 3 12c0-4.556 4.03-8.25 9-8.25s9 3.694 9 8.25Z" />
          </svg>
        </div>
        <span className="text-sm font-medium text-white">AI Coach</span>
      </div>
      <div className="text-sm text-gray-300 leading-relaxed space-y-2">
        <p>
          <strong className="text-white">Approach velocity is solid</strong> at
          8.7 m/s — you&apos;re in the competitive range. To push toward elite
          (&gt;9.2 m/s), focus on driving through your last three steps.
        </p>
        <p>
          <strong className="text-white">Pole bend looks great</strong> at 74%
          chord ratio — right in the elite window. Your energy transfer into the
          pole is efficient.
        </p>
        <p>
          <strong className="text-white">Work on velocity retention</strong> —
          you&apos;re losing speed at the plant. Try the &quot;run through the
          plant&quot; cue to maintain momentum.
        </p>
      </div>
    </div>
  );
}

export function FeatureHighlights() {
  return (
    <section className="max-w-6xl mx-auto px-6 py-16 md:py-20 space-y-16">
      <div className="text-center">
        <h2 className="text-3xl md:text-4xl font-bold">
          Built for serious vaulters
        </h2>
        <p className="text-gray-400 text-lg max-w-2xl mx-auto mt-3">
          Computer vision and AI working together to break down your technique
        </p>
      </div>
      {features.map((feature, i) => (
        <div
          key={feature.title}
          className={`grid grid-cols-1 md:grid-cols-2 gap-8 items-center ${
            i % 2 === 1 ? 'md:direction-rtl' : ''
          }`}
        >
          <div className={`space-y-4 ${i % 2 === 1 ? 'md:order-2' : ''}`}>
            <h3 className="text-2xl font-semibold">{feature.title}</h3>
            <p className="text-gray-400 leading-relaxed">
              {feature.description}
            </p>
          </div>
          <div className={i % 2 === 1 ? 'md:order-1' : ''}>
            {feature.mockCoaching ? (
              <CoachingMock />
            ) : (
              <MediaPlaceholder
                label={feature.mediaLabel}
                sublabel={feature.mediaSublabel}
                aspectRatio="aspect-[4/3]"
              />
            )}
          </div>
        </div>
      ))}
    </section>
  );
}
