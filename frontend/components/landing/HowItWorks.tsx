const steps = [
  {
    number: 1,
    title: 'Upload',
    description:
      'Record your vault from the side and upload the video. We accept MP4 and MOV files.',
  },
  {
    number: 2,
    title: 'Detect & Track',
    description:
      'YOLO person detection finds you in frame. MediaPipe extracts 33 skeletal landmarks per frame.',
  },
  {
    number: 3,
    title: 'Analyze',
    description:
      'Custom pole segmentation measures bend. An MLP classifier detects foot-strikes for cadence and stride.',
  },
  {
    number: 4,
    title: 'Results',
    description:
      'Get an annotated video, performance metrics with elite benchmarks, charts, CSV exports, and AI coaching insights.',
  },
];

export function HowItWorks() {
  return (
    <section className="max-w-6xl mx-auto px-6 py-16 md:py-20">
      <h2 className="text-3xl md:text-4xl font-bold text-center">
        How it works
      </h2>
      <p className="text-gray-400 text-lg text-center max-w-2xl mx-auto mt-3">
        From video upload to coaching insights in minutes
      </p>
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mt-12">
        {steps.map((step) => (
          <div
            key={step.number}
            className="bg-gray-900 border border-gray-800 rounded-xl p-6 space-y-3"
          >
            <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center text-sm font-bold">
              {step.number}
            </div>
            <h3 className="text-lg font-semibold">{step.title}</h3>
            <p className="text-gray-400 text-sm leading-relaxed">
              {step.description}
            </p>
          </div>
        ))}
      </div>
    </section>
  );
}
