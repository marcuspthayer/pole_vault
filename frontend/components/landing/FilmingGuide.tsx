import { MediaPlaceholder } from './MediaPlaceholder';

const requirements = [
  {
    icon: (
      <svg className="w-5 h-5 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v6h4.5m4.5 0a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z" />
      </svg>
    ),
    title: 'Frame Rate',
    description:
      'Shoot at 120-240 fps — 240 is ideal. High frame rate is essential for accurate stride detection and velocity measurement. Most modern smartphones support 240 fps in slow-motion mode.',
    highlight: '120-240 fps',
  },
  {
    icon: (
      <svg className="w-5 h-5 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M6.827 6.175A2.31 2.31 0 0 1 5.186 7.23c-.38.054-.757.112-1.134.175C2.999 7.58 2.25 8.507 2.25 9.574V18a2.25 2.25 0 0 0 2.25 2.25h15A2.25 2.25 0 0 0 21.75 18V9.574c0-1.067-.75-1.994-1.802-2.169a47.865 47.865 0 0 0-1.134-.175 2.31 2.31 0 0 1-1.64-1.055l-.822-1.316a2.192 2.192 0 0 0-1.736-1.039 48.774 48.774 0 0 0-5.232 0 2.192 2.192 0 0 0-1.736 1.039l-.821 1.316Z" />
        <path strokeLinecap="round" strokeLinejoin="round" d="M16.5 12.75a4.5 4.5 0 1 1-9 0 4.5 4.5 0 0 1 9 0Z" />
      </svg>
    ),
    title: 'Camera Position',
    description:
      'Set up off to the side of the runway. A tripod gives the most accurate results, but holding the camera still by hand works too.',
    highlight: 'Side of the runway',
  },
  {
    icon: (
      <svg className="w-5 h-5 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 9V4.5M9 9H4.5M9 9 3.75 3.75M9 15v4.5M9 15H4.5M9 15l-5.25 5.25M15 9h4.5M15 9V4.5M15 9l5.25-5.25M15 15h4.5M15 15v4.5m0-4.5 5.25 5.25" />
      </svg>
    ),
    title: 'Keep it Still',
    description:
      'Use a stationary, horizontal frame. Do not pan or follow the athlete — the camera must remain completely still throughout the vault.',
    highlight: 'No panning',
  },
  {
    icon: (
      <svg className="w-5 h-5 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M3.375 19.5h17.25m-17.25 0a1.125 1.125 0 0 1-1.125-1.125M3.375 19.5h1.5C5.496 19.5 6 18.996 6 18.375m-2.625 0V5.625m0 12.75v-1.5c0-.621.504-1.125 1.125-1.125m18.375 2.625V5.625m0 12.75c0 .621-.504 1.125-1.125 1.125m1.125-1.125v-1.5c0-.621-.504-1.125-1.125-1.125m0 3.75h-1.5A1.125 1.125 0 0 1 18 18.375M20.625 4.5H3.375m17.25 0c.621 0 1.125.504 1.125 1.125M20.625 4.5h-1.5C18.504 4.5 18 5.004 18 5.625m3.75 0v1.5c0 .621-.504 1.125-1.125 1.125M3.375 4.5c-.621 0-1.125.504-1.125 1.125M3.375 4.5h1.5C5.496 4.5 6 5.004 6 5.625m-3.75 0v1.5c0 .621.504 1.125 1.125 1.125m0 0h1.5m-1.5 0c-.621 0-1.125.504-1.125 1.125v1.5c0 .621.504 1.125 1.125 1.125m1.5-3.75C5.496 8.25 6 7.746 6 7.125v-1.5M4.875 8.25C5.496 8.25 6 8.754 6 9.375v1.5m-1.5 0h1.5m-1.5 0c-.621 0-1.125.504-1.125 1.125v1.5c0 .621.504 1.125 1.125 1.125M6 10.875v-1.5m0 0C6 8.754 6.504 8.25 7.125 8.25h9.75c.621 0 1.125.504 1.125 1.125m-10.875 0v5.25c0 .621.504 1.125 1.125 1.125h9.75c.621 0 1.125-.504 1.125-1.125m0-5.25v-1.5c0-.621.504-1.125 1.125-1.125m0 0h1.5c.621 0 1.125.504 1.125 1.125v1.5c0 .621-.504 1.125-1.125 1.125m-17.25 0h1.5m14.25 0h1.5" />
      </svg>
    ),
    title: 'Frame Coverage',
    description:
      'Capture the last 3-5 steps of the approach run plus the full vault. Start recording before the athlete enters the frame.',
    highlight: 'Last 3-5 steps + vault',
  },
  {
    icon: (
      <svg className="w-5 h-5 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M2.036 12.322a1.012 1.012 0 0 1 0-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178Z" />
        <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" />
      </svg>
    ),
    title: 'Side Angle',
    description:
      'Film from the side, roughly perpendicular to the runway. Avoid diagonal or front-facing angles — side view gives the best skeletal tracking accuracy.',
    highlight: 'Perpendicular to runway',
  },
];

export function FilmingGuide() {
  return (
    <section className="max-w-6xl mx-auto px-6 py-16 md:py-20">
      <div className="bg-blue-950/30 border border-blue-900/50 rounded-2xl p-6 md:p-10 space-y-8">
        <div className="text-center">
          <h2 className="text-3xl md:text-4xl font-bold">
            How to film your vault
          </h2>
          <p className="text-gray-400 text-lg max-w-2xl mx-auto mt-3">
            Follow these guidelines for the best analysis results
          </p>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {requirements.map((req) => (
            <div
              key={req.title}
              className="bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-2"
            >
              <div className="flex items-center gap-2">
                {req.icon}
                <h3 className="text-white font-medium text-sm">{req.title}</h3>
              </div>
              <p className="text-gray-400 text-sm leading-relaxed">
                {req.description}
              </p>
            </div>
          ))}
        </div>
        <div className="max-w-2xl mx-auto">
          <MediaPlaceholder
            label="Video filming guide"
            sublabel="Coming soon"
          />
        </div>
      </div>
    </section>
  );
}
