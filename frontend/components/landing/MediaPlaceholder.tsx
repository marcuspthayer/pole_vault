interface Props {
  label: string;
  sublabel?: string;
  aspectRatio?: string;
  src?: string;
  type?: 'gif' | 'video';
}

export function MediaPlaceholder({
  label,
  sublabel,
  aspectRatio = 'aspect-video',
  src,
  type = 'gif',
}: Props) {
  if (src) {
    if (type === 'video') {
      return (
        <video
          src={src}
          autoPlay
          loop
          muted
          playsInline
          className={`w-full ${aspectRatio} rounded-xl object-cover`}
        />
      );
    }
    return (
      // eslint-disable-next-line @next/next/no-img-element
      <img
        src={src}
        alt={label}
        className={`w-full ${aspectRatio} rounded-xl object-cover`}
      />
    );
  }

  return (
    <div
      className={`w-full ${aspectRatio} rounded-xl bg-gray-900 border border-gray-800 flex flex-col items-center justify-center gap-3`}
    >
      <svg
        className="w-12 h-12 text-gray-700"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
        strokeWidth={1.5}
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="m15.75 10.5 4.72-4.72a.75.75 0 0 1 1.28.53v11.38a.75.75 0 0 1-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 0 0 2.25-2.25v-9A2.25 2.25 0 0 0 13.5 5.25h-9A2.25 2.25 0 0 0 2.25 7.5v9A2.25 2.25 0 0 0 4.5 18.75Z"
        />
      </svg>
      <div className="text-center">
        <p className="text-gray-500 text-sm font-medium">{label}</p>
        {sublabel && <p className="text-gray-600 text-xs mt-1">{sublabel}</p>}
      </div>
    </div>
  );
}
