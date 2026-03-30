'use client';

import { apiClient } from '@/lib/api-client';

interface Props {
  jobId: string;
  images: string[];
}

const CAPTIONS: Record<string, string> = {
  debug_tip_hand: 'Phase 1: Ground Tip to Bottom Hand',
  debug_top_hand: 'Phase 2: Top Hand to Tip',
  debug_plant: 'Plant Frame: Reconstructed Tip (blue dot)',
  debug_bend: 'Max Bend: Projected Tip (blue) vs Top Hand (red)',
};

function getCaption(filename: string): string {
  for (const [key, caption] of Object.entries(CAPTIONS)) {
    if (filename.includes(key)) return caption;
  }
  return filename;
}

export function DebugImages({ jobId, images }: Props) {
  if (!images.length) return null;

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {images.map(img => (
        <div key={img} className="space-y-2">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={apiClient.getResultFileUrl(jobId, img)}
            alt={getCaption(img)}
            className="w-full rounded-lg"
          />
          <div className="flex justify-between items-center">
            <p className="text-xs text-gray-400">{getCaption(img)}</p>
            <a
              href={apiClient.getResultFileUrl(jobId, img)}
              download={img}
              className="text-xs text-blue-400 hover:underline"
            >
              Download
            </a>
          </div>
        </div>
      ))}
    </div>
  );
}
