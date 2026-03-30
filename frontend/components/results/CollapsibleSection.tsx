'use client';

import { useState } from 'react';

interface Props {
  title: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}

export function CollapsibleSection({ title, defaultOpen = true, children }: Props) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className="border border-gray-800 rounded-xl overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex justify-between items-center px-5 py-3 bg-gray-900 hover:bg-gray-800 transition-colors"
      >
        <span className="font-medium text-sm">{title}</span>
        <span className="text-gray-500 text-sm">{open ? '−' : '+'}</span>
      </button>
      {open && <div className="px-5 py-4 space-y-6">{children}</div>}
    </div>
  );
}
