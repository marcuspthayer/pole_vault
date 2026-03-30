'use client';

import { useState, useCallback } from 'react';
import type { JobMetrics } from '@/lib/types';

interface Props {
  metrics: JobMetrics;
}

export function LLMPanel({ metrics }: Props) {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [visible, setVisible] = useState(false);
  const [generated, setGenerated] = useState(false);

  const generate = useCallback(async () => {
    setLoading(true);
    setVisible(true);
    setText('');

    try {
      const res = await fetch('/api/interpret', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ metrics }),
      });

      if (!res.ok) {
        setText('Failed to get coaching insights. Please try again.');
        setLoading(false);
        return;
      }

      const reader = res.body?.getReader();
      if (!reader) {
        setText('Streaming not supported.');
        setLoading(false);
        return;
      }

      const decoder = new TextDecoder();
      let buffer = '';
      let fullText = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const data = line.slice(6).trim();
          if (data === '[DONE]') continue;

          try {
            const parsed = JSON.parse(data);
            if (parsed.type === 'content_block_delta' && parsed.delta?.text) {
              fullText += parsed.delta.text;
              setText(fullText);
            }
          } catch {
            // skip unparseable lines
          }
        }
      }
      setGenerated(true);
    } catch {
      setText('Failed to connect. Check your connection and try again.');
    } finally {
      setLoading(false);
    }
  }, [metrics]);

  return (
    <div className="border border-gray-800 rounded-xl overflow-hidden">
      <button
        onClick={() => {
          if (!generated) generate();
          else setVisible(!visible);
        }}
        className="w-full flex justify-between items-center px-5 py-3 bg-gray-900 hover:bg-gray-800 transition-colors"
      >
        <span className="font-medium text-sm">
          {loading ? 'Generating coaching insights…' : 'AI Coach'}
        </span>
        <span className="text-xs text-gray-500">
          {!generated ? 'Click to generate' : visible ? '−' : '+'}
        </span>
      </button>
      {visible && (
        <div className="px-5 py-4">
          {text ? (
            <div className="text-sm text-gray-300 whitespace-pre-wrap leading-relaxed">
              {text}
              {loading && <span className="inline-block w-1.5 h-4 bg-blue-400 ml-0.5 animate-pulse" />}
            </div>
          ) : (
            <div className="text-sm text-gray-500">Thinking…</div>
          )}
        </div>
      )}
    </div>
  );
}
