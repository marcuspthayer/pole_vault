'use client';

import { useState, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
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
            <div className="text-sm text-gray-300 leading-relaxed
              [&_h1]:text-lg [&_h1]:font-bold [&_h1]:text-white [&_h1]:mt-4 [&_h1]:mb-2
              [&_h2]:text-base [&_h2]:font-semibold [&_h2]:text-white [&_h2]:mt-3 [&_h2]:mb-1
              [&_h3]:text-sm [&_h3]:font-semibold [&_h3]:text-gray-200 [&_h3]:mt-2 [&_h3]:mb-1
              [&_strong]:text-white [&_strong]:font-semibold
              [&_ul]:list-disc [&_ul]:pl-5 [&_ul]:my-1
              [&_ol]:list-decimal [&_ol]:pl-5 [&_ol]:my-1
              [&_li]:my-0.5
              [&_hr]:border-gray-700 [&_hr]:my-3
              [&_p]:my-1">
              <ReactMarkdown>{text}</ReactMarkdown>
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
