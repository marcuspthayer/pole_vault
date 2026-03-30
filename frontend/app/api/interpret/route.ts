import { NextRequest } from 'next/server';
import type { JobMetrics } from '@/lib/types';

export async function POST(req: NextRequest) {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    return new Response(JSON.stringify({ error: 'ANTHROPIC_API_KEY not configured' }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  const { metrics } = (await req.json()) as { metrics: JobMetrics };

  const retention =
    metrics.peak_velocity_ms && metrics.takeoff_velocity_ms
      ? ((metrics.takeoff_velocity_ms / metrics.peak_velocity_ms) * 100).toFixed(0)
      : 'N/A';

  const prompt = `You are an expert pole vault biomechanics coach. Analyze the following metrics from a video analysis session and give specific, actionable training recommendations.

ATHLETE METRICS:
Approach run:
  - Peak velocity: ${metrics.peak_velocity_ms?.toFixed(2) ?? 'N/A'} m/s
  - Avg velocity: ${metrics.avg_velocity_ms?.toFixed(2) ?? 'N/A'} m/s
  - Takeoff velocity: ${metrics.takeoff_velocity_ms?.toFixed(2) ?? 'N/A'} m/s (retention: ${retention}%)
  - Cadence: ${metrics.cadence_spm?.toFixed(0) ?? 'N/A'} steps/min
Jump:
  - Max pole bend (chord ratio): ${metrics.max_pole_bend_pct?.toFixed(1) ?? 'N/A'}% of pole length
    (lower = more bent; elite target: 70-75%)
  - Peak hip height: ${metrics.max_hip_height_m?.toFixed(2) ?? 'N/A'} m
  - Predicted bar clearance: ${metrics.predicted_clear_m?.toFixed(2) ?? 'N/A'} m

ELITE BENCHMARKS (men):
  - Peak velocity: 9.2-10.1 m/s
  - Velocity retention: >93%
  - Cadence: 295-315 spm
  - Pole bend: 70-75% chord ratio
  - Predicted clearance: >5.2 m

Rules:
- 2-3 priority improvements, one drill or cue each
- Note what's working well
- Under 250 words, plain language
- Use both metric and imperial units where relevant`;

  const response = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
    },
    body: JSON.stringify({
      model: 'claude-haiku-4-5-20251001',
      max_tokens: 500,
      stream: true,
      messages: [{ role: 'user', content: prompt }],
    }),
  });

  if (!response.ok) {
    const err = await response.text();
    return new Response(JSON.stringify({ error: err }), { status: response.status });
  }

  // Forward the SSE stream directly
  return new Response(response.body, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
    },
  });
}
