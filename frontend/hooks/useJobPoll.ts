'use client';

import { useEffect, useRef, useState } from 'react';
import { apiClient } from '@/lib/api-client';
import type { JobResponse } from '@/lib/types';

export function useJobPoll(jobId: string | null, active: boolean) {
  const [job, setJob] = useState<JobResponse | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (!jobId || !active) {
      if (intervalRef.current) clearInterval(intervalRef.current);
      return;
    }

    async function poll() {
      if (!jobId) return;
      try {
        const data = await apiClient.getJob(jobId);
        setJob(data);
        if (data.status === 'complete' || data.status === 'failed') {
          if (intervalRef.current) clearInterval(intervalRef.current);
        }
      } catch (e) {
        console.error('Poll error:', e);
      }
    }

    poll();
    intervalRef.current = setInterval(poll, 2000);
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [jobId, active]);

  return job;
}
