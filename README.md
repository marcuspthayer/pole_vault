# VaultSense

AI-powered pole vault biomechanics analysis. Upload a video, select key frames, and get detailed metrics on approach velocity, stride cadence, pole bend, hip height, and predicted bar clearance — with benchmark comparisons and AI coaching insights.

**Live app:** [vaultsense.app](https://vaultsense.app)
**API:** [api.vaultsense.app/health](https://api.vaultsense.app/health)

---

## How It Works

VaultSense uses a **two-pass analysis pipeline**:

**Pass 1** — You upload a video and select three key frames (start of approach, pole plant, end of vault). The backend runs YOLO person detection + MediaPipe pose estimation on every frame, plus YOLO pole segmentation. This produces an annotated video with skeleton overlays and pole masks.

**Pass 2** — You review the annotated frames and select five pole reference frames (phase 1 calibration, phase 2 calibration, plant, max bend start, max bend end). The backend uses these to compute calibrated pole length, bend progression, approach velocity, stride analysis, hip height, and predicted bar clearance.

The result is a full biomechanics report with charts, debug images, downloadable CSVs, and optional AI-generated coaching recommendations.

---

## Architecture & Stack

```
                    Cloudflare (DNS + SSL)
                    /                    \
    vaultsense.app                api.vaultsense.app
         |                               |
      Vercel                          Railway
    (Next.js)                       (FastAPI)
         |                               |
    Supabase                     YOLO + MediaPipe
  (Auth + DB)                   (Video Processing)
```

### Why this architecture

The app has two fundamentally different workloads: a lightweight frontend serving HTML/JS, and a heavyweight backend running ML models (YOLO, MediaPipe, PyTorch) on video frames. Separating them means the frontend scales independently on Vercel's edge network while the ML backend gets dedicated RAM and CPU on Railway.

| Layer | Service | Why |
|---|---|---|
| **Frontend** | [Vercel](https://vercel.com) (Next.js) | Zero-config deploys from GitHub, edge CDN, free hobby tier. Next.js gives us server components for auth and API routes for the LLM proxy. |
| **Backend API** | [Railway](https://railway.app) (FastAPI + Docker) | Supports Docker with persistent volumes for job storage. The container runs CPU-only PyTorch + YOLO + MediaPipe — Railway's 4GB RAM tier handles this without needing a GPU. Auto-deploys on push to main. |
| **Auth & Database** | [Supabase](https://supabase.com) (Postgres) | Built-in email auth with Row-Level Security. Coach/athlete role isolation happens at the DB layer — no custom auth middleware needed. Free tier covers development. |
| **DNS & SSL** | [Cloudflare](https://cloudflare.com) | Free SSL termination, DDoS protection, and DNS management. Proxies traffic to both Vercel and Railway. |
| **LLM Coaching** | [Anthropic API](https://anthropic.com) (Claude Haiku) | Streaming coaching insights generated server-side in a Next.js API route. Runs on Vercel serverless — keeps Railway's RAM budget focused on video processing. |

### Why not a GPU backend?

CPU-only PyTorch processes a typical 5-second vault clip in 1-3 minutes. For a coaching tool where you upload one video at a time, this is acceptable. A GPU instance would cost $50-200/month for marginal speed improvement on short clips. If processing time becomes a bottleneck, Railway supports GPU instances as a drop-in upgrade.

---

## Project Structure

```
pole_vault/
  api/                             # FastAPI backend (deployed to Railway)
    routes/jobs.py                 # All endpoints: create, start, frame, pass2, results
    models/job.py                  # Pydantic schemas (JobConfig, Pass2Config, etc.)
    services/job_runner.py         # ProcessPoolExecutor wrapper around the pipeline
    main.py                        # CORS, health check, lifespan

  pvapp/                           # Core analysis engine (imported by api/)
    pipelines/
      unified_pipeline.py          # Main orchestrator: pose + pole + analysis + rendering
      pose_pipeline.py             # YOLO person detection + MediaPipe pose extraction
      pole_pipeline.py             # YOLO pole segmentation extraction
      hip_analysis.py              # Hip droop computation
    core/
      gait_analysis.py             # Foot strikes, cadence, stride, velocity, calibration
      pole_length.py               # Pole calibration, skeletonization, bend analysis
      analysis.py                  # Hip height time-series, body height computation
      calibration.py               # Height-based scale factor (m/px)

  frontend/                        # Next.js app (deployed to Vercel)
    app/
      analyze/page.tsx             # Two-pass analysis flow (state machine)
      dashboard/page.tsx           # User dashboard
      auth/                        # Login, signup, callback routes
      api/interpret/route.ts       # LLM coaching proxy (server-side)
    components/
      analyze/
        VideoUploader.tsx           # Upload + config (imperial/metric, height, pole)
        InitialFrameSelector.tsx    # Pre-pass-1: select start/plant/end frames
        FrameSelector.tsx           # Pre-pass-2: select 5 pole frames from annotated video
      results/
        ResultsPanel.tsx            # Orchestrates all result sections
        MetricsGrid.tsx             # Benchmark-badged metric cards
        VelocityChart.tsx           # Approach velocity line chart (Recharts)
        StrideChart.tsx             # Stride length bar chart (Recharts)
        BendChart.tsx               # Pole bend progression chart
        DebugImages.tsx             # Calibration debug image grid
        LLMPanel.tsx                # Streaming AI coaching insights
    lib/
      api-client.ts                # Typed fetch wrapper for all API calls
      types.ts                     # TypeScript types mirroring backend schemas
      benchmarks.ts                # Biomechanics benchmark thresholds
      supabase/                    # Auth client (browser + server)

  Dockerfile                       # Multi-stage build: python:3.11-slim + CPU PyTorch
  railway.toml                     # Railway config (healthcheck, volume mount)
  docker-compose.dev.yml           # Local development (API on localhost:8000)
  requirements-api.txt             # Python deps for the API container

  _archive/                        # Archived Streamlit app (reference only)
  step_detection/                  # ML training data for foot strike detection
```

---

## Metrics Produced

| Metric | Description | Source |
|---|---|---|
| **Peak velocity** | Maximum approach speed (m/s) | Hip midpoint displacement between frames |
| **Avg velocity** | Mean approach speed | Same, averaged over approach window |
| **Takeoff velocity** | Speed at the plant frame | Last velocity measurement before plant |
| **Velocity retention** | Takeoff / Peak ratio | Computed in frontend |
| **Cadence** | Steps per minute during approach | Foot strike count / approach duration |
| **Stride length** | Per-step distance (cm/in) | Ankle displacement, calibrated via athlete height |
| **Pole bend** | Chord-to-length ratio at max bend (%) | Pole mask skeletonization + arc-length |
| **Peak hip height** | Highest point of hips during vault (m) | Pose landmark tracking, height-calibrated |
| **Predicted clearance** | Estimated bar height the athlete could clear | Derived from hip peak + body geometry |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/jobs` | Upload video + config, optionally auto-start |
| `POST` | `/api/jobs/:id/start` | Set start/plant/end frames and begin processing |
| `GET` | `/api/jobs/:id` | Poll job status and metrics |
| `GET` | `/api/jobs/:id/stream` | SSE stream of progress updates |
| `GET` | `/api/jobs/:id/frame` | Extract single frame as JPEG (source=input\|output) |
| `POST` | `/api/jobs/:id/pass2` | Submit 5 pole frame selections for final analysis |
| `GET` | `/api/jobs/:id/results/:file` | Download output video, CSVs, or debug images |
| `DELETE` | `/api/jobs/:id` | Delete job and all files |
| `GET` | `/health` | Health check |

---

## Local Development

### Prerequisites
- Docker Desktop (for the API)
- Node.js 18+ (for the frontend)

### Setup

```bash
# Start the API locally (from repo root):
docker compose -f docker-compose.dev.yml build
docker compose -f docker-compose.dev.yml up

# In another terminal, start the frontend:
cd frontend
cp .env.local.example .env.local   # fill in your Supabase keys
npm install
npm run dev
```

The frontend runs at `http://localhost:3000` and calls the API at `http://localhost:8000`. Set `NEXT_PUBLIC_API_URL=http://localhost:8000` in `.env.local`.

### Without Docker

If you don't have Docker, you can point the frontend at the production API:

```bash
cd frontend
# In .env.local, set:
# NEXT_PUBLIC_API_URL=https://api.vaultsense.app
npm run dev
```

This lets you develop the frontend without running the ML pipeline locally.

---

## Database Schema

Supabase Postgres with Row-Level Security:

- **profiles** — user accounts with `role` (coach/athlete) and optional `coach_id` link
- **sessions** — saved analysis results (metrics JSON, config, timestamps)

Coaches can view their linked athletes' sessions. Athletes can only see their own data. Enforced at the database level via RLS policies.

---

## Environment Variables

### Frontend (Vercel / `.env.local`)
| Variable | Description |
|---|---|
| `NEXT_PUBLIC_API_URL` | Backend API URL (`https://api.vaultsense.app`) |
| `NEXT_PUBLIC_SUPABASE_URL` | Supabase project URL |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | Supabase publishable key |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase service role key (server-side only) |
| `ANTHROPIC_API_KEY` | Anthropic API key for AI coaching |

### Backend (Railway)
| Variable | Description |
|---|---|
| `PORT` | Injected by Railway (typically 8080) |
| `DATA_DIR` | Job storage directory (`/data`) |
| `ALLOWED_ORIGINS` | CORS origins (comma-separated) |

---

## License

Proprietary. All rights reserved.
