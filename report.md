# VaultSense: AI-Powered Pole Vault Biomechanics Analysis

**Marcus Thayer**
CH EN 426 — Machine Learning for Engineers
Brigham Young University
Professor John Hedengren
Winter 2026

---

## Links

- **Live Application:** [https://vaultsense.app](https://vaultsense.app)
- **API Health Check:** [https://api.vaultsense.app/health](https://api.vaultsense.app/health)
- **GitHub Repository:** [https://github.com/marcuspthayer/pole_vault](https://github.com/marcuspthayer/pole_vault) *(private — access available upon request)*
- **Test Video:** `ean low fps.mp4` (included with submission)

---

## App Description

VaultSense is a web application that performs automated biomechanics analysis on pole vault videos. An athlete or coach uploads a video of a vault, selects key frames to guide the analysis, and receives detailed performance metrics — including approach velocity, stride cadence, pole bend percentage, and predicted bar clearance height — along with AI-generated coaching recommendations. The app replaces manual video review with a reproducible, data-driven workflow that quantifies technique and benchmarks it against known performance thresholds.

---

## Intended Users

The primary users are pole vault coaches and athletes who want objective, quantitative feedback on vault technique. A coach can upload game-day or practice footage and receive metrics within minutes, without needing any software installed locally. The app is also useful for sports science researchers studying approach mechanics and for athletic programs that want to track athlete development over time.

---

## Features and Interactivity

### Video Upload and Configuration

Users upload a pole vault video (MP4 or MOV) and enter athlete parameters such as height and pole length. A unit toggle allows input in either imperial or metric. A confidence threshold slider controls the sensitivity of the person-detection model.

### Two-Pass Analysis Pipeline

The analysis is split into two guided passes:

1. **Pass 1 — Pose Extraction:** The user scrubs through the uploaded video and marks three key frames (start of approach, pole plant, and end of vault). The backend runs YOLO person detection and MediaPipe pose estimation to extract 33 skeletal landmarks per frame, producing an annotated skeleton-overlay video.
2. **Pass 2 — Detailed Metrics:** The user reviews the annotated video and selects five pole-reference frames used for calibration. The backend then computes all biomechanics metrics: approach velocity, stride length, cadence, pole bend, peak hip height, and predicted bar clearance.

### Interactive Results Dashboard

After analysis, the results panel displays:

- **Metrics Grid** — Cards for each metric with color-coded benchmark badges (green/yellow/red) comparing the athlete's numbers to known performance standards.
- **Velocity Chart** — A line chart (built with Recharts) showing approach velocity over time from the start of the run to the plant.
- **Stride Chart** — A bar chart showing stride length for each detected step.
- **Pole Bend Chart** — A timeline of pole bend percentage through the vault phases.
- **Debug Images** — A grid of calibration frames showing pole tip and hand-grip segments used during pole-length estimation.
- **AI Coaching Panel** — Streaming AI-generated coaching insights powered by the Anthropic Claude API. The model receives the athlete's metrics and returns targeted, actionable recommendations.

### User Accounts and History

Users can create accounts (coach or athlete roles) via Supabase authentication. Coaches can link to athletes and view their analysis history. All past sessions are saved and accessible from a dashboard.

---

## Technical Architecture

### Machine Learning Models

| Model                                            | Purpose                                                                     | Framework             |
| ------------------------------------------------ | --------------------------------------------------------------------------- | --------------------- |
| YOLO 11n (`yolo11n.pt`)                        | Person detection — isolates the athlete in each frame                      | PyTorch / Ultralytics |
| Custom YOLO Segmentation (`pole_detect_v3.pt`) | Pole mask detection — segments the pole from the background                | PyTorch / Ultralytics |
| MediaPipe Pose (0.10.9)                          | Skeletal landmark extraction — 33 keypoints per frame                      | MediaPipe             |
| MLP Classifier (`best_step_model.joblib`)      | Foot-strike detection — identifies ground contact frames for gait analysis | scikit-learn          |

### Metrics Computation

- **Approach Velocity:** Horizontal hip displacement between consecutive frames, converted from pixels to meters per second using a height-based calibration factor, smoothed with a 5-frame moving average.
- **Stride Length & Cadence:** The MLP foot-strike classifier identifies ground-contact frames. Stride length is the calibrated distance between successive ankle positions at contact. Cadence is the step count divided by approach duration.
- **Pole Bend:** The pole mask is skeletonized to extract a centerline. Bend is computed as the chord-to-arc-length ratio, expressed as a percentage of total pole length.
- **Peak Hip Height & Predicted Clearance:** The hip landmark's vertical position is tracked through the vault. Peak height is converted to meters via calibration and combined with body geometry to estimate clearable bar height.
- **Height Calibration:** The app measures the athlete's body height (nose to ankle) in pixels and divides by their reported height in meters to establish a `scale_m_per_px` factor applied to all spatial measurements.

### Technology Stack

| Layer            | Technology                                 | Purpose                                                 |
| ---------------- | ------------------------------------------ | ------------------------------------------------------- |
| Frontend         | Next.js (React) + TypeScript               | User interface, state management, server-side rendering |
| Charts           | Recharts                                   | Interactive metric visualizations                       |
| Backend API      | FastAPI (Python)                           | Video processing endpoints, job management              |
| Auth & Database  | Supabase (PostgreSQL + Row-Level Security) | User accounts, session history, access control          |
| AI Coaching      | Anthropic Claude API                       | Streaming coaching recommendations from metrics         |
| Containerization | Docker                                     | Reproducible backend environment                        |

### Key Python Dependencies

```
opencv-contrib-python-headless
numpy>=1.24,<2
pandas
mediapipe==0.10.9
ultralytics
imageio-ffmpeg
scikit-learn>=1.0
joblib>=1.2
fastapi
uvicorn[standard]
python-multipart
```

---

## Deployment

Rather than deploying a single Streamlit app to Streamlit Community Cloud, VaultSense is deployed as a production microservices architecture across multiple platforms. This was necessary because the app requires GPU-class CPU processing for video analysis (YOLO + MediaPipe), persistent job storage, user authentication, and a responsive modern frontend — requirements that exceed what Streamlit Community Cloud supports.

### Deployment Architecture

```
         Cloudflare (DNS + SSL)
          /                  \
  vaultsense.app         api.vaultsense.app
       |                        |
    Vercel                   Railway
   (Next.js)            (FastAPI + Docker)
       |                        |
   Supabase              YOLO + MediaPipe
 (Auth + DB)           (Video Processing)
```

- **Frontend (Vercel):** The Next.js application is deployed to Vercel with automatic deployments on every push to the `main` branch. Vercel handles server-side rendering, static optimization, and edge caching.
- **Backend API (Railway):** The FastAPI backend runs inside a Docker container on Railway with a persistent volume mounted at `/data` for job file storage. The container uses a CPU-only PyTorch build to keep the image size manageable. A health check endpoint at `/health` enables Railway's automatic restart on failure.
- **Authentication & Database (Supabase):** User accounts, roles (coach/athlete), and saved analysis sessions are stored in a Supabase-hosted PostgreSQL database with row-level security policies enforcing access control.
- **DNS & SSL (Cloudflare):** Cloudflare manages DNS routing for both the frontend and API subdomains, provides SSL termination, and adds DDoS protection.

### Job Lifecycle

Video processing jobs follow a state machine:

```
created → queued → running → pass1_done → pass2 → complete
                                                  ↘ failed
```

Jobs are stored on the Railway persistent volume and auto-cleaned after 30 minutes. Real-time progress is streamed to the frontend via Server-Sent Events (SSE).

---

## Using the Test Video

The included test video (`ean low fps.mp4`) can be used to verify the app's functionality:

1. Navigate to [vaultsense.app](https://vaultsense.app) and create an account or log in.
2. Click **Analyze** and upload `ean low fps.mp4`.
3. Enter the athlete's height (6 feet 0 inches) and optionally a pole length.
4. Select the three key frames when prompted (start of approach, pole plant, end of vault).
5. After Pass 1 completes, review the skeleton-annotated video and select the five pole-reference frames.
6. After Pass 2 completes, explore the results dashboard — metrics, charts, debug images, and AI coaching insights.

---

## What I Learned

Building and deploying VaultSense taught me several things that go well beyond what a simple Streamlit deployment would cover:

**Version Control at Scale:** Managing a full-stack application with a Python backend, TypeScript frontend, ML model files, and Docker configuration in a single Git repository required careful `.gitignore` management and attention to repository organization. Large binary files (YOLO models) needed to be tracked deliberately.

**Reproducible Environments:** Pinning dependency versions — especially `numpy<2` for MediaPipe compatibility and `mediapipe==0.10.9` specifically — was critical. A single unpinned dependency caused build failures on Railway that took significant debugging to resolve. Writing a `requirements.txt` that works identically in local development and in a Docker container on a cloud platform is a non-trivial skill.

**Cloud Deployment Complexity:** Deploying across multiple services (Vercel, Railway, Supabase, Cloudflare) meant understanding environment variables, CORS configuration, health checks, persistent storage, and DNS routing. Each platform has its own deployment model and failure modes. Learning to diagnose issues across service boundaries — "is the bug in the frontend, the API, or the database?" — is essential for real-world engineering.

**ML Model Serving:** Packaging PyTorch and YOLO models into a Docker container that runs on a cloud platform with limited RAM (4 GB) required careful optimization: CPU-only builds, single-worker Uvicorn, and process pool management to avoid memory exhaustion.

**Calibration and Accuracy:** Converting pixel measurements to real-world units (meters, m/s) depends entirely on the quality of the height calibration step. Small errors in the athlete's reported height propagate through every metric. This reinforced the importance of input validation and transparent uncertainty in any data pipeline.

---

## Repository Structure

```
pole_vault/
├── api/                        # FastAPI backend
│   ├── main.py                # App entry point, CORS, lifespan
│   ├── routes/jobs.py         # REST endpoints for job lifecycle
│   ├── models/job.py          # Pydantic request/response schemas
│   └── services/job_runner.py # Process pool executor, job management
├── pvapp/                      # Core analysis engine
│   ├── pipelines/             # Unified, pose, pole, and overlay pipelines
│   ├── core/                  # Detector, gait analysis, pole length, calibration
│   └── utils/                 # Drawing helpers
├── step_detection/            # ML foot-strike detection
│   ├── inference.py           # Model loading and prediction
│   ├── train_and_compare.py   # Training script
│   └── models/                # Trained MLP classifier
├── frontend/                   # Next.js application
│   ├── app/                   # Pages (home, analyze, dashboard, auth)
│   ├── components/            # UI components (upload, results, charts)
│   └── lib/                   # API client, types, benchmarks
├── Dockerfile                  # Multi-stage Docker build
├── docker-compose.dev.yml     # Local development setup
├── requirements-api.txt       # Python API dependencies
├── railway.toml               # Railway deployment config
├── yolo11n.pt                 # Person detection model
├── pole_detect_v3.pt          # Pole segmentation model
└── report.md                  # This report
```
