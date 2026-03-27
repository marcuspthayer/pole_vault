# ⚡ AlphaForm: Pole Vault Analysis

**AI-powered biomechanical analysis for pole vault approach runs.**

AlphaForm is a Streamlit-based video analysis tool that combines **YOLO object detection**, **MediaPipe pose estimation**, and **computer vision** to provide coaches and athletes with detailed, frame-by-frame biomechanical insights from pole vault videos.

---

## ▶ Features

### ▪ Skeleton Overlay
Full-body pose estimation with real-time skeleton rendering on video. Uses YOLO for athlete detection and MediaPipe Pose for 33-landmark body tracking, with a custom `PoseStabilizer` to eliminate jitter and left/right limb swaps.

### ▪ Approach Hip Analysis
Measures how much the athlete's hips sag during the approach run — a key indicator of fatigue or form breakdown. Reports:
- **Worst-case hip droop** (worst 5% of frames as % of body height)
- **Hip sag trend** (late approach vs. early baseline)
- Visual hip path overlay on the output video

### ▪ Gait / Step Analysis
Detects individual foot strikes from ankle trajectory data and computes:
- **Step count** and **cadence** (steps per minute)
- **Stride length in cm and inches** — automatically calibrated using the athlete's height input. The system finds the frame near the plant where the athlete is most upright (eye-to-ankle measurement with skull correction), establishes a meters-per-pixel scale, and converts all distances to real-world units. A visual height-reference indicator is drawn on the calibration frame in the output video.
- **Max hip height** — measures the peak vertical position of the hip midpoint between the plant and end frames, relative to the lowest foot point (heel/toe) near the plant. Displayed in feet/inches and meters both on-video (persistent magenta dot + label from the peak frame onward) and in the Streamlit results panel.
- **Foot path visualization** with detected contact points
- Missing step interpolation for occluded frames
- Exportable CSV with per-frame ankle coordinates and step flags

### ▪ Pole Detection & Bend Tracking
Uses a custom-trained **YOLOv8 segmentation model** to detect and segment the pole in every frame:
- **Two-phase pole calibration** via skeletonization + arc-length consensus clustering
- **Pole bend progression** — tracks chord-length ratio from plant to peak bend
- Debug visualizations for calibration frames and max-bend state

### ▪ Data Export
- Annotated output video (MP4) with all selected overlays
- Stride analysis plots (stride length vs. step number)
- Ankle path plots with foot-strike markers
- Gait analysis CSV export
- Downloadable debug images for pole calibration and bend

---

## ▶ Project Structure

```
polevault/
├── polevaultapp_modular.py      # Streamlit app (main entry point)
├── requirements.txt             # Python dependencies
├── pole_detector_v3.pt            # Custom YOLOv8 pole segmentation model
├── yolo11n.pt                   # YOLO person detection model
├── args.yaml                    # Model training configuration
│
├── pvapp/                       # Core application package
│   ├── __init__.py
│   ├── pose.py                  # Pose detection helpers (ROI + MediaPipe)
│   ├── render.py                # Drawing utilities (skeleton rendering)
│   ├── logging_utils.py         # Logging configuration
│   │
│   ├── core/                    # Analysis modules
│   │   ├── analysis.py          # Hip height time-series & hip-drop computation
│   │   ├── detector.py          # PersonDetector (YOLO wrapper with fallback)
│   │   ├── gait_analysis.py     # Foot strike detection, cadence, stride length, height calibration, max hip height
│   │   ├── pole_length.py       # Pole calibration, skeletonization, bend analysis
│   │   ├── pole_manager.py      # Pole detection state management
│   │   └── pose_stabilization.py# PoseStabilizer (smoothing, anti-swap logic)
│   │
│   ├── pipelines/               # End-to-end processing pipelines
│   │   ├── unified_pipeline.py  # Main pipeline orchestrating all analyses
│   │   ├── pose_pipeline.py     # Pose extraction pipeline
│   │   ├── pole_pipeline.py     # Pole extraction pipeline
│   │   ├── hip_analysis.py      # Standalone hip analysis pipeline
│   │   ├── hip_overlay.py       # Hip visualization overlay
│   │   └── skeleton_overlay.py  # Standalone skeleton overlay pipeline
│   │
│   └── utils/
│       └── cv_utils.py          # OpenCV drawing helpers
│
├── legacy/                      # Archived earlier implementations
├── GaitKeeper_Reference_Files/  # Reference code (TypeScript gait metrics)
├── Sprint_Analysis_Reference/   # Reference code (sprint analysis functions)
└── debug_output/                # Generated debug images (gitignored)
```

---

## ▶ Getting Started

### Prerequisites
- **Python 3.10+**
- **FFmpeg** (optional, for H.264 re-encoding of output videos)
- A CUDA-capable GPU is recommended but not required

### Installation

```bash
# Clone the repository
git clone https://github.com/alphapeakio/polevault.git
cd polevault

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Running the App

```bash
streamlit run polevaultapp_modular.py
```

The app will open in your browser at `http://localhost:8501`.

---

## ▶ Usage

1. **Upload** a pole vault video (MP4, MOV, or AVI)
2. **Select analysis types** from the sidebar:
   - Skeleton Overlay
   - Hip Analysis
   - Step Analysis
   - Pole Analysis
3. **Configure settings:**
   - Athlete height (for stride calibration)
   - Pole length (optional, for real-world unit conversion)
   - Step detection sensitivity (Advanced Settings)
4. **Set frame markers** — start, plant, and end frames to define the analysis window
5. **Run Analysis** — the unified pipeline processes the video and generates results
6. **Review outputs** — annotated video, plots, metrics, and downloadable data

---

## ▶ Tech Stack

| Component | Technology |
|---|---|
| **UI / App Framework** | [Streamlit](https://streamlit.io/) |
| **Pose Estimation** | [MediaPipe Pose](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) (33 landmarks) |
| **Object Detection** | [YOLO (Ultralytics)](https://docs.ultralytics.com/) — person detection + pole segmentation |
| **Computer Vision** | [OpenCV](https://opencv.org/) |
| **Data Processing** | NumPy, Pandas, Matplotlib |

---

## ▶ Configuration

### Sidebar Controls

| Setting | Description | Default |
|---|---|---|
| Skeleton Overlay | Draw pose skeleton on output video | ✅ On |
| Approach Hip Analysis | Compute hip droop metrics during approach | Off |
| Step Analysis | Detect foot strikes, compute cadence & stride | ✅ On |
| Max Hip Height | Measure peak hip height during bar clearance | ✅ On |
| Pole Analysis | Detect/track pole, compute bend | ✅ On |
| Athlete Height | Used for height-based stride calibration (eye-to-ankle + skull correction) | 5′7″ |
| Pole Length | Optional additional calibration via pole pixel measurement | Off |
| Min Lift Sensitivity | Vertical threshold for step detection | 0.015 |
| Min Step Distance | Minimum frames between detected steps | 10 |

---

## ▶ License

This project is proprietary to **AlphaPeak**.
