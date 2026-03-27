# Pole Vault Step Labeler

Standalone Streamlit app for labeling ground-contact (step) events in pole vault approach-run videos. Part of the **Step Detection & Stride Analysis** ML project.

## Quick Start

From the **repo root** (`polevault/`):

```bash
streamlit run step_detection/step_labeler_app.py
```

> The app imports modules from `pvapp/` — no additional installs needed beyond the main project's `requirements.txt`.

## Workflow

1. **Upload** a pole vault video (MP4 / MOV / AVI).
2. **Set start & end frames** to trim to the approach run.
3. **Process** — runs YOLO person detection + MediaPipe pose estimation on the selected window.
4. **Navigate** frames with the slider or jump buttons (±1, ±10, ±20, ±50).
5. **Label** each ground contact by choosing left/right foot and clicking "Mark Step."
6. **Auto-Suggest** (unlocked after 3 videos saved) — pre-populates labels using the heuristic foot-strike detector.
7. **Save** — writes all data to `step_detection/data/<video_name>/`.

## Output Format

Each saved video produces a folder in `step_detection/data/<video_name>/` containing:

| File | Description |
|---|---|
| `labels.csv` | `frame, time_sec, side, source, x_ankle, y_ankle` |
| `landmarks.json` | Full 33-joint MediaPipe landmarks for each labeled frame |
| `metadata.json` | Video metadata (fps, resolution, processing window, date) |
| `labels_120fps.csv` | Downsampled labels simulating 120 fps capture |
| `labels_60fps.csv` | Downsampled labels simulating 60 fps capture |
| `labels_30fps.csv` | Downsampled labels simulating 30 fps capture |

> **FPS Downsampling**: High-FPS labels are automatically decimated (240→120→60→30) to produce training data that makes the model robust across frame rates.
