import sys
from pvapp.pipelines.unified_pipeline import run_unified_pipeline

video_path = "videos/ean.MOV"

try:
    print(f"Running pipeline on {video_path}")
    res = run_unified_pipeline(
        video_path,
        enable_pose=True,
        enable_hip=False,
        enable_step=True,
        enable_pole=True,
        start_frame=0,
        plant_frame=150,
        end_frame=200
    )
    print(f"Success! Output saved to: {res[0]}")
except Exception as e:
    print(f"Pipeline failed: {e}")
