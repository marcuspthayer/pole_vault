import cv2
import os
import sys
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

def select_directory():
    """Opens a dialog to select the input directory."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    folder_path = filedialog.askdirectory(title="Select Folder Containing Videos")
    root.destroy()
    return folder_path

def get_video_files(directory):
    """Finds all video files in the given directory."""
    video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
    return [
        Path(p) for p in Path(directory).rglob("*") 
        if p.suffix.lower() in video_extensions
    ]

def draw_text(img, text, pos=(10, 30), color=(0, 255, 0), scale=0.7, thickness=2):
    """Helper to draw text on the image."""
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2) # Shadow
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

def extract_frames(cap, center_frame_idx, output_dir, video_name, neighbor_count=2):
    """Extracts the center frame and its neighbors."""
    
    saved_count = 0
    start_f = max(0, center_frame_idx - neighbor_count)
    end_f = center_frame_idx + neighbor_count + 1 # exclusive
    
    # We need to seek for each frame to be safe, or seek to start and read sequentially
    # Reading sequentially from start_f is usually faster/safer if frames are close
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    
    for f_idx in range(start_f, end_f):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Filename: video_name_frame_000123.jpg
        out_path = output_dir / f"{video_name}_frame_{f_idx:06d}.jpg"
        cv2.imwrite(str(out_path), frame)
        saved_count += 1
        
    print(f"Saved {saved_count} frames centered at {center_frame_idx} to {output_dir}")

def extract_incremental_frames(cap, start_f, end_f, fps, output_dir, video_name):
    """Extracts frames every 0.25 seconds between start and end."""
    if start_f is None or end_f is None:
        print("Start or End frame not set.")
        return
    
    if start_f > end_f:
        start_f, end_f = end_f, start_f
        
    step = max(1, int(fps * 0.25))
    saved_count = 0
    
    for f_idx in range(start_f, end_f + 1, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret:
            break
            
        out_path = output_dir / f"{video_name}_incremental_{f_idx:06d}.jpg"
        cv2.imwrite(str(out_path), frame)
        saved_count += 1
        
    print(f"Saved {saved_count} incremental frames from {start_f} to {end_f} to {output_dir}")

def process_video(video_path, output_root):
    """Handles the interactive loop for a single video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30.0 # Fallback
    video_name = video_path.stem
    
    output_dir = output_root
    output_dir.mkdir(parents=True, exist_ok=True)

    current_frame = total_frames // 2
    start_frame = None
    end_frame = None
    
    cv2.namedWindow("Frame Extractor", cv2.WINDOW_NORMAL)
    
    redraw = True
    
    while True:
        # Clamp frame
        current_frame = max(0, min(current_frame, total_frames - 1))
        
        if redraw:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame.")
                break
            
            # Draw HUD
            h, w = frame.shape[:2]
            info_text = f"Frame: {current_frame}/{total_frames} ({video_name})"
            interval_text = f"Start: {start_frame if start_frame is not None else '-'} | End: {end_frame if end_frame is not None else '-'}"
            controls_text = "A/D: -/+1 | Q/E: -/+10 | Z/C: -/+50 | [: Set Start | ]: Set End | Enter: Incr. Extr. | Space: Save 5 | Esc: Next"
            
            draw_text(frame, info_text, pos=(10, 30))
            draw_text(frame, interval_text, pos=(10, 60), color=(255, 255, 0))
            draw_text(frame, controls_text, pos=(10, h - 20), scale=0.5)
            
            # Scaling 200%
            disp_frame = cv2.resize(frame, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
            cv2.imshow("Frame Extractor", disp_frame)
            redraw = False

        key = cv2.waitKey(0) & 0xFF

        # Navigation
        if key == ord('d'): # D
            current_frame += 1
            redraw = True
        elif key == ord('a'): # A
            current_frame -= 1
            redraw = True
        elif key == ord('e'): # E
            current_frame += 10
            redraw = True
        elif key == ord('q'): # Q
            current_frame -= 10
            redraw = True
        elif key == ord('c'): # C
            current_frame += 50
            redraw = True
        elif key == ord('z'): # Z
            current_frame -= 50
            redraw = True
        
        # Set Start/End
        elif key == ord('['):
            start_frame = current_frame
            redraw = True
        elif key == ord(']'):
            end_frame = current_frame
            redraw = True
            
        # Incremental Extraction
        elif key == 13: # Enter
            if start_frame is not None and end_frame is not None:
                print(f"Running incremental extraction from {start_frame} to {end_frame}...")
                extract_incremental_frames(cap, start_frame, end_frame, fps, output_dir, video_name)
                # Visual feedback
                cv2.putText(disp_frame, "INCREMENTAL SAVED!", (w - 200, h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
                cv2.imshow("Frame Extractor", disp_frame)
                cv2.waitKey(500)
                redraw = True
            else:
                print("Please set both Start ([) and End (]) frames.")

        # Save Specific
        elif key == 32: # Space
            print(f"Saving sequence at frame {current_frame}...")
            # Visual feedback on disp_frame (already scaled)
            cv2.putText(disp_frame, "SAVING...", (w - 100, h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.imshow("Frame Extractor", disp_frame)
            cv2.waitKey(1)
            
            extract_frames(cap, current_frame, output_dir, video_name)
            
            cv2.putText(disp_frame, "SAVED!", (w - 100, h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
            cv2.imshow("Frame Extractor", disp_frame)
            cv2.waitKey(500)
            redraw = True
            
        # Exit / Next
        elif key == 27: # Esc
            break
            
    cap.release()
    cv2.destroyWindow("Frame Extractor")

def main():
    print("Select a directory containing videos...")
    input_dir_str = select_directory()
    
    if not input_dir_str:
        print("No directory selected. Exiting.")
        return

    input_dir = Path(input_dir_str)
    video_files = get_video_files(input_dir)
    
    if not video_files:
        print(f"No video files found in {input_dir}")
        return

    print(f"Found {len(video_files)} videos.")
    
    # Output directory
    output_dir = input_dir / "extracted_frames"
    if not output_dir.exists():
        output_dir.mkdir()
    
    print(f"Output directory: {output_dir}")
    print("Controls:")
    print("  A / D : Previous / Next Frame")
    print("  Q / E : -10 / +10 Frames")
    print("  Z / C : -50 / +50 Frames")
    print("  Space : Save Frame + Neighbors")
    print("  Esc   : Skip to Next Video (or Exit if last)")

    for i, video_path in enumerate(video_files):
        print(f"\n[{i+1}/{len(video_files)}] Processing: {video_path.name}")
        process_video(video_path, output_dir)
        
    print("\nDone! All videos processed.")

if __name__ == "__main__":
    main()
