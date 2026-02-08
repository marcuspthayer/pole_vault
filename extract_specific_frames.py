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

def process_video(video_path, output_root):
    """Handles the interactive loop for a single video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_name = video_path.stem
    
    # Create output directory for this video (or shared? User asked for "an output directory", 
    # but grouping by video might be cleaner. Let's put them all in one folder but distinct names)
    # Re-reading: "extract_frames folder inside the selected input directory"
    # let's just make one shared folder "extracted_frames"
    output_dir = output_root # / "extracted_frames" # Passed in
    output_dir.mkdir(parents=True, exist_ok=True)

    # Start at 50%
    current_frame = total_frames // 2
    
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
            controls_text = "A/D: -/+1 | Q/E: -/+10 | Z/C: -/+50 | Space: Save | Esc: Next Video"
            
            draw_text(frame, info_text, pos=(10, 30))
            draw_text(frame, controls_text, pos=(10, h - 20), scale=0.6)
            
            cv2.imshow("Frame Extractor", frame)
            redraw = False

        key = cv2.waitKey(0) & 0xFF

        # Navigation
        if key == ord('d'): # Right arrow / D
            current_frame += 1
            redraw = True
        elif key == ord('a'): # Left arrow / A
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
        
        # Save
        elif key == 32: # Space
            print(f"Saving sequence at frame {current_frame}...")
            # Visual feedback
            temp_frame = frame.copy()
            draw_text(temp_frame, "SAVING...", pos=(w//2 - 50, h//2), color=(0, 0, 255), scale=2.0)
            cv2.imshow("Frame Extractor", temp_frame)
            cv2.waitKey(1) # Force update
            
            extract_frames(cap, current_frame, output_dir, video_name)
            
            # Show "Saved" briefly
            draw_text(temp_frame, "SAVED!", pos=(w//2 - 50, h//2), color=(255, 0, 0), scale=2.0)
            cv2.imshow("Frame Extractor", temp_frame)
            cv2.waitKey(500)
            redraw = True # Reload original frame to clear text
            
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
