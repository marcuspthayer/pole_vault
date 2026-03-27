import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from src.db.os.database_paths import basename, is_directory, join_paths, list_directory, make_dirs, path_exists, remove_directory, rename


def zoom_preprocess(fn,destination_folder,fps, movement_type): #src/db src/util
    """
    Processes a video to detect, track, and stabilize a person, saving a zoomed video.

    Uses YOLO to detect and track a person, applies smoothing for stabilization, and outputs
    a video at 15 FPS in the destination folder. Processing varies by folder:
    - 'Posterior Distance Analysis': Dynamic sizing with aspect ratio preservation.
    - 'Max Velocity Analysis': Uses max bounding box, trims 15 frames from start/end.

    Args:
        fn (str): Input video filename.
        destination_folder (str): Folder for input/output video.
        fps (int): Output FPS (fixed at 15).

    Returns:
        None: Saves processed video to destination_folder, cleans up 'zoom' subfolder.

    Raises:
        FileNotFoundError: If video or model files are missing.
        Exception: For errors in processing or file operations, logged to console.
    """

    if "Posterior Distance Analysis" in destination_folder:
        process_post_distance(fn, destination_folder, fps)
    elif "Max Velocity Analysis" in destination_folder:
        process_max_velocity(fn, destination_folder, fps)

    if movement_type == "horizontal":
        process_max_velocity(fn, destination_folder, fps)
    elif movement_type == "vertical":
        process_post_distance(fn, destination_folder, fps)
    else:
        pass

    zoom_folder = join_paths(destination_folder, 'zoom')  # Ensure zoom_folder is a directory
    video_name = basename(fn)  # Extract just the filename
    
    # Extract base name and extension to add _zoom suffix
    import os
    video_base, video_ext = os.path.splitext(video_name)
    zoom_video_name = f"{video_base}_zoom{video_ext}"
    
    video_path = join_paths(zoom_folder, video_name)
    zoom_output_path = join_paths(destination_folder, zoom_video_name)
    
    # Remove old zoom video if it exists to ensure fresh processing
    if path_exists(zoom_output_path):
        try:
            from src.db.os.database_paths import remove
            remove(zoom_output_path)
            print(f"Removed old zoom video: {zoom_video_name}")
        except Exception as e:
            print(f"Warning: Could not remove old zoom video {zoom_output_path}: {e}")
    
    if path_exists(video_path):
        try:
            # Rename zoomed video with _zoom suffix (preserves original video)
            rename(video_path, zoom_output_path)
            print(f"Zoomed video saved to {zoom_output_path} (original preserved)")
        except Exception as e:
            print(f"Error moving zoomed video: {e}")
    else:
        print(f"Video not found in {zoom_folder}: {video_name}")
    
    # Remove the zoom folder if it's empty
    try:
        if is_directory(zoom_folder) and not list_directory(zoom_folder):
            remove_directory(zoom_folder)
            print(f"Zoom folder removed: {zoom_folder}")
    except Exception as e:
        print(f"Error removing zoom folder: {e}")

    print(f"Zoomed video saved to {zoom_output_path}")



def process_max_velocity(fn,destination_folder,fps):
    model_path = join_paths('required','models','yolo11n.pt')
    print(f"fn: {fn}")
    print(f"Loading YOLO model from: {model_path}")
    
    if not path_exists(model_path):
        print(f"ERROR: YOLO model not found at {model_path}")
        return
    
    try:
        model = YOLO(model_path)
        print("YOLO model loaded successfully")
    except Exception as e:
        print(f"ERROR: Failed to load YOLO model: {e}")
        return
    # model = YOLO(os.path.join('required','models','yolo11n.pt'))

    # Smaller buffer keeps the crop tighter around the runner
    buffer_ratio = 0.2
    output_size = (720, 960)  # width, height

    # make video and output paths
    video_path, output_path = handle_paths(destination_folder, fn)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the original video's FPS and frame dimensions
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 15  # Output video frame rate
    out = cv2.VideoWriter(output_path, fourcc, fps, output_size)

    # Variables for tracking
    # Slightly smaller smoothing window so tracking locks on sooner
    smoothing_window = 20  # Moving average window size for smoothing
    positions = []
    all_frames = []
    frame_index = 0

    # First loop: Identify largest person object
    frame_index = identify_largest_person(cap, model, frame_width, frame_height, positions, buffer_ratio, frame_index)

    cap.release()

    # Check if any person was detected
    if not positions:
        print(f"No person detected in video {fn}. Skipping zoom preprocessing.")
        return

    # Convert positions to a DataFrame
    positions_df = positions_to_dataframe(positions)

    # Ensure we have a position entry for every frame so we never drop frames.
    # Use detections to initialize, then forward/backfill across the full frame range.
    total_frames = frame_index  # identify_largest_person increments frame_index for every frame
    full_frames = pd.DataFrame({"frame": np.arange(0, total_frames)})
    positions_df = full_frames.merge(positions_df, on="frame", how="left")
    # Forward/back-fill bounding box coordinates
    for col in ["x1", "y1", "x2", "y2"]:
        positions_df[col] = positions_df[col].ffill().bfill()
    # Recompute derived columns after filling
    positions_df["center_x"] = (positions_df["x1"] + positions_df["x2"]) // 2
    positions_df["center_y"] = (positions_df["y1"] + positions_df["y2"]) // 2
    positions_df["width"] = positions_df["x2"] - positions_df["x1"]
    positions_df["height"] = positions_df["y2"] - positions_df["y1"]

    # Determine width and height - use 60th percentile for width to zoom in a bit more
    # This still avoids over-zooming on occasional wide arm-swing frames.
    width_percentile = 0.60
    max_width = int(positions_df["width"].quantile(width_percentile))
    # Add extra buffer (10%) horizontally
    max_width = int(max_width * 1.1)

    # Use a fixed height as well to keep the zoom level constant (no scale "pulsing")
    height_percentile = 0.75
    max_height = int(positions_df["height"].quantile(height_percentile))
    max_height = int(max_height * 1.05)

    # Apply smoothing window settings (still used for center smoothing below)
    width_window = max(20, smoothing_window * 2)
    height_window = max(10, smoothing_window * 4)
    
    # Fixed width and height to remove zoom in/out jitter
    width_smooth = pd.Series([max_width] * len(positions_df))
    height_smooth = pd.Series([max_height] * len(positions_df))
    
    # Use stronger smoothing for vertical to reduce jitter
    # Use LESS smoothing for horizontal to keep runner centered and responsive
    center_y_smoothing_window = max(30, smoothing_window * 3)
    center_x_smoothing_window = max(15, smoothing_window)  # Much smaller window for responsive horizontal tracking
    
    # Smooth person's center position - this centers on the person's center (not head)
    positions_df["smooth_center_y"] = positions_df["center_y"].rolling(
        center_y_smoothing_window, center=True, min_periods=10
    ).mean()
    positions_df["smooth_center_x"] = positions_df["center_x"].rolling(
        center_x_smoothing_window, center=True, min_periods=5  # Smaller min_periods for faster response
    ).mean()

    # Fill NaN values with the first/last valid value for immediate tracking
    positions_df["smooth_center_y"] = positions_df["smooth_center_y"].bfill().ffill()
    positions_df["smooth_center_x"] = positions_df["smooth_center_x"].bfill().ffill()
    
    # STABILIZE RUNNER: Use smoothed position directly - no additional blending
    # The smoothing window is already small enough to track accurately while reducing jitter
    # Center the crop on the person's center position (tracks both horizontally and vertically)
    positions_df["smooth_y"] = positions_df["smooth_center_y"]
    
    # TRACK runner horizontally: crop follows runner's position to keep them centered
    positions_df["smooth_x"] = positions_df["smooth_center_x"]
    
    # Set width and height
    positions_df["smooth_w"] = width_smooth
    positions_df["smooth_h"] = height_smooth

    # Reopen the video for the second loop
    cap = cv2.VideoCapture(video_path)
    frame_index = 0

    # Second loop: Apply smoothing and segmentation, then write output
    all_frames = process_frames(cap, positions_df, output_size, frame_width, frame_height, all_frames=all_frames)

    cap.release()

    # Trim and write the final video
    if len(all_frames) > 20:  # Only trim if we have enough frames
        subtracted_frames = 1
        trimmed_frames = all_frames[subtracted_frames:-subtracted_frames]
        
        # Write trimmed frames to new video
        out = cv2.VideoWriter(output_path, fourcc, fps, output_size)
        for frame in trimmed_frames:
            out.write(frame)
        out.release()
    else:
        print("Video too short for trimming")
        # Write all frames if video is too short
        out = cv2.VideoWriter(output_path, fourcc, fps, output_size)
        for frame in all_frames:
            out.write(frame)
        out.release()

    cv2.destroyAllWindows()
    

def process_post_distance(fn,destination_folder,fps):
    model_path = join_paths('required','models','yolo11n.pt')
    print(f"Loading YOLO model from: {model_path}")
    
    if not path_exists(model_path):
        print(f"ERROR: YOLO model not found at {model_path}")
        return
    
    try:
        model = YOLO(model_path)
        print("YOLO model loaded successfully")
    except Exception as e:
        print(f"ERROR: Failed to load YOLO model: {e}")
        return

    buffer_ratio = 0.3  

    video_path, output_path = handle_paths(destination_folder, fn)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the original video's FPS and frame dimensions
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Use original dimensions (no rotation)
    aspect_ratio = frame_width / frame_height

    # Adjust output size to maintain aspect ratio
    if aspect_ratio > 1:
        output_size = (960, int(960 / aspect_ratio))  # Landscape
    else:
        output_size = (int(720 * aspect_ratio), 720)  # Portrait
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 15  # Output video frame rate
    out = cv2.VideoWriter(output_path, fourcc, fps, output_size)

    # Variables for tracking
    smoothing_window = 30  # Moving average window size for smoothing
    positions = []
    frame_index = 0

    # First loop: Identify largest person object - use original dimensions
    frame_index = identify_largest_person(cap, model, frame_width, frame_height, positions, buffer_ratio, frame_index, rotate=False)

    cap.release()

    # Check if any person was detected
    if not positions:
        print(f"No person detected in video {fn}. Skipping zoom preprocessing.")
        return

    # Convert positions to a DataFrame
    positions_df = positions_to_dataframe(positions)

    # Apply smoothing with robust window sizes
    width_window = max(10, smoothing_window * 2)  # Ensure minimum window size of 10
    height_window = max(10, smoothing_window // 2)  # Ensure minimum window size of 10
    
    width_smooth = positions_df["width"].rolling(width_window, center=True, min_periods=5).mean()
    height_smooth = positions_df["height"].rolling(height_window, center=True, min_periods=5).mean()
    positions_df = apply_smoothing(positions_df, smoothing_window, 8, 2, width_smooth, height_smooth)

    # Reopen the video for the second loop
    cap = cv2.VideoCapture(video_path)
    frame_index = 0

    # Second loop: Apply smoothing and segmentation, then write output - use original dimensions
    process_frames(cap, positions_df, output_size, frame_width, frame_height, out=out, rotate=False)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


# is this needed?
def process_other(fn,destination_folder,fps):
    model_path = join_paths('required','models','yolo11n.pt')
    print(f"Loading YOLO model from: {model_path}")
    
    if not path_exists(model_path):
        print(f"ERROR: YOLO model not found at {model_path}")
        return
    
    try:
        model = YOLO(model_path)
        print("YOLO model loaded successfully")
    except Exception as e:
        print(f"ERROR: Failed to load YOLO model: {e}")
        return

    buffer_ratio = 0.3
    output_size = (720, 960)  # width, height

    # make video and output paths
    video_path, output_path = handle_paths(destination_folder, fn)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the original video's FPS and frame dimensions
    # original_fps = int(cap.get(cv2.CAP_PROP_FPS)) # this was not used 
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 15  # Output video frame rate
    out = cv2.VideoWriter(output_path, fourcc, fps, output_size)

    # Variables for tracking
    
    # smoothing_window = original_fps  # Moving average window size for smoothing # this was not used 
    positions = []
    frame_index = 0

    # First loop: Identify largest person object
    frame_index = identify_largest_person(cap, model, frame_width, frame_height, positions, buffer_ratio, frame_index, rotate=False)
    cap.release()

    # Check if any person was detected
    if not positions:
        print(f"No person detected in video {fn}. Skipping zoom preprocessing.")
        return

    # Convert positions to a DataFrame
    positions_df = positions_to_dataframe(positions)

    # Determine the global bounding box that encompasses the person
    min_x1 = positions_df["x1"].min()
    min_y1 = positions_df["y1"].min()
    max_x2 = positions_df["x2"].max()
    max_y2 = positions_df["y2"].max()

    # Ensure the box fits within the frame dimensions
    crop_x1 = max(0, min_x1)
    crop_y1 = max(0, min_y1)
    crop_x2 = min(frame_width, max_x2)
    crop_y2 = min(frame_height, max_y2)

    # Reopen the video for the second loop
    cap = cv2.VideoCapture(video_path)
    frame_index = 0

    # Second loop: Apply smoothing and segmentation, then write output
    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        if frame_index in positions_df["frame"].values:
            row = positions_df[positions_df["frame"] == frame_index].iloc[0]

            avg_x = int(row["center_x"]) if not pd.isna(row["center_x"]) else (crop_x1 + crop_x2) // 2
            avg_y = int(row["center_y"]) if not pd.isna(row["center_y"]) else (crop_y1 + crop_y2) // 2
            avg_w = int(row["width"]) if not pd.isna(row["width"]) else (crop_x2 - crop_x1)
            avg_h = int(row["height"]) if not pd.isna(row["height"]) else (crop_y2 - crop_y1)

            # Calculate the maximum width after filling NaN values
            max_width = positions_df["width"].max()

            # Fill the entire 'smooth_w' column with the maximum width
            positions_df["width"] = max_width

            fr_ratio = output_size[1] / output_size[0]
            if avg_h / max(1, avg_w) <= fr_ratio:
                avg_h = int(avg_w * fr_ratio)
            else:
                avg_w = int(avg_h / fr_ratio)

            crop_x1 = max(0, avg_x - avg_w // 2)
            crop_y1 = max(0, avg_y - avg_h // 2)
            crop_x2 = min(frame_width, avg_x + avg_w // 2)
            crop_y2 = min(frame_height, avg_y + avg_h // 2)

            person_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]

            # Resize while maintaining aspect ratio and padding with black
            scale_w = output_size[0] / person_frame.shape[1]
            scale_h = output_size[1] / person_frame.shape[0]
            scale = min(scale_w, scale_h)
            new_width = int(person_frame.shape[1] * scale)
            new_height = int(person_frame.shape[0] * scale)
            resized_frame = cv2.resize(person_frame, (new_width, new_height))

            canvas = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
            x_offset = (output_size[0] - new_width) // 2
            y_offset = (output_size[1] - new_height) // 2
            canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_frame
            out.write(canvas)

        frame_index += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def handle_paths(destination_folder, fn):
    """
    Constructs input and output video paths.

    Builds paths for the input video and output in a 'zoom' subfolder, creating the subfolder if needed.

    Args:
        destination_folder (str): Base folder for input/output.
        fn (str): Input video filename or full path.

    Returns:
        tuple: (video_path, output_path) for input video and output in 'zoom' subfolder.

    Notes:
        - Uses `join_paths` for OS-compatible paths.
        - Creates 'zoom' subfolder with `make_dirs`.
        - Uses `basename` to extract filename.
    """
    # If fn is a full path, use it directly; otherwise construct the path
    if path_exists(fn):
        video_path = fn
        video_name = basename(fn)
    else:
        video_path = join_paths(destination_folder, fn)
        video_name = basename(fn)
    
    output_path = join_paths(destination_folder, 'zoom')
    make_dirs(output_path, exist_ok=True)
    output_path = join_paths(output_path, video_name)
    return video_path, output_path


def identify_largest_person(cap, model, frame_width, frame_height, positions, buffer_ratio, frame_index, rotate=False):
    total_frames = 0
    frames_with_person = 0
    
    print(f"Starting person detection with frame dimensions: {frame_width}x{frame_height}")
    
    while cap.isOpened():
        success, frame = cap.read()

        if rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if not success:
            break

        total_frames += 1

        # Run YOLO model
        results = model.track(frame, persist=True)

        # Extract person detections
        largest_box = None
        largest_area = 0

        try:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            cls = results[0].boxes.cls.int().cpu().tolist()
            
            if total_frames == 1:  # Debug first frame
                print(f"First frame - Total detections: {len(boxes)}")
                print(f"Classes detected: {cls}")
                
        except Exception as e:
            if total_frames == 1:
                print(f"Error in YOLO detection on first frame: {e}")
            frame_index += 1
            continue

        for box, track_id, cl in zip(boxes, track_ids, cls):
            if cl >= 2:  # Detect only people
                continue

            x, y, w, h = box
            area = w * h
            if area > largest_area:
                largest_area = area
                largest_box = (x, y, w, h)

        if largest_box is not None:
            frames_with_person += 1
            x, y, w, h = largest_box
            x1 = max(0, x - w * (1 + buffer_ratio) / 2)
            y1 = max(0, y - h * (1 + buffer_ratio) / 2)
            x2 = min(frame_width, x + w * (1 + buffer_ratio) / 2)
            y2 = min(frame_height, y + h * (1 + buffer_ratio) / 2)

            positions.append({
                "frame": frame_index,
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2)
            })

        frame_index += 1
    
    print(f"Person detection summary: {frames_with_person}/{total_frames} frames had person detections")
    return frame_index


def positions_to_dataframe(positions):
    positions_df = pd.DataFrame(positions)
    positions_df["center_x"] = (positions_df["x1"] + positions_df["x2"]) // 2
    positions_df["center_y"] = (positions_df["y1"] + positions_df["y2"]) // 2
    positions_df["width"] = positions_df["x2"] - positions_df["x1"]
    positions_df["height"] = positions_df["y2"] - positions_df["y1"]
    return positions_df

def apply_smoothing(positions_df, smoothing_window, smooth_x, smooth_y, smooth_w, smooth_h):
    # Ensure window sizes are at least 25 to accommodate min_periods
    x_window = max(25, smoothing_window * smooth_x)
    
    # For vertical smoothing, use a much smaller window when smooth_y is 1 to allow natural bouncing
    if smooth_y == 1:
        y_window = max(5, smoothing_window // 8)  # Very small window for minimal vertical smoothing
    else:
        y_window = max(25, smoothing_window // smooth_y)
    
    positions_df["smooth_x"] = positions_df["center_x"].rolling(x_window, center=True, min_periods=10).mean()
    positions_df["smooth_y"] = positions_df["center_y"].rolling(y_window, center=True, min_periods=5).mean()
    positions_df["smooth_w"] = smooth_w 
    positions_df["smooth_h"] = smooth_h  
    return positions_df

def process_frames(cap, positions_df, output_size, frame_width, frame_height, all_frames=None, out=None, rotate=False):
    """
    Processes video frames, crops to person, resizes, and either stores or writes frames.

    Args:
        cap (cv2.VideoCapture): Video capture object.
        positions_df (pd.DataFrame): DataFrame with smoothed position data.
        output_size (tuple): Output frame size (width, height).
        frame_width (int): Original frame width.
        frame_height (int): Original frame height.
        all_frames (list, optional): List to store frames if not writing directly.
        out (cv2.VideoWriter, optional): Video writer to write frames directly.
        rotate (bool): Whether to rotate frames 90 degrees counterclockwise.

    Returns:
        list: Updated all_frames list if provided, else None.
    """
    frame_index = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if frame_index in positions_df["frame"].values:
            row = positions_df[positions_df["frame"] == frame_index].iloc[0]
            
            # Handle NaN with mean values
            means = positions_df[["smooth_x", "smooth_y", "smooth_w", "smooth_h"]].mean()
            avg_x = int(row["smooth_x"]) if not pd.isna(row["smooth_x"]) else int(means["smooth_x"])
            avg_y = int(row["smooth_y"]) if not pd.isna(row["smooth_y"]) else int(means["smooth_y"])
            avg_w = int(row["smooth_w"]) if not pd.isna(row["smooth_w"]) else int(means["smooth_w"])
            avg_h = int(row["smooth_h"]) if not pd.isna(row["smooth_h"]) else int(means["smooth_h"])

            # Adjust aspect ratio
            fr_ratio = output_size[1] / output_size[0]
            if avg_h / max(1, avg_w) <= fr_ratio:
                avg_h = int(avg_w * fr_ratio)
            else:
                avg_w = int(avg_h / fr_ratio)

            # Crop frame: ALWAYS center on runner's position to keep them in frame
            # Horizontal: center crop on runner's current position (avg_x tracks runner as they move)
            crop_x1_desired = avg_x - avg_w // 2
            crop_x2_desired = avg_x + avg_w // 2
            
            # Adjust crop if it would go out of bounds (keep runner visible)
            if crop_x1_desired < 0:
                # Crop would go past left edge - shift right but keep runner in frame
                crop_x1 = 0
                crop_x2 = min(frame_width, avg_w)
            elif crop_x2_desired > frame_width:
                # Crop would go past right edge - shift left but keep runner in frame
                crop_x2 = frame_width
                crop_x1 = max(0, frame_width - avg_w)
            else:
                # Crop fits perfectly - centered on runner
                crop_x1 = crop_x1_desired
                crop_x2 = crop_x2_desired
            
            # Vertical: center crop on runner's vertical position
            crop_y1 = max(0, avg_y - int(avg_h * 0.5))
            crop_y2 = min(frame_height, avg_y + int(avg_h * 0.5))
            person_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]

            # Resize and pad
            scale = min(output_size[0] / person_frame.shape[1], output_size[1] / person_frame.shape[0])
            new_width, new_height = int(person_frame.shape[1] * scale), int(person_frame.shape[0] * scale)
            resized_frame = cv2.resize(person_frame, (new_width, new_height))
            canvas = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
            
            # STABILIZE RUNNER: Always center runner horizontally in output frame
            # Calculate where runner's center is within the crop (in original frame coordinates)
            # This is the runner's position relative to the left edge of the crop
            runner_center_in_crop = avg_x - crop_x1
            # After resizing, runner's center position in resized frame (in pixels from left)
            runner_center_in_resized = runner_center_in_crop * scale
            # Center of output frame - this is where runner MUST be positioned
            output_center_x = output_size[0] // 2
            # Position resized frame so runner's center aligns with output center
            # x_offset positions the left edge of resized frame such that runner is at output_center_x
            x_offset = int(output_center_x - runner_center_in_resized)
            
            # Ensure the entire resized frame stays within canvas bounds
            # If we can't fit it perfectly centered, keep it as close as possible
            if x_offset < 0:
                x_offset = 0
            elif x_offset + new_width > output_size[0]:
                x_offset = output_size[0] - new_width
            
            # Adjust y_offset to center the person's center in the output frame
            # With a symmetric 50/50 crop, the person's center is at 50% of the crop height
            # After resizing, person's center is at 50% * new_height from top of resized frame
            # To center person in output: y_offset should position person center at output_center_y
            output_center_y = output_size[1] // 2
            person_center_in_resized = int(new_height * 0.5)  # Person center is 50% from top of crop
            y_offset = output_center_y - person_center_in_resized
            
            # Ensure y_offset keeps the entire resized frame within canvas bounds
            y_offset = max(0, min(y_offset, output_size[1] - new_height))
            
            canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_frame

            # Store or write frame
            if all_frames is not None:
                all_frames.append(canvas)
            elif out is not None:
                out.write(canvas)

        frame_index += 1

    return all_frames