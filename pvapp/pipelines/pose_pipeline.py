import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose

from mediapipe.framework.formats import landmark_pb2
from pvapp.core.detector import PersonDetector


class PoseResultWrapper:
    def __init__(self, mp_result, roi_box):
        self.pose_landmarks = mp_result.pose_landmarks
        self.roi_box = roi_box


class ROIBoxSmoother:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.history = []

    def smooth(self, bbox):
        if bbox is None:
            return None
        self.history.append(bbox)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        arr = np.array(self.history)
        return tuple(arr.mean(axis=0).astype(int))


def extract_pose_data(video_path: str, yolo_model_path="yolo11n.pt", conf=0.25, start_frame=0, end_frame=None, progress_callback=None, device=None):
    """
    Run MediaPipe Pose on the video, guided by YOLO person detection.
    
    CRITICAL: We MUST use YOLO to find the person first. 
    Running MediaPipe on the full frame leads to poor accuracy and tracking failures 
    when the background is complex or the athlete is small.
    The pipeline is: YOLO -> ROI Crop -> MediaPipe -> Remap to Full Frame.

    Args:
        video_path (str): Path to the input video.
        yolo_model_path (str): Path to YOLO detection model.
        start_frame (int): Frame index to start processing.
        end_frame (int): Frame index to stop processing (inclusive). If None, process to end.
        progress_callback (callable, optional): function(pct, status_text)

    Returns:
        list: A list where each element is the `results` object (or equivalent) 
              containing .pose_landmarks in full-frame coordinates.
              Indices correspond to video frames. Frames outside [start, end] are None.
              Also includes .roi_box for visualization.
    """
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame is None:
        end_frame = total_frames - 1
        
    start_frame = max(0, start_frame) if start_frame is not None else 0
    end_frame = min(end_frame, total_frames - 1)
    
    # Initialize YOLO Detector
    detector = PersonDetector(model_path=yolo_model_path, conf=conf, device=device)
    
    # Pre-fill results with None for frames before start_frame
    pose_results = [None] * start_frame
    
    # Seek to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Import the new stabilizer
    from pvapp.core.pose_stabilization import PoseStabilizer

    # Use the Stabilizer instead of the old Smoother
    # Note: Using alpha=1.0 to DISABLE EMA smoothing (fixing lag), but keeping velocity clamping.
    pose_stabilizer = PoseStabilizer(smoothing_alpha=1.0, conf_thresh=0.3)
    roi_smoother = ROIBoxSmoother(window_size=2) # Reduced from 5 to 2 to minimize camera lag

    # Pose context
    with mp_pose.Pose(
        min_detection_confidence=0.5, # Unchanged
        min_tracking_confidence=0.5,  # Unchanged
        model_complexity=2,           # Unchanged
        static_image_mode=False,
        smooth_landmarks=True         # Allow MediaPipe internal smoothing
    ) as pose:
        
        frame_idx = start_frame
        process_count = 0
        total_to_process = end_frame - start_frame + 1

        while frame_idx <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_h, frame_w = frame.shape[:2]
            
            # 1. Detect Person (YOLO)
            bbox = detector.detect_largest_person(frame)
            
            # 2. Smooth ROI
            # Even if YOLO misses a frame (None), we might want to use the last known ROI?
            # For now, strict behavior: if no detection, no ROI.
            # But the smoother handles "None" by returning none? No, we should reset or skip.
            # If bbox is None, we skip.
            
            roi_box = None
            landmark_list_full = None
            
            if bbox is not None:
                # Apply ROI Smoothing
                bbox = roi_smoother.smooth(bbox)
                
                x1, y1, x2, y2 = bbox
                
                # Expand box
                margin = 0.3
                bw = x2 - x1
                bh = y2 - y1
                cx = x1 + bw / 2
                cy = y1 + bh / 2
                
                roi_w = int(bw * (1 + margin))
                roi_h = int(bh * (1 + margin))
                
                roi_x1 = max(0, int(cx - roi_w / 2))
                roi_y1 = max(0, int(cy - roi_h / 2))
                roi_x2 = min(frame_w, int(cx + roi_w / 2))
                roi_y2 = min(frame_h, int(cy + roi_h / 2))
                
                if roi_x2 > roi_x1 and roi_y2 > roi_y1:
                    roi_box = (roi_x1, roi_y1, roi_x2, roi_y2)
                    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
                    
                    # 3. Run Pose on ROI
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    results = pose.process(roi_rgb)
                    
                    # 4. Remap landmarks to full frame
                    if results.pose_landmarks:
                        lm = results.pose_landmarks.landmark
                        roi_h_actual, roi_w_actual = roi.shape[:2]
                        
                        landmarks_full = []
                        for lm_i in lm:
                            x_roi = lm_i.x * roi_w_actual
                            y_roi = lm_i.y * roi_h_actual
                            x_full = roi_x1 + x_roi
                            y_full = roi_y1 + y_roi
                            
                            landmarks_full.append(
                                landmark_pb2.NormalizedLandmark(
                                    x=x_full / frame_w,
                                    y=y_full / frame_h,
                                    z=lm_i.z,
                                    visibility=lm_i.visibility,
                                    presence=lm_i.presence,
                                )
                            )
                            
                        raw_list_full = landmark_pb2.NormalizedLandmarkList(
                            landmark=landmarks_full
                        )
                        
                        # 5. Apply Stabilization (Side Constraint + Clamping + EMA)
                        landmark_list_full = pose_stabilizer.process(raw_list_full)
                    else:
                        # MediaPipe failed on this ROI
                        # We should potentially reset stabilizer or just pass None
                         pass

            # Wrap result
            class MappedResult:
                pass
            res = MappedResult()
            res.pose_landmarks = landmark_list_full
            
            pose_results.append(PoseResultWrapper(res, roi_box))
            
            frame_idx += 1
            process_count += 1
            if progress_callback and process_count % 10 == 0:
                pct = process_count / max(total_to_process, 1)
                progress_callback(pct, f"Extracting Pose (YOLO+Smoothed): {int(pct*100)}%")

    cap.release()
    
    # Fill remaining frames with None if video ended early
    while len(pose_results) < total_frames:
        pose_results.append(None)
    
    if progress_callback:
        progress_callback(1.0, "Pose Extraction Complete")
        
    return pose_results
