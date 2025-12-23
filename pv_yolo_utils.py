import numpy as np
from ultralytics import YOLO
import cv2
from tkinter import filedialog
import tkinter as tk

def draw_outlined_text(
    frame,
    text,
    org,
    font,
    font_scale,
    color,
    thickness,
    line_type=cv2.LINE_AA,
    outline_color=(0, 0, 0),
    outline_thickness=None,
):
    """
    Draw text with a black outline so it stands out on light backgrounds.
    """
    if outline_thickness is None:
        outline_thickness = thickness + 2

    # Outline
    cv2.putText(
        frame,
        text,
        org,
        font,
        font_scale,
        outline_color,
        outline_thickness,
        line_type,
    )
    # Fill
    cv2.putText(
        frame,
        text,
        org,
        font,
        font_scale,
        color,
        thickness,
        line_type,
    )


class PersonDetector:
    """
    Lightweight wrapper around a YOLO model to detect the largest 'person'
    in a frame and return a bounding box.

    Usage:
        detector = PersonDetector(model_path="yolo11n.pt")
        bbox = detector.detect_largest_person(frame)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
    """

    def __init__(self, model_path="yolo11n.pt", conf=0.25, device=None, imgsz=1280):
        """
        model_path:
            Path or name of a YOLO model. "yolo11n.pt" will be auto-downloaded
            by ultralytics on first use if not found locally.
        conf:
            Confidence threshold for detections.
        device:
            Optional device string ("cpu", "cuda", etc.). None = let YOLO decide.
        imgsz:
            Base inference size. Will be rounded up to a multiple of 32 internally
            to avoid Ultralytics 'imgsz must be multiple of max stride' warnings.
        """
        self.model = YOLO(model_path)
        self.conf = conf
        self.device = device
        self.imgsz = imgsz

    def detect_largest_person(self, frame):
        """
        Run YOLO on a single frame and return (x1, y1, x2, y2) for the largest
        detected person, or None if no person is found.

        Behavior:
            - If no person is found on the raw frame, it will create a slightly
              brighter / more saturated copy of the frame and run YOLO once more.
            - If still no person is found, returns None.
        """
        if frame is None:
            return None

        h, w = frame.shape[:2]

        # Choose an inference size and round it up to a multiple of 32
        base = self.imgsz or max(h, w)
        imgsz = int(((base + 31) // 32) * 32)

        # --- first YOLO pass on original frame ---
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            imgsz=imgsz,
            device=self.device,
            verbose=False,   # keeps Ultralytics progress quiet
        )

        # Helper to extract the largest person box from a YOLO result list
        def _largest_person_box(results_obj):
            if len(results_obj) == 0 or len(results_obj[0].boxes) == 0:
                return None

            res0 = results_obj[0]
            boxes_xyxy = res0.boxes.xyxy.cpu().numpy()
            classes = res0.boxes.cls.cpu().numpy().astype(int)

            # class 0 is 'person' in COCO
            person_indices = np.where(classes == 0)[0]
            if person_indices.size == 0:
                return None

            best_idx = None
            best_area = 0.0
            for idx in person_indices:
                x1b, y1b, x2b, y2b = boxes_xyxy[idx]
                area = max(0.0, (x2b - x1b)) * max(0.0, (y2b - y1b))
                if area > best_area:
                    best_area = area
                    best_idx = idx

            if best_idx is None:
                return None

            x1b, y1b, x2b, y2b = boxes_xyxy[best_idx]

            # Clamp to image bounds
            x1b = int(max(0, min(w - 1, x1b)))
            y1b = int(max(0, min(h - 1, y1b)))
            x2b = int(max(0, min(w, x2b)))
            y2b = int(max(0, min(h, y2b)))

            if x2b <= x1b or y2b <= y1b:
                return None

            return (x1b, y1b, x2b, y2b)

        bbox = _largest_person_box(results)

        # --- second-chance YOLO with boosted saturation/brightness ---
        if bbox is None:
            # Convert BGR -> HSV and boost S and V a bit
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
            h_ch, s_ch, v_ch = cv2.split(hsv)

            s_ch *= 1.2  # boost saturation
            v_ch *= 1.2  # boost brightness
            s_ch = np.clip(s_ch, 0, 255)
            v_ch = np.clip(v_ch, 0, 255)

            hsv_boosted = cv2.merge([h_ch, s_ch, v_ch]).astype(np.uint8)
            frame_boosted = cv2.cvtColor(hsv_boosted, cv2.COLOR_HSV2BGR)

            results_boost = self.model.predict(
                source=frame_boosted,
                conf=self.conf,
                imgsz=imgsz,
                device=self.device,
                verbose=False,
            )

            bbox = _largest_person_box(results_boost)

        if bbox is None:
            return None

        return bbox

def pick_video_file():
    """
    Open a file picker dialog to select a video.
    """
    root = tk.Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(
        title="Select video file",
        filetypes=[
            ("Video files", "*.mp4;*.avi;*.mov;*.mkv;*.m4v"),
            ("All files", "*.*"),
        ],
    )
    root.update()
    root.destroy()
    return video_path