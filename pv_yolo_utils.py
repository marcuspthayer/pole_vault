import numpy as np
from ultralytics import YOLO


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
        """
        self.model = YOLO(model_path)
        self.conf = conf
        self.device = device
        self.imgsz = imgsz

    def detect_largest_person(self, frame):
        """
        Run YOLO on the given BGR frame and return (x1, y1, x2, y2) for
        the largest 'person' detection, or None if no person is found.
        Coordinates are in absolute pixel units (image coordinate system).
        """
        if frame is None:
            return None

        h, w = frame.shape[:2]

        # Choose an inference size and round it up to a multiple of 32
        base = self.imgsz or max(h, w)
        imgsz = int(((base + 31) // 32) * 32)

        results = self.model.predict(
            source=frame,
            conf=self.conf,
            imgsz=imgsz,
            device=self.device,
            verbose=False,   # keeps Ultralytics progress quiet
        )


        if len(results) == 0 or len(results[0].boxes) == 0:
            return None

        res = results[0]
        boxes_xyxy = res.boxes.xyxy.cpu().numpy()  # (N, 4): x1, y1, x2, y2
        classes = res.boxes.cls.cpu().numpy().astype(int)  # class indices

        # COCO 'person' class is typically 0
        person_indices = np.where(classes == 0)[0]
        if person_indices.size == 0:
            return None

        # Find largest person box by area
        best_idx = None
        best_area = 0.0
        for idx in person_indices:
            x1, y1, x2, y2 = boxes_xyxy[idx]
            area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
            if area > best_area:
                best_area = area
                best_idx = idx

        if best_idx is None:
            return None

        x1, y1, x2, y2 = boxes_xyxy[best_idx]
        # Clamp coordinates to frame bounds
        x1 = int(max(0, min(w - 1, x1)))
        y1 = int(max(0, min(h - 1, y1)))
        x2 = int(max(0, min(w, x2)))
        y2 = int(max(0, min(h, y2)))

        if x2 <= x1 or y2 <= y1:
            return None

        return (x1, y1, x2, y2)
