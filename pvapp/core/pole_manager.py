import numpy as np
from ultralytics import YOLO
import cv2

class PoleDetector:
    """
    Wrapper around a YOLO segmentation model to detect the pole.
    Returns masks/polygons.
    """

    def __init__(self, model_path="pole_detector_v3.pt", conf=0.25, device=None, imgsz=640):
        """
        model_path: Path to the custom pole detection model.
        conf: Confidence threshold.
        """
        self.model = YOLO(model_path)
        self.conf = conf
        self.device = device
        self.imgsz = imgsz

    def detect(self, frame):
        """
        Run YOLO segmentation on a single frame.
        
        Returns:
            list of masks (as binary numpy arrays) or None if no detection.
            For now, we might just return the raw Results object or list of masks.
        """
        if frame is None:
            return None

        # Run inference
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
            retina_masks=True # High quality masks
        )

        if not results:
            return None

        result = results[0]
        
        if result.masks is None:
            return None
            
        return result
