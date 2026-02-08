import logging
import copy
import cv2
import mediapipe as mp
from typing import TYPE_CHECKING

from pv_yolo_utils import PersonDetector

mp_pose = mp.solutions.pose
logger = logging.getLogger("pv.pose")

if TYPE_CHECKING:
    from mediapipe.python.solutions.pose import Pose as MpPose


def landmarks_full_from_roi(pose_landmarks, roi_box, frame_w, frame_h, roi_w, roi_h):
    """Convert ROI-normalized landmarks to full-frame normalized landmarks.

    NOTE: We avoid importing mediapipe.framework.formats.landmark_pb2 because some
    MediaPipe wheels don't ship that module. Instead, we deep-copy the returned
    pose_landmarks object and edit x/y in place.
    """
    roi_x1, roi_y1, _, _ = roi_box

    lms_full = copy.deepcopy(pose_landmarks)
    for lm in lms_full.landmark:
        x_full = roi_x1 + (lm.x * roi_w)
        y_full = roi_y1 + (lm.y * roi_h)
        lm.x = float(x_full / frame_w)
        lm.y = float(y_full / frame_h)
        # keep z/visibility/presence as-is

    return lms_full


def detect_person_roi(frame, detector: PersonDetector, margin: float = 0.30):
    """Detect largest person and return expanded ROI box (x1,y1,x2,y2) or None."""
    bbox = detector.detect_largest_person(frame)
    if bbox is None:
        logger.debug("YOLO returned no bbox")
        return None

    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]

    bw = x2 - x1
    bh = y2 - y1
    cx = x1 + bw / 2
    cy = y1 + bh / 2

    roi_w = int(bw * (1 + margin))
    roi_h = int(bh * (1 + margin))

    roi_x1 = max(0, int(cx - roi_w / 2))
    roi_y1 = max(0, int(cy - roi_h / 2))
    roi_x2 = min(w, int(cx + roi_w / 2))
    roi_y2 = min(h, int(cy + roi_h / 2))

    if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
        return None

    roi_box = (roi_x1, roi_y1, roi_x2, roi_y2)
    logger.debug(f"ROI box: {roi_box} (margin={margin})")
    return roi_box


def run_pose_on_frame(frame_bgr, pose: "MpPose", roi_box):
    """Run MediaPipe Pose on ROI and return full-frame landmarks or None."""
    if roi_box is None:
        return None

    frame_h, frame_w = frame_bgr.shape[:2]
    roi_x1, roi_y1, roi_x2, roi_y2 = roi_box
    roi = frame_bgr[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_h, roi_w = roi.shape[:2]
    if roi_h <= 0 or roi_w <= 0:
        return None

    image_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        logger.debug("No pose landmarks found in ROI")
        return None

    lms = landmarks_full_from_roi(
        results.pose_landmarks,  # <-- pass the whole object now
        roi_box,
        frame_w,
        frame_h,
        roi_w,
        roi_h,
    )
    logger.debug("Pose landmarks converted to full-frame")
    return lms
