import cv2

def draw_simple_skeleton(
    frame,
    landmark_list_full,
    connections,
    point_color=(252, 0, 219),
    line_color=(0, 255, 0),
    point_radius=3,
    line_thickness=2,
    visibility_threshold=0.1,
):
    """Draw a simple skeleton in full-frame coords (MediaPipe NormalizedLandmarkList)."""
    frame_h, frame_w = frame.shape[:2]
    lm = landmark_list_full.landmark

    # Lines
    for start_idx, end_idx in connections:
        if start_idx >= len(lm) or end_idx >= len(lm):
            continue

        lms = lm[start_idx]
        lme = lm[end_idx]

        if (hasattr(lms, "visibility") and lms.visibility < visibility_threshold) or \
           (hasattr(lme, "visibility") and lme.visibility < visibility_threshold):
            continue

        x1 = int(lms.x * frame_w)
        y1 = int(lms.y * frame_h)
        x2 = int(lme.x * frame_w)
        y2 = int(lme.y * frame_h)

        if 0 <= x1 < frame_w and 0 <= y1 < frame_h and 0 <= x2 < frame_w and 0 <= y2 < frame_h:
            cv2.line(frame, (x1, y1), (x2, y2), line_color, line_thickness)

    # Points
    for lmp in lm:
        px = int(lmp.x * frame_w)
        py = int(lmp.y * frame_h)
        if 0 <= px < frame_w and 0 <= py < frame_h:
            cv2.circle(frame, (px, py), point_radius, point_color, -1)
