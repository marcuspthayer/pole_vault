import cv2

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

def draw_simple_skeleton(frame, landmark_list_full, connections, 
                         point_color=(0, 0, 255), line_color=(0, 255, 0),
                         point_radius=3, line_thickness=2,
                         visibility_threshold=0.1):
    """
    Draw a simple skeleton using OpenCV directly, based on a NormalizedLandmarkList
    and a list of (start_idx, end_idx) connections (e.g. mp_pose.POSE_CONNECTIONS).
    """
    frame_h, frame_w = frame.shape[:2]
    lm = landmark_list_full.landmark

    # Draw lines between connected joints
    for start_idx, end_idx in connections:
        if start_idx >= len(lm) or end_idx >= len(lm):
            continue
        lms = lm[start_idx]
        lme = lm[end_idx]

        # Optionally skip low-visibility landmarks
        if (hasattr(lms, "visibility") and lms.visibility < visibility_threshold) or \
           (hasattr(lme, "visibility") and lme.visibility < visibility_threshold):
            continue

        x1 = int(lms.x * frame_w)
        y1 = int(lms.y * frame_h)
        x2 = int(lme.x * frame_w)
        y2 = int(lme.y * frame_h)

        if 0 <= x1 < frame_w and 0 <= y1 < frame_h and 0 <= x2 < frame_w and 0 <= y2 < frame_h:
            cv2.line(frame, (x1, y1), (x2, y2), line_color, line_thickness)

    # Draw points at each joint
    for lmp in lm:
        px = int(lmp.x * frame_w)
        py = int(lmp.y * frame_h)
        if 0 <= px < frame_w and 0 <= py < frame_h:
            cv2.circle(frame, (px, py), point_radius, point_color, -1)
