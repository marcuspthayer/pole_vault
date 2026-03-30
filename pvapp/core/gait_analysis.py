import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

def estimate_stride_length(height_meters):
    """
    Estimate stride length based on athlete height.
    Logic ported from GaitKeeper metrics.ts:
    Leg Length = Height * 0.45
    Stride Length = Leg Length * 1.15 (Walking/Jogging heuristic)
    """
    if height_meters <= 0:
        return 0.0
    leg_length = height_meters * 0.45
    return leg_length * 1.15

def detect_foot_strikes(pose_landmarks_list, frame_indices, fps=30, min_lift=0.015, min_peak_dist_frames=None):
    """
    Detect foot strikes based on ankle vertical movement.
    
    Args:
        pose_landmarks_list: List of MediaPipe pose landmarks (or None).
        frame_indices: List of original frame indices corresponding to the data.
        fps: Frames per second of the video.
        min_lift: Minimum vertical rise (normalized 0-1) to consider a step. Default 0.015.
        min_peak_dist_frames: Minimum frames between steps. If None, calculated from FPS.
        
    Returns:
        strikes: List of dicts {'frame': int, 'side': 'left'|'right', 'confidence': float, 'pt': (x, y)}
    """
    if not pose_landmarks_list or len(pose_landmarks_list) < 5:
        return [], [], []

    # Extract ankle Y positions
    left_ankles_y = []
    right_ankles_y = []
    frames_with_data = []
    
    # Store full coords for point output
    left_ankles_coords = []
    right_ankles_coords = []

    for i, lm in enumerate(pose_landmarks_list):
        if lm is None:
            left_ankles_y.append(np.nan)
            right_ankles_y.append(np.nan)
            left_ankles_coords.append(None)
            right_ankles_coords.append(None)
            frames_with_data.append(frame_indices[i])
            continue

        l_ankle = lm.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        r_ankle = lm.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
        
    # Anatomical filtering: Foot must be below knee (Y is larger)
        l_knee = lm.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
        r_knee = lm.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
        
        # Check visibility and anatomical validity
        # Y increases downwards. Ankle > Knee means Ankle is lower than Knee.
        if l_ankle.visibility > 0.3 and l_ankle.y > l_knee.y:
            left_ankles_y.append(l_ankle.y)
            left_ankles_coords.append((l_ankle.x, l_ankle.y))
        else:
            left_ankles_y.append(np.nan)
            left_ankles_coords.append(None)
            
        if r_ankle.visibility > 0.3:
            right_ankles_y.append(r_ankle.y)
            right_ankles_coords.append((r_ankle.x, r_ankle.y))
        else:
            right_ankles_y.append(np.nan)
            right_ankles_coords.append(None)
            
        frames_with_data.append(frame_indices[i])

    # Helper: Smooth data
    def smooth_signal(signal, window=5):
        # Handle simple smoothing ignoring NaNs if possible or just filling them
        # Simple convolution
        s = np.array(signal)
        # Fill NaNs with nearest or interp? For step detection, linear interp is fine
        nans, x = np.isnan(s), lambda z: z.nonzero()[0]
        if np.all(nans):
             return s # All nans
        
        s[nans] = np.interp(x(nans), x(~nans), s[~nans])
        
        kernel = np.ones(window) / window
        return np.convolve(s, kernel, mode='same')

    smooth_l = smooth_signal(left_ankles_y)
    smooth_r = smooth_signal(right_ankles_y)
    
    strikes = []
    
    # Parameters from reference or override
    if min_peak_dist_frames is not None:
        min_peak_dist = min_peak_dist_frames
    else:
        # minPeakDistance: 10 frames at 30fps
        min_peak_dist = max(5, int((fps / 30.0) * 10))
        
    # Threshold for "lift": Use passed arg
      
    
    def find_strikes(signal, coords, side):
        found_strikes = []
        last_strike_idx = -min_peak_dist
        
        # Look for local MAXIMA (highest Y = lowest point in image = foot on ground)
        # WAIT! In image coords (0,0 is top-left), Y increases DOWNWARDS.
        # So "Highest Y" means "Lowest point on screen" (Foot on ground).
        # Reference `countSteps`:
        # "isLocalMax = positions[i] > positions[i-1] && positions[i] > positions[i+1]"
        # So they treat larger Y value (lower on screen) as the peak/strike. Correct.
        
        for i in range(1, len(signal) - 1):
            val = signal[i]
            if np.isnan(val): continue
            
            # Local Max check
            if val > signal[i-1] and val > signal[i+1]:
                # Distance check
                if i - last_strike_idx >= min_peak_dist:
                    # Amplitude check (lift)
                    # We want to see if it went *UP* (smaller Y) before coming down.
                    # Look back min_peak_dist frames for a minimum
                    start_look = max(0, i - min_peak_dist)
                    window = signal[start_look:i]
                    if len(window) > 0:
                        min_y = np.min(window) # Highest physical point (smallest Y)
                        lift = val - min_y
                        
                        if lift > min_lift:
                            # Register strike
                            last_strike_idx = i
                            # Get coordinate (use original coord if available, else smooth Y)
                            orig_pt = coords[i]
                            pt = orig_pt if orig_pt else (0.5, val) # fallback X
                            
                            found_strikes.append({
                                'frame': frames_with_data[i],
                                'side': side,
                                'confidence': 1.0, # Placeholder
                                'pt': pt
                            })
        return found_strikes

    strikes.extend(find_strikes(smooth_l, left_ankles_coords, 'left'))
    strikes.extend(find_strikes(smooth_r, right_ankles_coords, 'right'))
    
    # Sort by frame
    strikes.sort(key=lambda x: x['frame'])
    
    print(f"[DEBUG-STRIDE] Function returning {len(strikes)} strikes, {len(left_ankles_y)} L_y, {len(right_ankles_y)} R_y")
    return strikes, left_ankles_y, right_ankles_y

def calculate_cadence(strikes, total_duration_minutes):
    """
    Calculate Steps Per Minute (Cadence).
    """
    if total_duration_minutes <= 0:
        return 0
    
    step_count = len(strikes)
    return step_count / total_duration_minutes

def calculate_pixel_stride_and_convert(strikes, scale_factor=None):
    """
    Calculate stride length from foot strikes.
    
    Args:
        strikes: List of 'strike' dicts (sorted by frame).
        scale_factor: Meters per pixel (float) or None.
        
    Returns:
        stride_data: List of dicts:
            {
                'frame': int (frame of end-of-stride),
                'stride_px': float,
                'stride_m': float or None,
                'rate_spm': float (instantaneous rate)
            }
    """
    stride_data = []
    if len(strikes) < 2:
        return stride_data
        
    for i in range(1, len(strikes)):
        curr = strikes[i]
        prev = strikes[i-1]
        
        # Check if sides alternate (L->R or R->L)
        # If same side (L->L), we missed a step or hopping? 
        # For now, simplistic: just take X distance.
        # But real stride is between alternating feet.
        
        # Distance |x_curr - x_prev|
        dist_px = abs(curr['pt'][0] - prev['pt'][0]) # Assuming pt is normalized 0-1?
        # WAIT: 'pt' in detect_foot_strikes seems to be (l_ankle.x, l_ankle.y) which are NORMALIZED (0-1).
        # We need to know if the caller wants pixels or normalized.
        # The function signature says "pixel stride". 
        # Let's return normalized difference first, caller converts to px?
        # Or, we assume 'pt' is normalized, so dist_px is actually dist_norm.
        # Let's clarify: 'detect_foot_strikes' uses 'left_ankles_coords.append((l_ankle.x, l_ankle.y))'.
        # MediaPipe uses normalized [0,1].
        
        # To get real pixels, we need image width.
        # But this function doesn't have image width.
        # Let's return "Normalized Stride" unless we pass width.
        # Actually, 'scale_factor' usually implies "Meters per Pixel".
        # If we only have normalized, we can't apply scale_factor yet without Width.
        # BUT: The 'scale_factor' calculated in pipeline will likely be "Meters / PixelWidth".
        # Let's assume input 'strikes' has normalized coordinates.
        
        dist_norm = abs(curr['pt'][0] - prev['pt'][0])
        
        # Rate: Steps per minute = 60 / (time_diff_sec)
        # We need FPS for this. Pass FPS? Or just return frame diff.
        d_frame = curr['frame'] - prev['frame']
        
        entry = {
            'frame': curr['frame'],
            'stride_norm': dist_norm, # 0.0-1.0
            'frames_dur': d_frame,
            'side': curr['side']
        }
        stride_data.append(entry)
        
    return stride_data

def interpolate_missed_steps(strikes, max_gap_ratio=2.5, min_gap_ratio=1.5):
    """
    Interpolate missing steps based on spatial/temporal gaps.
    
    Logic:
    1. Calculate median step distance (normalized X) and median frame delta.
    2. Iterate through steps. If a step gap is ~2x the median, insert a step at midpoint.
    3. Mark inserted step as 'interpolated': True.
    """
    if len(strikes) < 3:
        return strikes # Not enough to establish median
    
    # 1. Calculate medians
    x_diffs = []
    f_diffs = []
    for i in range(1, len(strikes)):
        dx = abs(strikes[i]['pt'][0] - strikes[i-1]['pt'][0])
        df = strikes[i]['frame'] - strikes[i-1]['frame']
        x_diffs.append(dx)
        f_diffs.append(df)
        
    med_dx = np.median(x_diffs)
    med_df = np.median(f_diffs)
    
    if med_dx < 0.001: return strikes # Standing still?
    
    new_strikes = []
    new_strikes.append(strikes[0])
    
    for i in range(1, len(strikes)):
        curr = strikes[i]
        prev = strikes[i-1]
        
        dx = abs(curr['pt'][0] - prev['pt'][0])
        df = curr['frame'] - prev['frame']
        
        # Check for ~2x gap
        # Condition: Time is ~2x AND Space is ~2x (approx)
        # If we missed a step, the interval should be double.
        
        ratio_f = df / med_df
        ratio_x = dx / med_dx
        
        # We look for ratio ~ 2.0 (e.g. 1.5 to 2.5)
        if min_gap_ratio <= ratio_f <= max_gap_ratio:
            # INTERPOLATE
            # Midpoint
            interp_frame = int((curr['frame'] + prev['frame']) / 2)
            interp_x = (curr['pt'][0] + prev['pt'][0]) / 2
            interp_y = (curr['pt'][1] + prev['pt'][1]) / 2
            
            # Guess side: If prev=Left, Curr=Left (missed Right), then Interp=Right.
            # If Prev=Left, Curr=Right (standard), then we wouldn't have a gap?
            # Actually, if we miss a step, we usually see L -> L (missed R).
            # So the side usually repeats.
            interp_side = 'right' if prev['side'] == 'left' else 'left'
            
            new_step = {
                'frame': interp_frame,
                'side': interp_side,
                'confidence': 0.5, # Low conf
                'pt': (interp_x, interp_y),
                'interpolated': True
            }
            new_strikes.append(new_step)
            
        new_strikes.append(curr)
        
    return new_strikes


def compute_height_scale_factor(body_h_arr, pose_landmarks_list, frame_height, frame_width,
                                 athlete_height_m, plant_frame, search_window=8):
    """
    Find the frame near plant_frame where the athlete appears tallest (most upright)
    and compute a meters-per-pixel scale factor.

    Uses eye-to-ankle distance (highest eye to lowest ankle) to measure body height,
    then applies a 0.95 correction for the remaining skull above the eyes.

    Args:
        body_h_arr: Per-frame body height array (used only for length/validity check).
        pose_landmarks_list: List of MediaPipe pose landmark objects.
        frame_height: Video frame height in pixels.
        frame_width: Video frame width in pixels.
        athlete_height_m: Known athlete height in meters.
        plant_frame: Frame index of the pole plant.
        search_window: Number of frames +/- around plant_frame to search.

    Returns:
        Tuple of (scale_m_per_px, calib_info) where:
            scale_m_per_px: meters per pixel (float), or None on failure
            calib_info: dict with 'frame', 'top_px', 'ankle_px', 'body_height_px'
                        for drawing the visual indicator, or None on failure
    """
    if pose_landmarks_list is None or len(pose_landmarks_list) == 0 or athlete_height_m <= 0:
        return None, None

    if plant_frame is None:
        plant_frame = len(pose_landmarks_list) - 1

    search_start = max(0, plant_frame - search_window)
    search_end = min(len(pose_landmarks_list), plant_frame + search_window + 1)

    if search_end <= search_start:
        return None, None

    EYE_TO_TOP_RATIO = 0.95  # eye-to-ankle is ~95% of total height

    # Compute eye-to-ankle distance for each frame in the search window
    best_frame = None
    best_dist_norm = 0.0

    for f in range(search_start, search_end):
        if f >= len(pose_landmarks_list) or pose_landmarks_list[f] is None:
            continue

        lm = pose_landmarks_list[f].landmark
        l_eye = lm[mp_pose.PoseLandmark.LEFT_EYE.value]
        r_eye = lm[mp_pose.PoseLandmark.RIGHT_EYE.value]
        l_ankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        eye_y = min(l_eye.y, r_eye.y)       # highest eye
        ankle_y = max(l_ankle.y, r_ankle.y)  # lowest ankle
        dist = abs(ankle_y - eye_y)

        if dist > best_dist_norm:
            best_dist_norm = dist
            best_frame = f

    if best_frame is None or best_dist_norm <= 1e-4:
        return None, None

    # Re-extract landmarks from the best frame for the final measurement + visual
    lm = pose_landmarks_list[best_frame].landmark
    l_eye = lm[mp_pose.PoseLandmark.LEFT_EYE.value]
    r_eye = lm[mp_pose.PoseLandmark.RIGHT_EYE.value]
    l_ankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    r_ankle = lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

    eye_y = min(l_eye.y, r_eye.y)
    eye_x = l_eye.x if l_eye.y <= r_eye.y else r_eye.x
    ankle_y = max(l_ankle.y, r_ankle.y)
    ankle_x = l_ankle.x if l_ankle.y >= r_ankle.y else r_ankle.x

    eye_to_ankle_px = best_dist_norm * frame_height
    effective_height_m = athlete_height_m * EYE_TO_TOP_RATIO
    scale_m_per_px = effective_height_m / eye_to_ankle_px

    eye_px = (int(eye_x * frame_width), int(eye_y * frame_height))
    ankle_px = (int(ankle_x * frame_width), int(ankle_y * frame_height))

    calib_info = {
        'frame': best_frame,
        'top_px': eye_px,
        'ankle_px': ankle_px,
        'body_height_px': eye_to_ankle_px,
        'scale_m_per_px': scale_m_per_px,
    }

    return scale_m_per_px, calib_info


def compute_max_hip_height(pose_landmarks_list, frame_height, frame_width,
                            plant_frame, end_frame, scale_m_per_px,
                            ground_search_window=5):
    """
    Compute max hip height and chest height, accounting for sideways clearance.
    
    Returns:
        dict with height_m, peak_hip_px, predicted_clear_m, pc_px, pc_frame, etc.
    """
    if (pose_landmarks_list is None or len(pose_landmarks_list) == 0
            or scale_m_per_px is None or scale_m_per_px <= 0
            or plant_frame is None or end_frame is None
            or end_frame <= plant_frame):
        return None

    # --- 1. Ground reference: lowest foot point near plant ---
    FOOT_LANDMARKS = [
        mp_pose.PoseLandmark.LEFT_HEEL.value,
        mp_pose.PoseLandmark.RIGHT_HEEL.value,
        mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value,
        mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value,
    ]

    ground_start = max(0, plant_frame - ground_search_window)
    ground_end = min(len(pose_landmarks_list), plant_frame + ground_search_window + 1)

    ground_y_norm = 0.0  # max normalized Y (lowest point)
    for f in range(ground_start, ground_end):
        if f >= len(pose_landmarks_list) or pose_landmarks_list[f] is None:
            continue
        lm = pose_landmarks_list[f].landmark
        for idx in FOOT_LANDMARKS:
            if lm[idx].visibility > 0.3:
                ground_y_norm = max(ground_y_norm, lm[idx].y)

    if ground_y_norm <= 1e-4:
        return None

    # --- 2. Peak hip & chest tracking ---
    best_hip_y_norm = 1.0  # min normalized Y (highest point)
    best_hip_x_norm = 0.5
    best_hip_frame = plant_frame

    best_chest_y_norm = 1.0
    best_chest_frame = plant_frame

    for f in range(plant_frame, min(end_frame, len(pose_landmarks_list))):
        if pose_landmarks_list[f] is None:
            continue
        lm = pose_landmarks_list[f].landmark
        
        # Hips
        l_hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]
        if l_hip.visibility >= 0.3 or r_hip.visibility >= 0.3:
            hip_mid_y = (l_hip.y + r_hip.y) / 2.0
            hip_mid_x = (l_hip.x + r_hip.x) / 2.0
            if hip_mid_y < best_hip_y_norm:
                best_hip_y_norm = hip_mid_y
                best_hip_x_norm = hip_mid_x
                best_hip_frame = f
                
        # Chest (Shoulders)
        l_shldr = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shldr = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        if l_shldr.visibility >= 0.3 or r_shldr.visibility >= 0.3:
            chest_mid_y = (l_shldr.y + r_shldr.y) / 2.0
            if chest_mid_y < best_chest_y_norm:
                best_chest_y_norm = chest_mid_y
                best_chest_frame = f

    if best_hip_y_norm >= ground_y_norm:
        return None

    # --- 3. Use the LOWER point on the peak frame ---
    
    # Hips:
    lm_hip = pose_landmarks_list[best_hip_frame].landmark
    l_h = lm_hip[mp_pose.PoseLandmark.LEFT_HIP.value]
    r_h = lm_hip[mp_pose.PoseLandmark.RIGHT_HIP.value]
    if l_h.y > r_h.y:
        lower_hip_y_norm = l_h.y
        lower_hip_x_norm = l_h.x
    else:
        lower_hip_y_norm = r_h.y
        lower_hip_x_norm = r_h.x

    # Apply 2-inch drop to the LOWER hip point to account for sideways clearance
    offset_inches = 2.0
    offset_m = offset_inches * 0.0254
    offset_y_norm = (offset_m / scale_m_per_px) / frame_height
    annotated_hip_y_norm = lower_hip_y_norm + offset_y_norm
    annotated_hip_x_norm = lower_hip_x_norm

    # Chest (Shoulders):
    lm_chest = pose_landmarks_list[best_chest_frame].landmark
    l_s = lm_chest[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    r_s = lm_chest[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    if l_s.y > r_s.y:
        lower_chest_y_norm = l_s.y
        lower_chest_x_norm = l_s.x
    else:
        lower_chest_y_norm = r_s.y
        lower_chest_x_norm = r_s.x

    # --- 4. Convert to real-world units ---
    # Hip max height (using annotated 2-inch dropped point)
    hip_height_px = (ground_y_norm - annotated_hip_y_norm) * frame_height
    hip_height_m = hip_height_px * scale_m_per_px

    # Chest max height
    chest_height_px = (ground_y_norm - lower_chest_y_norm) * frame_height
    chest_height_m = chest_height_px * scale_m_per_px

    peak_hip_px = (int(annotated_hip_x_norm * frame_width), int(annotated_hip_y_norm * frame_height))
    peak_chest_px = (int(lower_chest_x_norm * frame_width), int(lower_chest_y_norm * frame_height))
    ground_y_px = int(ground_y_norm * frame_height)

    # Predicted max clear height = MIN(hip, chest) - 6 inches
    clear_offset_m = 6 * 0.0254  # 6 inches in meters
    if hip_height_m < chest_height_m:
        predicted_clear_m = hip_height_m - clear_offset_m
        pc_controlling_frame = best_hip_frame
        pc_px = (peak_hip_px[0], int(peak_hip_px[1] + (clear_offset_m / scale_m_per_px)))
    else:
        predicted_clear_m = chest_height_m - clear_offset_m
        pc_controlling_frame = best_chest_frame
        pc_px = (peak_chest_px[0], int(peak_chest_px[1] + (clear_offset_m / scale_m_per_px)))

    return {
        'height_m': hip_height_m,
        'height_cm': hip_height_m * 100.0,
        'height_in': hip_height_m * 39.3701,
        'peak_frame': best_hip_frame,
        'peak_hip_px': peak_hip_px,
        'ground_y_px': ground_y_px,
        'predicted_clear_m': predicted_clear_m,
        'predicted_clear_in': predicted_clear_m * 39.3701,
        'chest_height_m': chest_height_m,
        'peak_chest_frame': best_chest_frame,
        'peak_chest_px': peak_chest_px,
        'pc_frame': pc_controlling_frame,
        'pc_px': pc_px
    }


def compute_approach_velocity(pose_landmarks_list, start_frame, plant_frame,
                              fps, frame_width, frame_height,
                              scale_m_per_px=None, smooth_window=5):
    """
    Compute athlete velocity over the approach phase using hip center displacement.

    Tracks frame-to-frame horizontal displacement of the hip midpoint from
    start_frame through plant_frame + 0.25 seconds.

    Args:
        pose_landmarks_list: list of MediaPipe pose landmark objects (one per frame)
        start_frame: first frame of the approach
        plant_frame: frame where the pole is planted
        fps: video frame rate
        frame_width: video width in pixels
        frame_height: video height in pixels
        scale_m_per_px: meters-per-pixel calibration (None = output in pixels/s)
        smooth_window: number of frames for moving-average smoothing

    Returns:
        list of dicts: [{'frame': int, 'time_s': float, 'velocity_m_s': float,
                         'velocity_mph': float, 'hip_x_px': float}, ...]
        or empty list on failure
    """
    if not pose_landmarks_list or fps <= 0:
        return []

    # Stop at plant frame — post-plant horizontal velocity is meaningless
    end_frame = min(plant_frame, len(pose_landmarks_list) - 1)

    # Extract hip midpoint X in pixels for each frame
    hip_x = []
    for f in range(start_frame, end_frame + 1):
        if f >= len(pose_landmarks_list) or pose_landmarks_list[f] is None:
            hip_x.append(np.nan)
            continue
        lm = pose_landmarks_list[f].landmark
        l_hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]
        if l_hip.visibility < 0.3 and r_hip.visibility < 0.3:
            hip_x.append(np.nan)
            continue
        mid_x_px = (l_hip.x + r_hip.x) / 2.0 * frame_width
        hip_x.append(mid_x_px)

    hip_x = np.array(hip_x, dtype=float)

    # Interpolate small gaps
    valid = ~np.isnan(hip_x)
    if valid.sum() < 3:
        return []
    indices = np.arange(len(hip_x))
    hip_x[~valid] = np.interp(indices[~valid], indices[valid], hip_x[valid])

    # Frame-to-frame displacement (absolute, handles left-to-right or right-to-left)
    dx = np.abs(np.diff(hip_x))  # pixels per frame
    velocity_px_s = dx * fps  # pixels per second

    # Smooth with moving average
    if smooth_window > 1 and len(velocity_px_s) >= smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        velocity_px_s = np.convolve(velocity_px_s, kernel, mode='same')

    # Convert to real-world units if scale available
    if scale_m_per_px and scale_m_per_px > 0:
        velocity_m_s = velocity_px_s * scale_m_per_px
    else:
        velocity_m_s = velocity_px_s  # fallback: px/s

    results = []
    for i in range(len(velocity_m_s)):
        f_idx = start_frame + i + 1  # velocity is between frame i and i+1
        t = (f_idx - start_frame) / fps
        v_ms = float(velocity_m_s[i])
        v_mph = v_ms * 2.23694 if scale_m_per_px else 0.0
        results.append({
            'frame': f_idx,
            'time_s': round(t, 4),
            'velocity_m_s': round(v_ms, 3),
            'velocity_mph': round(v_mph, 2),
            'hip_x_px': round(float(hip_x[i + 1]), 1),
        })

    return results

