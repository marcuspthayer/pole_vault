import cv2
import math
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from collections import Counter
import os
import shutil

mp_pose = mp.solutions.pose

def heal_pole_mask(poly_pts, shape):
    """
    Reconstructs a binary mask from polygon points and heals gaps
    using morphological closing.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    if len(poly_pts) > 0:
        pts = np.array(poly_pts, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)
        
    # Morphological Close to bridge gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def get_pole_centerline_points(mask):
    """
    Skeletonizes the binary mask to find the centerline.
    Returns:
        points: (N, 2) array of (x, y) coordinates ordered from one tip to the other.
    """
    # 1. Skeletonize using iterative erosion (or Distance Transform Ridge)
    # Distance Transform is faster and gives a centered ridge more reliably for thick shapes
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    
    img = mask.copy()
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    
    # Limit iterations to avoid infinite loops, though pole should erode fast
    for _ in range(100):
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
            
    # 2. Extract Points
    ys, xs = np.nonzero(skel)
    if len(xs) < 2:
        return None
        
    points = np.column_stack((xs, ys))
    
    # Sort Points to form a line
    # Simple strategy: Find point furthest from centroid, call it start.
    centroid = np.mean(points, axis=0)
    dists = np.sum((points - centroid)**2, axis=1)
    start_idx = np.argmax(dists)
    
    # Walk the graph
    ordered_points = [points[start_idx]]
    current_idx = start_idx
    mask_visited = np.zeros(len(points), dtype=bool)
    mask_visited[start_idx] = True
    
    while len(ordered_points) < len(points):
        curr = points[current_idx]
        unvisited_indices = np.where(~mask_visited)[0]
        if len(unvisited_indices) == 0:
            break
            
        unvisited_pts = points[unvisited_indices]
        d_sq = np.sum((unvisited_pts - curr)**2, axis=1)
        
        nearest_local_idx = np.argmin(d_sq)
        nearest_global_idx = unvisited_indices[nearest_local_idx]
        
        if d_sq[nearest_local_idx] > 900: 
             break
             
        ordered_points.append(points[nearest_global_idx])
        mask_visited[nearest_global_idx] = True
        current_idx = nearest_global_idx
        
    return np.array(ordered_points)

def trim_pole_tip(centerline, hand_pt, fit_ratio=0.75, deviation_thresh=5.0):
    """
    Trims the 'Tip' end of the pole if it deviates from the straight line 
    defined by the 'Hand' end (first fit_ratio%).
    
    Args:
        centerline: (N, 2) ordered points.
        hand_pt: (2,) Point of hand, used to orient centerline.
    Returns:
        trimmed_centerline: (M, 2) where M <= N.
    """
    if len(centerline) < 10:
        return centerline
        
    # 1. Orient Centerline: Start should be closest to hand
    d_start = np.sum((centerline[0] - hand_pt)**2)
    d_end = np.sum((centerline[-1] - hand_pt)**2)
    
    if d_end < d_start:
        centerline = centerline[::-1]
        
    # Now centerline[0] is Hand End, centerline[-1] is Tip End
    
    # 2. Fit Line to first X%
    n_points = len(centerline)
    n_fit = int(n_points * fit_ratio)
    if n_fit < 5:
        return centerline
        
    fit_points = centerline[:n_fit]
    
    # cv2.fitLine returns (vx, vy, x0, y0) normalized vector and point on line
    [vx, vy, x0, y0] = cv2.fitLine(fit_points, cv2.DIST_L2, 0, 0.01, 0.01)
    
    # 3. Check deviation of the TAIL (remaining points)
    # We walk BACKWARDS from the Tip (-1) towards the Hand
    # The moment we find a point that is ON the line (low deviation), we assume 
    # everything before it is also good (or at least connected to the main body).
    # Basically we want to cut off the dangling tail.
    
    # Actually, simpler: Walk from n_fit towards end. 
    # Find the *first* point that deviates too much, and cut everything after it.
    
    cut_idx = n_points # Default: keep all
    
    start_check_idx = n_fit
    
    for i in range(start_check_idx, n_points):
        # Distance from point to line
        pt = centerline[i]
        
        # d = |(p - p0) - proj| ? 
        # dist = |det([p-p0, v])| / |v|  (since v is normalized |v|=1)
        # dist = (x-x0)*(-vy) + (y-y0)*vx
        dist = abs((pt[0] - x0) * (-vy) + (pt[1] - y0) * vx)
        
        if dist > deviation_thresh:
            cut_idx = i
            break
            
    return centerline[:cut_idx]

def project_hand_to_curve(hand_pt, centerline):
    """
    Finds the closest point on the centerline to the hand_pt.
    Returns:
        proj_pt: (x, y) on the curve
        arc_len: distance along curve from start to proj_pt
        fraction: 0..1 (approximate)
    """
    if centerline is None or len(centerline) == 0:
        return None, 0, 0
        
    # Find closest point index
    dists = np.sum((centerline - hand_pt)**2, axis=1)
    idx = np.argmin(dists)
    
    closest_pt = centerline[idx]
    
    # Compute Arc Length up to idx
    # sum of Euclidian distances between consecutive points
    diffs = centerline[1:] - centerline[:-1]
    segment_lens = np.sqrt(np.sum(diffs**2, axis=1))
    
    arc_len_val = np.sum(segment_lens[:idx])
    total_len = np.sum(segment_lens)
    
    return closest_pt, arc_len_val, (arc_len_val / total_len if total_len > 0 else 0)

def get_hand_positions(pose_landmarks, frame_w, frame_h):
    if not pose_landmarks:
        return None, None
    lm = pose_landmarks.landmark
    l_wrist = lm[mp_pose.PoseLandmark.LEFT_WRIST]
    r_wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
    def to_pix(l): return (l.x * frame_w, l.y * frame_h)
    return to_pix(l_wrist), to_pix(r_wrist)

def select_consensus_length(measurements, name="p1", output_dir=None, bin_size=10):
    if not measurements:
        return None
        
    lengths = np.array([m['len'] for m in measurements])
    max_len = np.max(lengths)
    
    # 1. Filter outliers (must be at least 50% of max length seen)
    valid_indices = np.where(lengths > 0.5 * max_len)[0]
    if len(valid_indices) == 0:
        valid_indices = [np.argmax(lengths)]
        
    valid_measurements = [measurements[i] for i in valid_indices]
    valid_lengths = lengths[valid_indices]
    
    # 2. Binning
    min_v = np.min(valid_lengths)
    max_v = np.max(valid_lengths)
    
    if max_v - min_v < bin_size:
        bins = [min_v, max_v + 1]
    else:
        bins = np.arange(int(min_v), int(max_v) + bin_size + 1, bin_size)
    
    hist, bin_edges = np.histogram(valid_lengths, bins=bins)
    
    # --- DETAILED BIN LOGGING ---
    print(f"\n[CONSENSUS-DEBUG] Binning Detail for {name} (bin_size={bin_size:.1f}px)")
    for b in range(len(hist)):
        lower = bin_edges[b]
        upper = bin_edges[b+1]
        count = hist[b]
        in_bin = [l for l in valid_lengths if lower <= l < upper]
        if count > 0:
            val_str = ", ".join([f"{x:.1f}" for x in in_bin])
            print(f"  Bin {b:2d} [{lower:6.1f}, {upper:6.1f}]: count={count:2d} | values=[{val_str}]")
    
    # 3. Find Dominant Bin
    max_bin_idx = np.argmax(hist)
    bin_start = bin_edges[max_bin_idx]
    bin_end = bin_edges[max_bin_idx+1]
    
    # 4. Filter measurements in this bin
    cluster = [m for m in valid_measurements if bin_start <= m['len'] < bin_end]
    
    if not cluster:
        return valid_measurements[0]
        
    # 5. Average of cluster
    cluster_mean = np.mean([m['len'] for m in cluster])
    
    best_measurement = min(cluster, key=lambda x: abs(x['len'] - cluster_mean))
    best_measurement['consensus_mean'] = cluster_mean
    
    print(f"[DEBUG-CONSENSUS] {name}: Valid: {len(valid_lengths)}/{len(measurements)}. Peak Bin: {bin_start}-{bin_end}. Mean: {cluster_mean:.1f}")

    if output_dir:
        plt.figure(figsize=(6, 4))
        plt.hist(lengths, bins=20, alpha=0.5, label='All', color='gray')
        plt.hist(valid_lengths, bins=bins, alpha=0.7, label='Valid', color='blue')
        plt.axvspan(bin_start, bin_end, color='green', alpha=0.3, label='Chosen')
        plt.axvline(cluster_mean, color='red', linestyle='--', label=f'Mean: {cluster_mean:.0f}')
        plt.title(f"Pole Length Distribution ({name})")
        plt.xlabel("Length (px)")
        plt.ylabel("Count")
        plt.legend()
        hist_path = os.path.join(output_dir, f"hist_{name}.png")
        plt.savefig(hist_path)
        plt.close()
        print(f"[DEBUG] Saved Histogram to {hist_path}")

    return best_measurement

def analyze_pole_segment_single_frame(
    pole_results, pose_results, video_path,
    frame_idx, frame_w, frame_h, phase="phase1"
):
    """
    Analyze a single user-selected frame for pole length measurement.

    Args:
        phase: "phase1" (tip-to-bottom-hand) or "phase2" (top-to-bottom-hand)

    Returns:
        Measurement dict compatible with the multi-frame consensus output,
        or None on failure.
    """
    if frame_idx >= len(pole_results) or frame_idx >= len(pose_results):
        return None

    p_res = pole_results[frame_idx]
    pose_res = pose_results[frame_idx]

    if not p_res or not p_res.masks or not pose_res:
        return None

    poly = p_res.masks.xy[0]
    if len(poly) < 3:
        return None

    mask = heal_pole_mask(poly, (frame_h, frame_w))
    centerline = get_pole_centerline_points(mask)

    if centerline is None or len(centerline) < 5:
        return None

    lh, rh = get_hand_positions(pose_res.pose_landmarks, frame_w, frame_h)
    if not lh or not rh:
        return None

    if phase == "phase1":
        # Tip-to-forward-hand: tip is the mask point FURTHEST from athlete center.
        # Use the polygon directly — the centerline skeleton can be squiggly/looped.
        athlete_p = np.array([(lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2])
        dists_to_athlete = np.hypot(poly[:, 0] - athlete_p[0], poly[:, 1] - athlete_p[1])
        tip_idx = np.argmax(dists_to_athlete)
        pole_end = np.array(poly[tip_idx])
    else:
        # Phase 2: top of pole above the athlete's hand (smallest Y = highest in frame)
        # Use the actual mask's topmost point rather than centerline endpoints,
        # which may not extend to the true top of the pole mask.
        top_idx = np.argmin(poly[:, 1])
        pole_end = np.array(poly[top_idx])

    if phase == "phase1":
        # Forward hand: whichever wrist is closer to the tip (more forward in running direction)
        d_lh = np.hypot(lh[0] - pole_end[0], lh[1] - pole_end[1])
        d_rh = np.hypot(rh[0] - pole_end[0], rh[1] - pole_end[1])
        lead_hand = lh if d_lh < d_rh else rh
    else:
        # Phase 2: lower hand in frame (highest Y coordinate)
        lead_hand = lh if lh[1] > rh[1] else rh

    # Straight-line distance from forward hand to pole endpoint
    straight_len = np.hypot(lead_hand[0] - pole_end[0], lead_hand[1] - pole_end[1])

    area = np.sum(mask > 0)
    actual_len = len(centerline)
    avg_thickness = area / actual_len if actual_len > 10 else 10

    return {
        'len': straight_len,
        'consensus_mean': straight_len,
        'frame': frame_idx,
        'pts': (pole_end, np.array(lead_hand)),
        'centerline': centerline,
        'hand_pt': lead_hand,
        'width': avg_thickness,
    }


def analyze_pole_segments(
    pole_results,
    pose_results,
    video_path,
    start_frame,
    plant_frame
):
    """
    Two-phase calibration using Skeletonization + Arc Length + Consensus Clustering.
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    save_dir = os.path.join(os.getcwd(), "debug_output")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # --- Phase 1: Tip to Bottom Hand (Early Run) ---
    # PRE-PLANT: Pole is straight. Apply Linear Trimming.
    
    # User Request: "expand... to all frames from start frame to the frame where the pole drops below 1 degree above the horizontal"
    # We iterate until angle < 1.0 degree.
    
    p1_measurements = []
    
    print(f"[DEBUG] Phase 1: Checking frames starting from {start_frame}")
    
    for f_idx in range(start_frame, plant_frame):
        if f_idx >= len(pole_results) or f_idx >= len(pose_results): continue
        p_res = pole_results[f_idx]
        pose_res = pose_results[f_idx]
        
        if not p_res or not p_res.masks or not pose_res: continue
        
        poly = p_res.masks.xy[0]
        if len(poly) < 3: continue
        
        mask = heal_pole_mask(poly, (height, width))
        centerline = get_pole_centerline_points(mask)
        
        if centerline is None or len(centerline) < 5: continue
        
        # Hands
        lh, rh = get_hand_positions(pose_res.pose_landmarks, width, height)
        if not lh: continue
        
        # Identify Tip: mask point FURTHEST from the athlete's hands.
        # Use the polygon directly — the centerline skeleton can be squiggly/looped.
        athlete_p = np.array([(lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2])
        dists_to_athlete = np.hypot(poly[:, 0] - athlete_p[0], poly[:, 1] - athlete_p[1])
        tip_idx = np.argmax(dists_to_athlete)
        tip = np.array(poly[tip_idx])
            
        # Forward hand: whichever wrist is closer to the tip (more forward in running direction)
        d_lh = np.hypot(lh[0] - tip[0], lh[1] - tip[1])
        d_rh = np.hypot(rh[0] - tip[0], rh[1] - tip[1])
        lead_hand = lh if d_lh < d_rh else rh

        # --- REFINED WIDTH (THICKNESS) ---
        area = np.sum(mask > 0)
        actual_len = len(centerline)
        avg_thickness = area / actual_len if actual_len > 10 else 10

        # --- ANGLE CHECK ---
        dy = lead_hand[1] - tip[1]
        dx = abs(lead_hand[0] - tip[0])
        if dx < 1: dx = 1
        angle_deg = np.degrees(np.arctan2(dy, dx))

        if f_idx > start_frame + 5 and angle_deg < -5.0:
            print(f"[DEBUG] Pole Tipped Down ({angle_deg:.1f} deg) at frame {f_idx}. Stopping Phase 1.")
            break

        # Straight-line distance from forward hand to tip
        straight_len = np.hypot(lead_hand[0] - tip[0], lead_hand[1] - tip[1])

        p1_measurements.append({
            'len': straight_len,
            'frame': f_idx,
            'pts': (tip, np.array(lead_hand)),
            'centerline': centerline,
            'hand_pt': lead_hand,
            'width': avg_thickness
        })

    # --- Phase 2: Top to Bottom Hand (Plant) ---
    # POLE BENDS: Do NOT Trim!
    
    p2_start = max(0, plant_frame - 2)
    p2_end = min(total_frames, plant_frame + 6)
    p2_measurements = []
    
    for f_idx in range(p2_start, p2_end):
        if f_idx >= len(pole_results) or f_idx >= len(pose_results): continue
        p_res = pole_results[f_idx]
        pose_res = pose_results[f_idx]
        
        if not p_res or not p_res.masks or not pose_res: continue
        
        poly = p_res.masks.xy[0]
        if len(poly) < 3: continue
        
        mask = heal_pole_mask(poly, (height, width))
        centerline = get_pole_centerline_points(mask)
        
        if centerline is None or len(centerline) < 5: continue
        
        # Identify Top (smallest Y = highest in frame)
        # Use the actual mask's topmost point rather than centerline endpoints,
        # which may not extend to the true top of the pole mask.
        top_idx = np.argmin(poly[:, 1])
        top = np.array(poly[top_idx])
            
        lh, rh = get_hand_positions(pose_res.pose_landmarks, width, height)
        if lh is None or rh is None: continue

        # Lower hand in frame (highest Y coordinate) for Phase 2
        lead_hand = lh if lh[1] > rh[1] else rh

        # Refined thickness for P2
        area = np.sum(mask > 0)
        actual_len = len(centerline)
        avg_thickness = area / actual_len if actual_len > 10 else 10

        # Straight-line distance from forward hand to top
        straight_len = np.hypot(lead_hand[0] - top[0], lead_hand[1] - top[1])

        p2_measurements.append({
            'len': straight_len,
            'frame': f_idx,
            'pts': (top, np.array(lead_hand)),
            'centerline': centerline,
            'hand_pt': lead_hand,
            'width': avg_thickness
        })

    res_p1 = None
    if p1_measurements:
        avg_w1 = np.mean([m['width'] for m in p1_measurements])
        res_p1 = select_consensus_length(p1_measurements, name="Phase1_TipToHand", output_dir=save_dir, bin_size=avg_w1)
        
    res_p2 = None
    if p2_measurements:
        avg_w2 = np.mean([m['width'] for m in p2_measurements])
        res_p2 = select_consensus_length(p2_measurements, name="Phase2_TopToHand", output_dir=save_dir, bin_size=avg_w2)
    
    return res_p1, res_p2

def _find_top_backward_point(poly, backward_dir):
    """Find the point on a polygon that is highest and furthest backward (opposite approach).

    Uses equal weighting (50/50) between height and backward position.
    """
    ys = poly[:, 1]
    xs = poly[:, 0]
    y_range = ys.max() - ys.min()
    if y_range < 1e-8:
        return poly[np.argmin(ys)]
    y_norm = (ys - ys.min()) / y_range  # 0=top, 1=bottom
    x_backward = xs * backward_dir  # higher = more backward
    x_range = x_backward.max() - x_backward.min()
    if x_range < 1e-8:
        return poly[np.argmin(ys)]
    x_norm = (x_backward - x_backward.min()) / x_range  # 0=forward, 1=backward
    score = -y_norm + x_norm  # equal weight: high + backward
    return poly[np.argmax(score)]


def analyze_pole_bend(pole_results, pose_results, plant_frame, p1_len, p2_len, width, height, end_frame, manual_bend_range=None, fps=None):
    """
    Reconstructs occluded tip at plant_frame and tracks chord length to 'top' of pole.

    Args:
        manual_bend_range: If provided, a (start_frame, end_frame) tuple defining the
                           search window for max bend instead of using the full auto window.
        fps: Video frame rate, used for time-based thresholds. Falls back to 30 if None.
    Returns: {
        'max_bend': float (smoothed),
        'bend_series': list (smoothed),
        'raw_series': list,
        'max_bend_frame': int,
        'points': (reconstructed_tip, top_pt_at_max_bend)
    }
    """
    if plant_frame >= len(pole_results) or plant_frame >= len(pose_results):
        return None

    effective_fps = fps if fps else 30.0

    # 1. Reconstruct Tip at plant_frame
    p_res = pole_results[plant_frame]
    pose_res = pose_results[plant_frame]
    if not p_res or not p_res.masks or not pose_res:
        return None

    poly = p_res.masks.xy[0]
    mask = heal_pole_mask(poly, (height, width))
    centerline = get_pole_centerline_points(mask)
    if centerline is None or len(centerline) < 5:
        return None

    lh, rh = get_hand_positions(pose_res.pose_landmarks, width, height)
    if lh is None or rh is None:
        return None

    # User: "Hand lower down in the frame"
    bottom_hand = lh if lh[1] > rh[1] else rh

    total_len = p1_len + p2_len
    if total_len == 0:
        return None

    # Fit a line through the mask pixels to get the true pole angle.
    # This is more accurate than using the top->hand vector, which can be skewed
    # by hand position and doesn't reflect the actual mask orientation.
    mask_ys, mask_xs = np.where(mask > 0)
    if len(mask_xs) < 10:
        return None
    mask_pts = np.column_stack((mask_xs, mask_ys)).astype(np.float32)
    [vx, vy, _x0, _y0] = cv2.fitLine(mask_pts, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy = float(vx), float(vy)

    # Ensure the direction vector points downward (from sky tip toward the ground),
    # i.e. positive vy (larger Y = lower in the frame).
    if vy < 0:
        vx, vy = -vx, -vy

    unit_vec = np.array([vx, vy])

    # Backward direction: opposite the approach (running) direction.
    # vx points in the running direction (from sky-end toward tip/ground).
    backward_dir = -np.sign(vx) if abs(vx) > 1e-6 else 1.0

    # Identify the Top point at the plant frame using up-and-backward scoring
    poly_plant = p_res.masks.xy[0]
    if len(poly_plant) == 0:
        return None
    top_at_plant = _find_top_backward_point(poly_plant, backward_dir)

    # Reconstructed tip: start at the visible sky-end of the pole and travel
    # exactly total_len pixels along the fitted pole angle.
    reconstructed_tip = top_at_plant + unit_vec * total_len

    # 2. Track Chord Length
    bend_history = [] # List of {'f': frame, 'pct': float, 'top': pt}
    total_len = p1_len + p2_len
    if total_len == 0: return None

    # Analyze from plant to end_frame (or until pole nears vertical)
    analysis_end = min(len(pole_results), end_frame + 1)
    min_window_frames = int(0.75 * effective_fps)

    for f_idx in range(plant_frame, analysis_end):
        pr = pole_results[f_idx]
        if not pr or not pr.masks:
            bend_history.append({'f': f_idx, 'pct': 100.0, 'top': None})
            continue

        poly_f = pr.masks.xy[0]

        # Check pole angle from vertical — stop if nearly straight and past min window
        frames_elapsed = f_idx - plant_frame
        if len(poly_f) >= 10 and frames_elapsed > min_window_frames:
            poly_pts_f = poly_f.astype(np.float32)
            [fvx, fvy, _, _] = cv2.fitLine(poly_pts_f, cv2.DIST_L2, 0, 0.01, 0.01)
            angle_from_vert = math.degrees(math.atan2(abs(float(fvx)), abs(float(fvy))))
            if angle_from_vert < 5.0:
                break

        # Identify Top using up-and-backward scoring
        top_pt = _find_top_backward_point(poly_f, backward_dir)

        chord_len = np.hypot(top_pt[0] - reconstructed_tip[0], top_pt[1] - reconstructed_tip[1])
        bend_pct = (chord_len / total_len) * 100
        bend_history.append({'f': f_idx, 'pct': float(bend_pct), 'top': top_pt})

    if not bend_history:
        return None

    # 2b. Filter outlier short cord lengths (bad mask detections)
    valid_pcts = np.array([e['pct'] for e in bend_history if e['top'] is not None])
    min_filter_frames = int(0.33 * effective_fps)
    if len(valid_pcts) >= max(min_filter_frames, 5):
        q1, q3 = np.percentile(valid_pcts, [25, 75])
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        for e in bend_history:
            if e['top'] is not None and e['pct'] < lower_fence:
                e['pct'] = 100.0  # neutralize outlier so it won't win min-search

    # Smooth the series a bit (3-frame moving average)
    raw_series = [w['pct'] for w in bend_history]
    smoothed_series = []
    for i in range(len(bend_history)):
        window = bend_history[max(0, i-1):min(len(bend_history), i+2)]
        avg_pct = np.mean([w['pct'] for w in window])
        smoothed_series.append(avg_pct)
        bend_history[i]['smoothed_pct'] = avg_pct

    # Find the frame with absolute minimum smoothed percentage
    if manual_bend_range is not None:
        # Search within user-selected frame range
        range_start, range_end = manual_bend_range
        candidates = [e for e in bend_history if range_start <= e['f'] <= range_end]
        if candidates:
            best_entry = min(candidates, key=lambda x: x.get('smoothed_pct', x['pct']))
        else:
            # Fallback: use full history
            best_entry = min(bend_history, key=lambda x: x.get('smoothed_pct', x['pct']))
    else:
        best_entry = min(bend_history, key=lambda x: x.get('smoothed_pct', x['pct']))

    # 3. Fit a quadratic curve to the smoothed series to find the true mathematical minimum
    # X = frames from 0, Y = bend percentage
    x_frames = np.arange(len(smoothed_series))

    # We only want to fit if we have enough points (e.g. > 5 frames of bend)
    poly_fit_series = None
    poly_min_val = None

    if len(smoothed_series) > 5:
        # Fit a 2nd degree polynomial (quadratic): y = ax^2 + bx + c
        # We expect a U-shape where it dips down to the minimum bend percentage
        z = np.polyfit(x_frames, smoothed_series, 2)
        p = np.poly1d(z)
        poly_fit_series = p(x_frames).tolist()

        # The theoretical minimum of a quadratic ax^2 + bx + c is at x = -b / (2a)
        # We only care if the parabola opens upwards (a > 0)
        a, b, c = z
        if a > 0:
            min_x = -b / (2 * a)
            # Evaluate the polynomial at the minimum x
            poly_min_val = float(p(min_x))

            # If the theoretical minimum happens outside our video window, bounds-check it
            if min_x < 0:
                poly_min_val = poly_fit_series[0]
            elif min_x > x_frames[-1]:
                poly_min_val = poly_fit_series[-1]

    # Fallback to smoothed minimum if the polynomial fit is degenerate (e.g. inverted parabola)
    if poly_min_val is None:
        poly_min_val = best_entry['smoothed_pct']
        poly_fit_series = smoothed_series # just output smoothed

    return {
        'max_bend': best_entry['smoothed_pct'], # Original smoothed value
        'poly_max_bend': poly_min_val,          # True mathematical minimum from the fit curve
        'bend_series': smoothed_series,
        'poly_series': poly_fit_series,         # The U-shaped fitted curve values
        'raw_series': raw_series,
        'max_bend_frame': best_entry['f'],
        'points': (reconstructed_tip, best_entry['top']),
        'plant_points': (tuple(map(float, bottom_hand)), tuple(map(float, reconstructed_tip)), tuple(map(float, top_at_plant)))
    }

def draw_debug_calib(video_path, res, output_filename, color=(0, 255, 255)):
    if not res: return
    f_idx = res['frame']
    
    centerline = res.get('centerline', None)
    pts = res['pts']
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret: return
    
    if centerline is not None:
        cv2.polylines(frame, [centerline.astype(np.int32)], False, (255, 0, 0), 2) 
        
        hand_pt = res['hand_pt']
        proj_pt = pts[1]
        
        cv2.line(frame, 
                 (int(hand_pt[0]), int(hand_pt[1])), 
                 (int(proj_pt[0]), int(proj_pt[1])), 
                 (0, 255, 0), 1) 
                 
        cv2.circle(frame, (int(pts[0][0]), int(pts[0][1])), 5, (0, 0, 255), -1) 
        cv2.circle(frame, (int(pts[1][0]), int(pts[1][1])), 5, (0, 255, 255), -1) 

    disp_len = res.get('consensus_mean', res['len'])
    
    cv2.putText(frame, f"Len: {disp_len:.1f} px", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imwrite(output_filename, frame)

def draw_debug_bend(video_path, bend_res, output_filename):
    if not bend_res: return
    f_idx = bend_res['max_bend_frame']
    pts = bend_res['points'] # (tip, top)
    tip, top = pts
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret: return
    
    if tip is not None and top is not None:
        # Draw Tip (Blue)
        cv2.circle(frame, (int(tip[0]), int(tip[1])), 8, (255, 0, 0), -1)
        # Draw Top (Red)
        cv2.circle(frame, (int(top[0]), int(top[1])), 8, (0, 0, 255), -1)
        # Draw Chord Line (Yellow)
        cv2.line(frame, 
                 (int(tip[0]), int(tip[1])), 
                 (int(top[0]), int(top[1])), 
                 (0, 255, 255), 2)
                 
        # Label
        txt = f"Max Bend: {bend_res['max_bend']:.1f}%"
        cv2.putText(frame, txt, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    cv2.imwrite(output_filename, frame)
    print(f"[DEBUG] Saved bend debug image to {output_filename}")

def draw_debug_plant(video_path, plant_frame, plant_points, p1_len, output_filename):
    if not plant_points: return
    bottom_hand, reconstructed_tip, top_at_plant = plant_points
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, plant_frame)
    ret, frame = cap.read()
    cap.release()
    if not ret: return
    
    # Draw Hand
    if bottom_hand is not None:
        cv2.circle(frame, (int(bottom_hand[0]), int(bottom_hand[1])), 8, (0, 255, 0), -1) # Green
    
    # Draw Reconstructed Tip
    if reconstructed_tip is not None:
        cv2.circle(frame, (int(reconstructed_tip[0]), int(reconstructed_tip[1])), 8, (255, 0, 0), -1) # Blue
        
    # Draw Top at Plant
    if top_at_plant is not None:
        cv2.circle(frame, (int(top_at_plant[0]), int(top_at_plant[1])), 8, (0, 0, 255), -1) # Red
        
    # Draw line from pole top (sky end) to reconstructed tip — this is the full projected pole
    if top_at_plant is not None and reconstructed_tip is not None:
        cv2.line(frame,
                 (int(top_at_plant[0]), int(top_at_plant[1])),
                 (int(reconstructed_tip[0]), int(reconstructed_tip[1])),
                 (0, 255, 255), 2) # Yellow
                 
    txt = f"Projected Tip (L={p1_len:.1f}px)"
    cv2.putText(frame, txt, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    cv2.imwrite(output_filename, frame)
    print(f"[DEBUG] Saved plant debug image to {output_filename}")
