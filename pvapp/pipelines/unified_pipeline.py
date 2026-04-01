import cv2
import os
import numpy as np
import subprocess
import mediapipe as mp
import logging

logger = logging.getLogger("pv.pipeline")

from pvapp.pipelines.pose_pipeline import extract_pose_data
from pvapp.pipelines.pole_pipeline import extract_pole_data
from pvapp.core.analysis import process_pose_data, compute_hip_drop
from pvapp.core.gait_analysis import calculate_cadence, compute_height_scale_factor, compute_max_hip_height, compute_approach_velocity
from pvapp.utils.cv_utils import draw_outlined_text, draw_simple_skeleton
from pvapp.core.pole_length import analyze_pole_segments, analyze_pole_segment_single_frame, draw_debug_calib, analyze_pole_bend, draw_debug_bend, draw_debug_plant

# Initialize MediaPipe Pose for drawing constants
mp_pose = mp.solutions.pose

def run_unified_pipeline(
    video_path,
    output_path=None,
    pole_model_path="pole_detector_v3.pt",
    pole_conf=0.25,
    enable_pose=True,
    enable_hip=False, # New Toggle
    enable_step=True,
    enable_max_hip_height=True,
    enable_pole=True,
    start_frame=None,
    plant_frame=None,
    end_frame=None,
    athlete_height_m=1.70, # Default to 1.70m
    progress_callback=None,
    pole_length_m=None, # New Optional Arg for Calibration
    skip_pole_metrics=False, # Pass 1: detect pole masks but skip length/bend calculations
    manual_pole_frames=None, # Pass 2: dict with phase1, phase2, plant, max_bend frame indices
    precomputed_pose=None, # Pre-extracted pose results (avoids re-running detection)
    precomputed_pole=None, # Pre-extracted pole results (avoids re-running detection)
    debug_dir=None, # Directory for debug images (if None, uses ./debug_output)
):
    """
    Unified pipeline that:
    1. Extracts Raw Data (Pose, Pole) - Pipelines
    2. Analyzes Data (Hip Drop, Gait, etc.) - Analysis Modules
    3. Renders Output Video - Visualization
    
    Args:
        progress_callback: function(float, str) -> None
    """
    
    if output_path is None:
        base, ext = os.path.splitext(video_path)
        output_path = f"{base}_analyzed.mp4"

    # --- Detect best available compute device ---
    try:
        import torch
        if torch.cuda.is_available():
            compute_device = "cuda"
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)} — using CUDA for YOLO inference.")
        else:
            compute_device = "cpu"
            logger.info("No CUDA GPU detected — using CPU for YOLO inference.")
    except Exception:
        compute_device = "cpu"
        logger.info("torch not available for device check — defaulting to CPU.")

    # --- Step 1: Data Extraction ---
    pose_results = precomputed_pose if precomputed_pose is not None else []
    pole_results = precomputed_pole if precomputed_pole is not None else []

    # 1.1 Pose Extraction (skip if precomputed)
    if enable_pose and precomputed_pose is None:
        if progress_callback:
            progress_callback(0.0, "Extracting Pose Data...")

        def _pose_cb(p, msg):
            if progress_callback:
                progress_callback(p * 0.3, msg) # 0-30%

        pose_results = extract_pose_data(
            video_path,
            start_frame=start_frame,
            end_frame=end_frame,
            progress_callback=_pose_cb,
            device=compute_device
        )

    # 1.2 Pole Extraction (skip if precomputed)
    if enable_pole and precomputed_pole is None:
        if progress_callback:
            progress_callback(0.3, "Extracting Pole Data...")

        def _pole_cb(p, msg):
            if progress_callback:
                progress_callback(0.3 + p * 0.3, msg) # 30-60%

        pole_results = extract_pole_data(
            video_path,
            model_path=pole_model_path,
            conf=pole_conf,
            start_frame=start_frame,
            end_frame=end_frame,
            progress_callback=_pole_cb,
            device=compute_device
        )

    # --- Step 2: Analysis ---
    if progress_callback:
        progress_callback(0.6, "Analyzing Data...")

    # Get video metadata first
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release() # Close for now
    
    # Hip Analysis Config
    hip_droop_pct = None
    hip_droop_trend_pct = None
    hip_points_normal = []
    hip_points_worst = []
    worst_frame_set = set()
    analysis_start_idx = 0
    analysis_end_idx = 0
    
    # Gait Analysis Config
    # stride_len_val removed — heuristic step detection no longer used
    cadence_val = None
    foot_strikes = []
    stride_data_list = [] # New list for plotting
    
    hip_y_arr = None
    pose_landmarks_list = None
    
    # Calibration Factor (Meters / Pixel)
    scale_factor = None
    height_scale_m_per_px = None
    height_calib_info = None
    p1_len_px = 0
    p2_len_px = 0
    l_ankles = [] # Now stores (x, y) tuples
    r_ankles = [] # Now stores (x, y) tuples
    max_bend = None
    bend_series = []
    bend_data = None
    max_hip_height_data = None
    if debug_dir is None:
        debug_dir = os.path.join(os.getcwd(), "debug_output")
    
    if enable_pose and pose_results:
        # Process raw pose data into hip metrics and body height time series
        hip_y_arr, body_h_arr, pose_landmarks_list, _ = process_pose_data(pose_results, (height, width))
        
        # --- HEIGHT-BASED CALIBRATION ---
        height_scale_m_per_px, height_calib_info = compute_height_scale_factor(
            body_h_arr, pose_landmarks_list,
            frame_height=height, frame_width=width,
            athlete_height_m=athlete_height_m,
            plant_frame=plant_frame,
            search_window=8
        )
        if height_calib_info:
            logger.debug(f"Height Calibration: frame={height_calib_info['frame']}, "
                  f"body_height_px={height_calib_info['body_height_px']:.1f}, "
                  f"scale={height_scale_m_per_px:.6f} m/px")
        
        # --- MAX HIP HEIGHT ---
        if enable_max_hip_height and height_scale_m_per_px is not None and plant_frame is not None and end_frame is not None:
            max_hip_height_data = compute_max_hip_height(
                pose_landmarks_list, frame_height=height, frame_width=width,
                plant_frame=plant_frame, end_frame=end_frame,
                scale_m_per_px=height_scale_m_per_px,
                ground_search_window=5
            )
            if max_hip_height_data:
                logger.debug(f"Max Hip Height: {max_hip_height_data['height_m']:.2f}m "
                      f"(frame {max_hip_height_data['peak_frame']})")
        
        # --- HIP ANALYSIS ---
        if enable_hip:
            # Compute Drop Metrics
            (
                hip_droop_pct,
                hip_droop_trend_pct,
                n_valid,
                worst_droop_frames,
                analysis_start_idx,
                analysis_end_idx,
            ) = compute_hip_drop(
                hip_y_arr,
                body_h_arr,
                first_step_frame=start_frame,
                last_step_frame=plant_frame,
            )
            if worst_droop_frames:
                worst_frame_set = set(worst_droop_frames)
            
        # --- GAIT ANALYSIS ---
        if enable_step:
            # Restrict window to Start -> Plant + Buffer to avoid post-plant artifacts
            # Reduced buffer from 0.17s to 0.05s to prevent phantom steps after takeoff
            buffer_frames = int(0.05 * fps)
            step_window_end = min(len(pose_landmarks_list), plant_frame + buffer_frames) if plant_frame else len(pose_landmarks_list)
            step_window_start = start_frame if start_frame is not None else 0

            # --- ML-BASED STEP DETECTION (only method) ---
            import traceback
            try:
                logger.info("ML step detection: fps=%.1f, frame_window=[%d, %d) (%d frames)",
                            fps, step_window_start, step_window_end,
                            step_window_end - step_window_start)

                from step_detection.inference import load_model as load_step_model, predict_steps, clean_predictions

                ml_pipeline, ml_meta = load_step_model()
                feature_cols = ml_meta["feature_columns"]
                logger.info("ML step model loaded: %s (%d features)",
                            ml_meta.get("model_name", "unknown"), len(feature_cols))

                ml_predictions = predict_steps(
                    pose_results, feature_cols, ml_pipeline,
                    start_frame=step_window_start,
                    end_frame=step_window_end - 1,
                )
                logger.info("ML raw predictions: %d frames, %d left-contact, %d right-contact",
                            len(ml_predictions),
                            sum(1 for p in ml_predictions if p['left_contact']),
                            sum(1 for p in ml_predictions if p['right_contact']))

                # Log the raw contact pattern for debugging
                pattern = ""
                for p in ml_predictions:
                    if p['left_contact'] and p['right_contact']:
                        pattern += "B"
                    elif p['left_contact']:
                        pattern += "L"
                    elif p['right_contact']:
                        pattern += "R"
                    else:
                        pattern += "."
                logger.info("ML raw contact pattern: %s", pattern)

                ml_steps = clean_predictions(ml_predictions, fps=fps)
                logger.info("ML clean steps: %d steps after cleaning (min_step_frames=%d at %.0ffps)",
                            len(ml_steps), max(2, round(fps * 0.03)), fps)

                # Convert ML steps -> foot_strikes format for downstream compatibility
                foot_strikes = []
                for step in ml_steps:
                    mid_frame = (step['start_frame'] + step['end_frame']) // 2
                    pr = pose_results[mid_frame] if mid_frame < len(pose_results) else None
                    if pr and pr.pose_landmarks:
                        ankle_idx = (mp_pose.PoseLandmark.LEFT_ANKLE.value
                                     if step['side'] == 'left'
                                     else mp_pose.PoseLandmark.RIGHT_ANKLE.value)
                        lm = pr.pose_landmarks.landmark[ankle_idx]
                        pt = (lm.x, lm.y)
                    else:
                        pt = (0.5, 0.9)
                    foot_strikes.append({
                        'frame': mid_frame,
                        'side': step['side'],
                        'pt': pt,
                        'confidence': 1.0,
                    })

                # Build ankle Y arrays for visualization
                l_ankles_y = []
                r_ankles_y = []
                for i in range(step_window_start, step_window_end):
                    pr = pose_results[i] if i < len(pose_results) else None
                    if pr and pr.pose_landmarks:
                        l_ankles_y.append(pr.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].y)
                        r_ankles_y.append(pr.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y)
                    else:
                        l_ankles_y.append(None)
                        r_ankles_y.append(None)

                for i, strike in enumerate(foot_strikes):
                    logger.info("  step %d: frame=%d, side=%s, pt=(%.3f, %.3f)",
                                i + 1, strike['frame'], strike['side'],
                                strike['pt'][0], strike['pt'][1])
                logger.info("ML step detection found %d steps total", len(foot_strikes))

            except Exception as e:
                logger.error("ML step detection failed: %s\n%s", e, traceback.format_exc())
                foot_strikes = []
                l_ankles_y = []
                r_ankles_y = []

            # Cadence (Steps per Minute)
            from pvapp.core.gait_analysis import calculate_pixel_stride_and_convert
            c_start = start_frame if start_frame is not None else 0
            c_end = plant_frame if plant_frame is not None else (end_frame if end_frame else len(pose_landmarks_list))

            valid_strikes = [s for s in foot_strikes if c_start <= s['frame'] < c_end]

            duration_min = max(0, c_end - c_start) / fps / 60.0
            cadence_val = calculate_cadence(valid_strikes, duration_min) if foot_strikes else None

    # --- Step 2.4: Approach Velocity ---
    velocity_data = []
    if enable_pose and pose_landmarks_list and start_frame is not None and plant_frame is not None:
        velocity_data = compute_approach_velocity(
            pose_landmarks_list, start_frame, plant_frame,
            fps, width, height,
            scale_m_per_px=height_scale_m_per_px,
        )
        if velocity_data:
            peak_v = max(velocity_data, key=lambda x: x['velocity_m_s'])
            logger.debug(f"Peak approach velocity: {peak_v['velocity_m_s']:.2f} m/s at frame {peak_v['frame']}")

    # --- Step 2.5: Pole Length Calibration ---
    if enable_pole and not skip_pole_metrics and enable_pose and pose_results and pole_results and start_frame is not None and plant_frame is not None:
        if progress_callback:
            progress_callback(0.65, "Calibrating Pole Length and Stride...")

        logger.debug(f"Starting Pole Calibration. Start: {start_frame}, Plant: {plant_frame}")
        logger.debug(f"Pose Data Count: {len(pose_results)}, Pole Data Count: {len(pole_results)}")

        p1 = None
        p2 = None

        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)

        try:
            if manual_pole_frames:
                # Manual mode: single-frame analysis for Phase 1 and Phase 2
                logger.debug(f"Manual pole frames: {manual_pole_frames}")
                p1 = analyze_pole_segment_single_frame(
                    pole_results, pose_results, video_path,
                    manual_pole_frames['phase1'], width, height, phase="phase1"
                )
                p2 = analyze_pole_segment_single_frame(
                    pole_results, pose_results, video_path,
                    manual_pole_frames['phase2'], width, height, phase="phase2"
                )
            else:
                # Automatic mode: multi-frame consensus
                p1, p2 = analyze_pole_segments(
                    pole_results, pose_results, video_path,
                    start_frame, plant_frame
                )

            logger.debug(f"Calibration Results: P1={p1 is not None}, P2={p2 is not None}")

            p1_len_px = 0
            p2_len_px = 0
            base_name = os.path.splitext(os.path.basename(video_path))[0]

            if p1:
                out_p1 = os.path.join(debug_dir, f"{base_name}_debug_tip_hand.jpg")
                draw_debug_calib(video_path, p1, out_p1, color=(0, 255, 0))
                pt1, pt2 = p1['pts']
                p1_len_px = np.hypot(pt1[0]-pt2[0], pt1[1]-pt2[1])

            if p2:
                out_p2 = os.path.join(debug_dir, f"{base_name}_debug_top_hand.jpg")
                draw_debug_calib(video_path, p2, out_p2, color=(255, 0, 255))
                pt1, pt2 = p2['pts']
                p2_len_px = np.hypot(pt1[0]-pt2[0], pt1[1]-pt2[1])

            total_pole_px = p1_len_px + p2_len_px

        except Exception as e:
            import traceback
            logger.error(f"Calibration failed: {type(e).__name__}: {e}")
            traceback.print_exc()

    # --- Step 2.3: Pole Bend Analysis ---
    max_bend = None
    bend_series = []
    if enable_pole and not skip_pole_metrics and p1 and p2 and plant_frame:
        try:
            p1_len_px = np.hypot(p1['pts'][0][0]-p1['pts'][1][0], p1['pts'][0][1]-p1['pts'][1][1])
            p2_len_px = np.hypot(p2['pts'][0][0]-p2['pts'][1][0], p2['pts'][0][1]-p2['pts'][1][1])

            p1_l = p1.get('consensus_mean', p1_len_px)
            p2_l = p2.get('consensus_mean', p2_len_px)

            # Use manual plant frame for tip reconstruction if provided
            bend_plant = manual_pole_frames['plant'] if manual_pole_frames else plant_frame
            bend_frame_override = manual_pole_frames.get('max_bend') if manual_pole_frames else None

            max_bend_res = analyze_pole_bend(
                pole_results, pose_results,
                bend_plant,
                p1_l, p2_l,
                width, height,
                end_frame,
                manual_bend_range=bend_frame_override,
                fps=fps
            )
            if max_bend_res:
                bend_data = max_bend_res
                max_bend = max_bend_res['max_bend']
                bend_series = max_bend_res['bend_series']

                vid_base = os.path.splitext(os.path.basename(video_path))[0]
                bend_img_path = os.path.join(debug_dir, f"{vid_base}_debug_bend.jpg")
                draw_debug_bend(video_path, max_bend_res, bend_img_path)

                if 'plant_points' in max_bend_res:
                    plant_img_path = os.path.join(debug_dir, f"{vid_base}_debug_plant.jpg")
                    draw_debug_plant(video_path, bend_plant, max_bend_res['plant_points'], p1_l, plant_img_path)

                logger.debug(f"Max Bend: {max_bend:.2f}%")
            else:
                logger.debug(f"Bend Analysis Failed")
        except Exception as e:
            logger.error(f"Bend Analysis failed: {e}")

    # Re-Run Gait Analysis with Pixel Logic (Now that we have foot strikes)
    if enable_step and foot_strikes:
        stride_data_list = calculate_pixel_stride_and_convert(foot_strikes, scale_factor=None)
        
        # Enrich stride data with real-world units using height-based calibration
        if height_scale_m_per_px is not None:
            for entry in stride_data_list:
                stride_px = entry['stride_norm'] * width
                stride_m = stride_px * height_scale_m_per_px
                entry['stride_cm'] = stride_m * 100.0
                entry['stride_in'] = stride_m * 39.3701


    # --- Step 3: Render ---
    if progress_callback:
        progress_callback(0.7, "Rendering Video...")
        
    if output_path.lower().endswith('.mov') or output_path.lower().endswith('.avi'):
        output_path = output_path.rsplit('.', 1)[0] + '.mp4'
        logger.debug(f"Changed output container to .mp4 ({output_path})")

    cap = cv2.VideoCapture(video_path)
    logger.debug(f"RENDER Source Video Opened: {cap.isOpened()}, Frames: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}, Dim: {width}x{height}, FPS: {fps}")
    # Ensure width/height are integers
    width = int(width)
    height = int(height)
    
    # Cap FPS to max 120 to prevent OpenCV/MPEG4 timebase errors on high-FPS iPhone slow-mo video.
    # The timebase max denominator is 65535, so 1000/239493 fails.
    fps_out = min(fps, 120.0)
    logger.debug(f"RENDER output FPS capped to: {fps_out:.2f} (from raw {fps:.2f})")
    
    # Try native browser-supported H.264
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(output_path, fourcc, fps_out, (width, height))
    logger.debug(f"RENDER avc1 Writer Opened: {out.isOpened()} at {output_path}")
    
    if not out.isOpened():
        logger.info("avc1 failed, falling back to mp4v")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps_out, (width, height))
        logger.debug(f"RENDER mp4v Writer Opened: {out.isOpened()} at {output_path}")
        
    if not out.isOpened():
        logger.info("mp4v failed, falling back to avi container with XVID")
        output_path = output_path.rsplit('.', 1)[0] + '.avi'
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(output_path, fourcc, fps_out, (width, height))
        logger.debug(f"RENDER avi/XVID Writer Opened: {out.isOpened()} at {output_path}")
    
    # Prepare HUD text for corners
    # Top-right: cadence, stride (dynamic), hip droop/trend
    hud_top_right_static = []
    if enable_pose:
        if cadence_val:
            hud_top_right_static.append(f"Cadence: {cadence_val:.0f} spm")
        # Stride line inserted dynamically per-frame after cadence
        if hip_droop_pct is not None:
            hud_top_right_static.append(f"Hip droop: {hip_droop_pct:+.1f}%")
            hud_top_right_static.append(f"Hip trend: {hip_droop_trend_pct:+.1f}%")

    # Bottom-right: window info (static)
    hud_bottom_right = []
    if start_frame is not None:
        hud_bottom_right.append(f"Window: {start_frame}-{plant_frame}-{end_frame}")

    # Top-left: predicted clear + max bend (conditional, added per-frame)
    # These are built per-frame since they only appear after certain frames

    # Precompute predicted clear text
    predicted_clear_text = None
    predicted_clear_frame = None
    if max_hip_height_data and 'predicted_clear_m' in max_hip_height_data:
        pc_m = max_hip_height_data['predicted_clear_m']
        pc_in = max_hip_height_data['predicted_clear_in']
        pc_ft = int(pc_in // 12)
        pc_in_r = pc_in % 12
        predicted_clear_text = f"Pred Clear: {pc_ft}'{pc_in_r:.0f}\" ({pc_m:.2f}m)"
        predicted_clear_frame = max_hip_height_data.get('pc_frame', max_hip_height_data['peak_frame'])

    # Precompute max bend annotation data (drawn at pole, not HUD)
    max_bend_annotation = None
    if bend_data and bend_data.get('points'):
        poly_max = bend_data.get('poly_max_bend')
        mb_val = poly_max if poly_max is not None else bend_data.get('max_bend')
        tip_pt, top_pt = bend_data['points']
        if mb_val is not None and tip_pt is not None and top_pt is not None:
            max_bend_annotation = {
                'value': mb_val,
                'tip': (int(tip_pt[0]), int(tip_pt[1])),
                'top': (int(top_pt[0]), int(top_pt[1])),
                'frame': bend_data.get('max_bend_frame', plant_frame),
                'label': f"Max Bend: {mb_val:.1f}%"
            }

    # Precompute velocity HUD text (bottom-left)
    velocity_hud_lines = []
    if velocity_data:
        peak_v = max(velocity_data, key=lambda x: x['velocity_m_s'])
        avg_v_ms = sum(d['velocity_m_s'] for d in velocity_data) / len(velocity_data)

        # Takeoff velocity: average of the 0.25s before plant
        takeoff_window = int(0.25 * fps) if fps else 8
        if plant_frame is not None:
            takeoff_entries = [d for d in velocity_data if plant_frame - takeoff_window <= d['frame'] <= plant_frame]
        else:
            takeoff_entries = []
        takeoff_v_ms = (sum(d['velocity_m_s'] for d in takeoff_entries) / len(takeoff_entries)) if takeoff_entries else avg_v_ms

        if height_scale_m_per_px:
            peak_mph = peak_v['velocity_mph']
            avg_mph = avg_v_ms * 2.23694
            takeoff_mph = takeoff_v_ms * 2.23694
            velocity_hud_lines.append(f"Peak: {peak_v['velocity_m_s']:.1f} m/s ({peak_mph:.1f} mph)")
            velocity_hud_lines.append(f"Avg: {avg_v_ms:.1f} m/s ({avg_mph:.1f} mph)")
            velocity_hud_lines.append(f"Takeoff: {takeoff_v_ms:.1f} m/s ({takeoff_mph:.1f} mph)")
        else:
            velocity_hud_lines.append(f"Peak: {peak_v['velocity_m_s']:.0f} px/s")
            velocity_hud_lines.append(f"Avg: {avg_v_ms:.0f} px/s")
            velocity_hud_lines.append(f"Takeoff: {takeoff_v_ms:.0f} px/s")

    # Visual Window Config
    visual_start = start_frame if start_frame is not None else 0
    visual_end = end_frame if end_frame is not None else total_frames
    
    # Transform Foot Strikes for drawing
    strike_draw_list = []
    for s in foot_strikes:
        pt_norm = s['pt']
        pt_px = (int(pt_norm[0] * width), int(pt_norm[1] * height))
        
        is_interp = s.get('interpolated', False)
        # Distinct color for interpolated
        color = (0, 0, 255) if is_interp else (0, 165, 255) # Red for Interp, Orange for Normal
        
        strike_draw_list.append({
            'frame': s['frame'],
            'pt': pt_px,
            'color': color,
            'interpolated': is_interp
        })

    frame_idx = 0
    stride_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if progress_callback and frame_idx % 20 == 0:
            p = 0.7 + 0.3 * (frame_idx / max(total_frames, 1))
            progress_callback(p, f"Rendering: {int(p*100)}%")

        # 3.1 Draw Pole (Bottom Layer)
        if enable_pole and frame_idx < len(pole_results):
            p_res = pole_results[frame_idx]
            if p_res and p_res.masks:
                for poly in p_res.masks.xy:
                        if len(poly) > 0:
                            pts = np.array(poly, np.int32).reshape((-1, 1, 2))
                            overlay = frame.copy()
                            cv2.fillPoly(overlay, [pts], (0, 255, 255))
                            cv2.polylines(overlay, [pts], True, (0, 200, 200), 2)
                            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

        # 3.2 Draw Skeleton & Hip Path & Foot Strikes
        if enable_pose and frame_idx < len(pose_landmarks_list):
            landmarks = pose_landmarks_list[frame_idx]
            
            # Check visual window for drawing skeleton
            in_visual_window = (frame_idx >= visual_start and frame_idx <= visual_end)
            
            if landmarks and in_visual_window:
                l_ankle = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
                r_ankle = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
                l_ankles.append((l_ankle.x * width, l_ankle.y * height))
                r_ankles.append((r_ankle.x * width, r_ankle.y * height))
                
                draw_simple_skeleton(frame, landmarks, mp_pose.POSE_CONNECTIONS,
                                     point_color=(252, 0, 219), line_color=(0, 255, 0),
                                     visibility_threshold=0.3) # Enforce GaitKeeper strictness
                
                # Get Hip Point for path
                lm = landmarks.landmark
                lhip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
                rhip = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]
                hx = int((lhip.x + rhip.x) * 0.5 * width)
                hy = int((lhip.y + rhip.y) * 0.5 * height)
                
                # Add to path arrays if in analysis window
                in_analysis_window = (frame_idx >= analysis_start_idx and frame_idx < analysis_end_idx)
                if in_analysis_window:
                     if frame_idx in worst_frame_set:
                         hip_points_worst.append((hx, hy))
                     else:
                         hip_points_normal.append((hx, hy))

        # Draw Persistent Hip Path
        for pt in hip_points_normal:
            cv2.circle(frame, pt, 5, (255, 255, 0), -1)
        for pt in hip_points_worst:
            cv2.circle(frame, pt, 6, (0, 0, 255), -1)
            
        # Draw Foot Strikes (Persistent)

        # Draw height calibration visual indicator on the calibration frame
        if height_calib_info and frame_idx >= height_calib_info['frame']:
            top_pt = height_calib_info['top_px']  # eye position
            ankle_pt = height_calib_info['ankle_px']
            # Extend line above eye by ~5% to represent full height (skull above eyes)
            eye_to_ankle_dist = ankle_pt[1] - top_pt[1]
            extra_above = int(eye_to_ankle_dist * 0.05 / 0.95)  # 5% of full height
            full_top_pt = (top_pt[0], max(0, top_pt[1] - extra_above))
            # Full measurement line (top of head estimate to ankle)
            cv2.line(frame, full_top_pt, ankle_pt, (0, 255, 255), 3)
            # Tick marks at top (estimated head top), eye level, and ankle
            tick_w = 15
            cv2.line(frame, (full_top_pt[0] - tick_w, full_top_pt[1]), (full_top_pt[0] + tick_w, full_top_pt[1]), (0, 255, 255), 3)
            tick_s = 8  # smaller tick at eye level
            cv2.line(frame, (top_pt[0] - tick_s, top_pt[1]), (top_pt[0] + tick_s, top_pt[1]), (0, 255, 255), 2)
            cv2.line(frame, (ankle_pt[0] - tick_w, ankle_pt[1]), (ankle_pt[0] + tick_w, ankle_pt[1]), (0, 255, 255), 3)
            # Label
            h_total_in = athlete_height_m * 39.3701
            h_ft = int(h_total_in // 12)
            h_in_r = h_total_in % 12
            label = f"Height Ref: {h_ft}'{h_in_r:.0f}\" ({athlete_height_m:.2f}m)"
            label_x = top_pt[0] + 20
            label_y = (full_top_pt[1] + ankle_pt[1]) // 2
            draw_outlined_text(frame, label, (label_x, label_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        for s in strike_draw_list:
            if s['frame'] <= frame_idx:
                # Persistent dot
                cv2.circle(frame, s['pt'], 6, s['color'], -1)
                
                # If current frame matches strike, ping!
                if s['frame'] == frame_idx:
                     cv2.circle(frame, s['pt'], 14, s['color'], 2)
        
        # Draw Max Hip Height overlay (persistent from peak frame onward)
        if enable_max_hip_height and max_hip_height_data and frame_idx >= max_hip_height_data['peak_frame']:
            hip_pt = max_hip_height_data['peak_hip_px']
            cv2.circle(frame, hip_pt, 8, (255, 0, 255), -1)  # Magenta dot
            h_m = max_hip_height_data['height_m']
            h_in = max_hip_height_data['height_in']
            h_ft = int(h_in // 12)
            h_in_r = h_in % 12
            hip_label = f"Max Hip: {h_ft}'{h_in_r:.0f}\" ({h_m:.2f}m)"
            draw_outlined_text(frame, hip_label, (hip_pt[0] + 15, hip_pt[1]),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        # Draw Predicted Clear annotation directly on the frame
        if enable_max_hip_height and max_hip_height_data and 'pc_px' in max_hip_height_data and frame_idx >= max_hip_height_data['pc_frame']:
            pc_pt = max_hip_height_data['pc_px']
            cv2.circle(frame, pc_pt, 8, (0, 255, 0), -1)  # Green dot
            if predicted_clear_text:
                draw_outlined_text(frame, predicted_clear_text, (pc_pt[0] + 15, pc_pt[1]),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw Max Bend overlay (persistent from max bend frame onward)
        if max_bend_annotation and frame_idx >= max_bend_annotation['frame']:
            mb_tip = max_bend_annotation['tip']
            mb_top = max_bend_annotation['top']
            # Chord line from projected tip to top of pole (yellow)
            cv2.line(frame, mb_tip, mb_top, (0, 255, 255), 2)
            # Endpoints: blue for projected tip, red for top of pole
            cv2.circle(frame, mb_tip, 6, (255, 0, 0), -1)
            cv2.circle(frame, mb_top, 6, (0, 0, 255), -1)
            # Label at the top point
            draw_outlined_text(frame, max_bend_annotation['label'],
                               (mb_top[0] + 15, mb_top[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Compute running average stride for HUD
        current_stride_text = ""
        strides_so_far = [sd for sd in stride_data_list if sd['frame'] <= frame_idx]

        if strides_so_far:
            cm_vals = [sd['stride_cm'] for sd in strides_so_far if sd.get('stride_cm') is not None]
            in_vals = [sd['stride_in'] for sd in strides_so_far if sd.get('stride_in') is not None]

            if cm_vals and in_vals:
                avg_cm = sum(cm_vals) / len(cm_vals)
                avg_in = sum(in_vals) / len(in_vals)
                current_stride_text = f"Avg Stride: {avg_cm:.0f} cm ({avg_in:.1f} in)"
            else:
                px_vals = [sd.get('stride_norm', 0) * width for sd in strides_so_far]
                avg_px = sum(px_vals) / len(px_vals)
                current_stride_text = f"Avg Stride: {avg_px:.0f} px"

        # --- Draw HUD in corners ---
        hud_font = cv2.FONT_HERSHEY_SIMPLEX
        hud_scale = 1.0
        hud_thick = 2
        hud_spacing = 40

        # TOP-LEFT: Predicted clear height (conditional)
        hud_top_left = []
        if predicted_clear_text and frame_idx >= predicted_clear_frame:
            hud_top_left.append(predicted_clear_text)

        for i, txt in enumerate(hud_top_left):
            draw_outlined_text(frame, txt, (10, 60 + hud_spacing * i),
                               hud_font, hud_scale, (0, 255, 0), hud_thick)

        # TOP-RIGHT: Cadence, stride, hip droop/trend
        hud_top_right = list(hud_top_right_static)
        # Insert stride after cadence (index 1 if cadence exists, else index 0)
        if current_stride_text:
            insert_idx = 1 if (cadence_val and len(hud_top_right) > 0) else 0
            hud_top_right.insert(insert_idx, current_stride_text)

        for i, txt in enumerate(hud_top_right):
            (tw, _), _ = cv2.getTextSize(txt, hud_font, hud_scale, hud_thick)
            draw_outlined_text(frame, txt, (width - tw - 10, 60 + hud_spacing * i),
                               hud_font, hud_scale, (0, 255, 0), hud_thick)

        # BOTTOM-RIGHT: Window info
        for i, txt in enumerate(hud_bottom_right):
            (tw, _), _ = cv2.getTextSize(txt, hud_font, hud_scale, hud_thick)
            y = height - 20 - hud_spacing * (len(hud_bottom_right) - 1 - i)
            draw_outlined_text(frame, txt, (width - tw - 10, y),
                               hud_font, hud_scale, (0, 255, 0), hud_thick)

        # BOTTOM-LEFT: Velocity stats (shown once analysis window is reached)
        if velocity_hud_lines and frame_idx >= (start_frame or 0):
            for i, txt in enumerate(velocity_hud_lines):
                y = height - 20 - hud_spacing * (len(velocity_hud_lines) - 1 - i)
                draw_outlined_text(frame, txt, (10, y),
                                   hud_font, hud_scale, (0, 255, 0), hud_thick)

        out.write(frame)
        frame_idx += 1
        
    cap.release()
    out.release()
    
    # Re-encode to H.264 for browser/Streamlit playback
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        h264_path = os.path.splitext(output_path)[0] + "_h264.mp4"
        cmd = [
            ffmpeg_exe, "-y",
            "-i", output_path,
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            h264_path,
        ]
        # Use stdout=subprocess.PIPE, stderr=subprocess.PIPE to capture errors but not print them unless it fails
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0 and os.path.exists(h264_path):
            # Replace original with H.264 version so job_runner finds output.mp4
            os.remove(output_path)
            os.rename(h264_path, output_path)
            logger.debug(f"Re-encoded to H.264: {output_path}")
        else:
            logger.warning(f"ffmpeg re-encode failed with code {result.returncode}, video may not play in browser. Error: {result.stderr.decode('utf-8')[:200]}")
    except Exception as e:
        logger.warning(f"ffmpeg re-encode failed, video may not play in browser: {e}")
    
    # Prepare return values
    calib_px = (p1_len_px, p2_len_px)
    ankles = (l_ankles, r_ankles)
    return output_path, stride_data_list, calib_px, ankles, bend_data, height_scale_m_per_px, max_hip_height_data, pose_results, pole_results, velocity_data, cadence_val
