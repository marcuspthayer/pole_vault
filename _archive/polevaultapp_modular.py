import streamlit as st
import tempfile
import os
import logging
import cv2

# Import pipelines
from pvapp.pipelines.unified_pipeline import run_unified_pipeline
from pvapp.logging_utils import setup_logger

st.set_page_config(page_title="AlphaPeak PV Analysis", layout="wide")

# Logging
LOG_FILE = os.path.join(os.getcwd(), "logs", "pv_streamlit.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
setup_logger("pv", log_file=LOG_FILE, level=logging.DEBUG)
logger = logging.getLogger("pv.streamlit")

# ---------------- Styling ----------------
st.markdown("""
<style>
.gradient-header {
    background: linear-gradient(135deg, #FF3300 30%, #FF8C00 70%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    font-family: sans-serif;
}
.card { background-color: #F0F2F6; padding: 20px; border-radius: 10px; font-family: sans-serif; color: #262730; }
.stButton>button { background-color: #FF3300; color: white; font-weight: bold; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# ---------------- Session State ----------------
if "video_path" not in st.session_state:
    st.session_state.video_path = None
if "results" not in st.session_state:
    st.session_state.results = None
if "manual_pole_phase" not in st.session_state:
    st.session_state.manual_pole_phase = None  # None | "pass1_done"
if "pass1_video_path" not in st.session_state:
    st.session_state.pass1_video_path = None
if "pass1_pose_results" not in st.session_state:
    st.session_state.pass1_pose_results = None
if "pass1_pole_results" not in st.session_state:
    st.session_state.pass1_pole_results = None

st.markdown('<h1 class="gradient-header">AlphaForm: Pole Vault Analysis</h1>', unsafe_allow_html=True)

# ---------------- Sidebar / Configuration ----------------
with st.sidebar:
    try:
        import torch
        if torch.cuda.is_available():
            st.success(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            st.info("GPU: None detected (CPU mode)")
    except Exception:
        st.info("GPU: Unknown")

    st.header("Analysis Checklist")
    
    # Checklist
    use_skeleton = st.checkbox("Skeleton Overlay", value=True, help="Draw basic skeleton on the video")
    use_hip_analysis = st.checkbox("Approach Hip Analysis", value=False, help="Analyze hip sag/droop during the approach run")
    use_step_analysis = st.checkbox("Step Analysis", value=False, help="Analyze cadence, stride, and foot strikes")

    use_ml_steps = False
    if use_step_analysis:
        from pathlib import Path
        _ml_model_available = (
            Path("step_detection/models/best_step_model.joblib").exists()
            and Path("step_detection/models/best_step_model_meta.json").exists()
        )
        if _ml_model_available:
            use_ml_steps = st.checkbox(
                "Use ML Step Detection", value=True,
                help="Use the trained step detection model. Disable to fall back to heuristic detection."
            )

    use_max_hip_height = st.checkbox("Max Hip Height", value=False, help="Measure peak hip height during bar clearance")
    use_pole_analysis = st.checkbox("Pole Analysis", value=False, help="Detect and track the pole")

    use_manual_pole_frames = False
    if use_pole_analysis:
        use_manual_pole_frames = st.checkbox(
            "Manual Pole Frame Selection", value=False,
            help="Two-pass mode: first pass renders pole masks, then you pick specific frames with good masks for Phase 1, Phase 2, plant, and max bend."
        )

    st.divider()
    st.header("Settings")
    
    # Athlete Height
    st.markdown("### Athlete Height")
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        h_ft = st.number_input("Feet", min_value=3, max_value=8, value=5)
    with col_h2:
        h_in = st.number_input("Inches", min_value=0, max_value=11, value=7)
        
    athlete_height_m = (h_ft * 12 + h_in) * 0.0254
    st.caption(f"height: {athlete_height_m:.2f} m")

    # Pole Length (Optional Calibration)
    st.markdown("### Pole Length (for Stride Calibration)")
    use_pole_calib = st.checkbox("Enable Pole-Based Calibration", value=False)
    pole_length_m = None
    
    if use_pole_calib:
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            p_ft = st.number_input("Pole Feet", min_value=10, max_value=20, value=14, key="p_ft")
        with col_p2:
            p_in = st.number_input("Pole Inches", min_value=0, max_value=11, value=0, key="p_in")
        pole_length_m = (p_ft * 12 + p_in) * 0.0254
        st.caption(f"Pole Length: {pole_length_m:.2f} m")
    
    # Model config
    pole_model_path = st.text_input("Pole Model Path", value="pole_detect_v3.pt")
    pole_conf = st.slider("Pole Confidence", 0.1, 0.9, 0.25, 0.05)
    
    step_min_lift = 0.0150
    step_min_dist = 10
    if not use_ml_steps:
        with st.expander("Advanced Settings"):
            st.markdown("**Step Detection Sensitivity**")
            step_min_lift = st.slider("Min Lift (Sensitivity)", 0.0001, 0.0500, 0.0150, 0.0001, format="%.4f", help="Lower = More Sensitive. Minimum vertical rise to count as a step.")
            step_min_dist = st.slider("Min Step Distance (Frames)", 3, 30, 10, 1, help="Minimum frames between steps to prevent double counting.")
    
    
    # "Run Analysis" Button
    # ... (moved to main area)

# ---------------- Main Area ----------------

uploaded = st.file_uploader("Upload Vault Video (MP4/MOV)", type=["mp4", "mov", "avi"])

if uploaded:
    # Save to temp
    # Save to temp
    # Only write if we haven't already, or if the file changed
    if st.session_state.video_path is None or "last_vname" not in st.session_state or st.session_state.last_vname != uploaded.name:
        suffix = os.path.splitext(uploaded.name)[1].lower() or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            st.session_state.video_path = tmp.name
        st.session_state.last_vname = uploaded.name
        # Clear previous analysis results
        for key in ["result_video", "stride_data", "calib_px", "ankles", "bend_data", "scale_m_px", "max_hip_data"]:
            if key in st.session_state:
                del st.session_state[key]
        st.success(f"Loaded new video: {uploaded.name}")

    # Show raw video
    st.subheader("Source Video")
    if st.session_state.video_path.lower().endswith('.mov') or st.session_state.video_path.lower().endswith('.avi'):
        st.info("Note: Your browser may not support playing this video format natively (it may appear greyed out). The analysis will still function normally.")
    st.video(st.session_state.video_path)
    
    # Metadata for frame range
    cap = cv2.VideoCapture(st.session_state.video_path)
    if cap.isOpened():
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # Width? Typo in original? No, it was CAP_PROP_FRAME_COUNT
        # Correction: original line 113 was CAP_PROP_FRAME_COUNT. 
        # Wait, I am replacing lines 58-284? That is huge.
        # Let's be careful.
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        st.caption(f"Total Frames: {total_f}")
        cap.release()

    # ---------------- Frame Selection (Main Page) ----------------
    st.divider()
    st.header("Analysis Configuration")
    st.info("Select the key frames for your jump.")

    # Helper to read frame
    @st.cache_data
    def get_frame_image(p_video_path, frame_idx):
        cap = cv2.VideoCapture(p_video_path)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if ret:
            # Convert BGR to RGB for Streamlit
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

    if st.session_state.video_path:
        cap_temp = cv2.VideoCapture(st.session_state.video_path)
        total_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_temp.release()
        
        # Initialize session state for frames if they don't exist
        if "s_frame" not in st.session_state:
            st.session_state.s_frame = 0
        if "p_frame" not in st.session_state:
            st.session_state.p_frame = int(total_frames * 0.8)
        if "e_frame" not in st.session_state:
            st.session_state.e_frame = total_frames - 1

        def _dec(key, min_val=0):
            st.session_state[key] = max(min_val, st.session_state[key] - 1)

        def _inc(key, max_val):
            st.session_state[key] = min(max_val, st.session_state[key] + 1)

        # Step 1: Start Frame
        st.markdown("**1. Start Frame**")
        col_s1, col_s2, col_s3 = st.columns([1, 4, 1], vertical_alignment="bottom")
        with col_s1:
            st.button("⏪ -1", key="btn_s_minus", on_click=_dec, args=("s_frame",))
        with col_s2:
            st.slider("Start", 0, total_frames-1, key="s_frame", label_visibility="collapsed")
        with col_s3:
            st.button("Fast Fwd +1 ⏩", key="btn_s_plus", on_click=_inc, args=("s_frame", total_frames-1))

        start_frame = st.session_state.s_frame
        prev_img_start = get_frame_image(st.session_state.video_path, start_frame)
        if prev_img_start is not None:
            st.image(prev_img_start, caption=f"Start: {start_frame}")

        # Step 2: Plant Frame
        st.markdown("**2. Plant Frame**")
        col_p1, col_p2, col_p3 = st.columns([1, 4, 1], vertical_alignment="bottom")
        with col_p1:
            st.button("⏪ -1", key="btn_p_minus", on_click=_dec, args=("p_frame",))
        with col_p2:
            st.slider("Plant", 0, total_frames-1, key="p_frame", label_visibility="collapsed")
        with col_p3:
            st.button("Fast Fwd +1 ⏩", key="btn_p_plus", on_click=_inc, args=("p_frame", total_frames-1))

        plant_frame = st.session_state.p_frame
        prev_img_plant = get_frame_image(st.session_state.video_path, plant_frame)
        if prev_img_plant is not None:
            st.image(prev_img_plant, caption=f"Plant: {plant_frame}")

        # Step 3: End Frame
        st.markdown("**3. End Frame**")
        col_e1, col_e2, col_e3 = st.columns([1, 4, 1], vertical_alignment="bottom")
        with col_e1:
            st.button("⏪ -1", key="btn_e_minus", on_click=_dec, args=("e_frame",))
        with col_e2:
            st.slider("End", 0, total_frames-1, key="e_frame", label_visibility="collapsed")
        with col_e3:
            st.button("Fast Fwd +1 ⏩", key="btn_e_plus", on_click=_inc, args=("e_frame", total_frames-1))

        end_frame = st.session_state.e_frame
        prev_img_end = get_frame_image(st.session_state.video_path, end_frame)
        if prev_img_end is not None:
            st.image(prev_img_end, caption=f"End: {end_frame}")
            
        st.caption(f"Analysis Window: {start_frame} to {end_frame}")
    else:
        st.info("Upload video to enable frame selection.")
        start_frame = 0
        plant_frame = 0
        end_frame = 0
    
    # -----------------------------------------------------------

    force_full_video = st.checkbox("Ignore sliders (Process Full Video)", value=False)
    
    st.divider()
    
    def _unpack_and_store_results(results, is_pass1=False):
        """Unpack pipeline results tuple and store in session state."""
        if isinstance(results, tuple):
            out_path = results[0]
            stride_data = results[1]
            calib_px = results[2]
            l_ankles = results[3][0] if len(results) > 3 else []
            r_ankles = results[3][1] if len(results) > 3 else []
            bend_data = results[4] if len(results) > 4 else None
            height_scale = results[5] if len(results) > 5 else None
            max_hip_height_data = results[6] if len(results) > 6 else None
            pose_res = results[7] if len(results) > 7 else None
            pole_res = results[8] if len(results) > 8 else None
            velocity_data = results[9] if len(results) > 9 else []
        else:
            out_path = results
            stride_data, calib_px = [], (0, 0)
            l_ankles, r_ankles = [], []
            bend_data, height_scale, max_hip_height_data = None, None, None
            pose_res, pole_res = None, None
            velocity_data = []

        if is_pass1:
            st.session_state.pass1_video_path = out_path
            st.session_state.pass1_pose_results = pose_res
            st.session_state.pass1_pole_results = pole_res
            st.session_state.manual_pole_phase = "pass1_done"
        else:
            st.session_state.results = out_path
            st.session_state.stride_data = stride_data
            st.session_state.calib_px = calib_px
            st.session_state.pole_len_used = pole_length_m
            st.session_state.height_scale = height_scale
            st.session_state.l_ankles = l_ankles
            st.session_state.r_ankles = r_ankles
            st.session_state.bend_data = bend_data
            st.session_state.max_hip_height = max_hip_height_data
            st.session_state.velocity_data = velocity_data
            # Clear pass1 state
            st.session_state.manual_pole_phase = None
            st.session_state.pass1_video_path = None
            st.session_state.pass1_pose_results = None
            st.session_state.pass1_pole_results = None

    # --- Manual Pole Frame Selection: Pass 2 UI ---
    if use_manual_pole_frames and st.session_state.manual_pole_phase == "pass1_done":
        st.divider()
        st.subheader("Pass 2: Select Pole Frames")
        st.info("Review the pass-1 video below. Select frames where the pole mask is clean and accurate for each measurement phase.")

        if st.session_state.pass1_video_path:
            st.video(st.session_state.pass1_video_path)

        # Initialize manual frame selectors
        for key, default_expr in [
            ("manual_p1_frame", start_frame + 5),
            ("manual_p2_frame", plant_frame),
            ("manual_plant_frame", plant_frame),
            ("manual_bend_start", min(plant_frame + 5, end_frame)),
            ("manual_bend_end", min(plant_frame + 30, end_frame)),
        ]:
            if key not in st.session_state:
                st.session_state[key] = default_expr

        # Use the pass-1 rendered video for previews (shows pole masks + skeletons)
        _preview_vid = st.session_state.pass1_video_path or st.session_state.video_path

        st.markdown("**Phase 1 Frame** (tip-to-bottom-hand, pole should be straight)")
        col1, col2, col3 = st.columns([1, 4, 1], vertical_alignment="bottom")
        with col1:
            st.button("⏪ -1", key="btn_mp1_minus", on_click=_dec, args=("manual_p1_frame",))
        with col2:
            st.slider("Phase 1", 0, total_frames-1, key="manual_p1_frame", label_visibility="collapsed")
        with col3:
            st.button("⏩ +1", key="btn_mp1_plus", on_click=_inc, args=("manual_p1_frame", total_frames-1))
        img_p1 = get_frame_image(_preview_vid, st.session_state.manual_p1_frame)
        if img_p1 is not None:
            st.image(img_p1, caption=f"Phase 1: {st.session_state.manual_p1_frame}")

        st.markdown("**Phase 2 Frame** (top-to-bottom-hand, around plant)")
        col1, col2, col3 = st.columns([1, 4, 1], vertical_alignment="bottom")
        with col1:
            st.button("⏪ -1", key="btn_mp2_minus", on_click=_dec, args=("manual_p2_frame",))
        with col2:
            st.slider("Phase 2", 0, total_frames-1, key="manual_p2_frame", label_visibility="collapsed")
        with col3:
            st.button("⏩ +1", key="btn_mp2_plus", on_click=_inc, args=("manual_p2_frame", total_frames-1))
        img_p2 = get_frame_image(_preview_vid, st.session_state.manual_p2_frame)
        if img_p2 is not None:
            st.image(img_p2, caption=f"Phase 2: {st.session_state.manual_p2_frame}")

        st.markdown("**Plant Frame** (for tip reconstruction, needs solid mask)")
        col1, col2, col3 = st.columns([1, 4, 1], vertical_alignment="bottom")
        with col1:
            st.button("⏪ -1", key="btn_mpl_minus", on_click=_dec, args=("manual_plant_frame",))
        with col2:
            st.slider("Plant Mask", 0, total_frames-1, key="manual_plant_frame", label_visibility="collapsed")
        with col3:
            st.button("⏩ +1", key="btn_mpl_plus", on_click=_inc, args=("manual_plant_frame", total_frames-1))
        img_pl = get_frame_image(_preview_vid, st.session_state.manual_plant_frame)
        if img_pl is not None:
            st.image(img_pl, caption=f"Plant Mask: {st.session_state.manual_plant_frame}")

        st.markdown("**Max Bend Range** (search window for maximum pole bend)")
        st.markdown("_Start of range:_")
        col1, col2, col3 = st.columns([1, 4, 1], vertical_alignment="bottom")
        with col1:
            st.button("⏪ -1", key="btn_mbs_minus", on_click=_dec, args=("manual_bend_start",))
        with col2:
            st.slider("Bend Start", 0, total_frames-1, key="manual_bend_start", label_visibility="collapsed")
        with col3:
            st.button("⏩ +1", key="btn_mbs_plus", on_click=_inc, args=("manual_bend_start", total_frames-1))
        img_mbs = get_frame_image(_preview_vid, st.session_state.manual_bend_start)
        if img_mbs is not None:
            st.image(img_mbs, caption=f"Bend Start: {st.session_state.manual_bend_start}")

        st.markdown("_End of range:_")
        col1, col2, col3 = st.columns([1, 4, 1], vertical_alignment="bottom")
        with col1:
            st.button("⏪ -1", key="btn_mbe_minus", on_click=_dec, args=("manual_bend_end",))
        with col2:
            st.slider("Bend End", 0, total_frames-1, key="manual_bend_end", label_visibility="collapsed")
        with col3:
            st.button("⏩ +1", key="btn_mbe_plus", on_click=_inc, args=("manual_bend_end", total_frames-1))
        img_mbe = get_frame_image(_preview_vid, st.session_state.manual_bend_end)
        if img_mbe is not None:
            st.image(img_mbe, caption=f"Bend End: {st.session_state.manual_bend_end}")

        if st.button("🎯 Finalize Analysis", type="primary"):
            try:
                progress_bar = st.progress(0.0)
                status_text = st.empty()

                def _prog_cb(pct, msg=""):
                    progress_bar.progress(min(pct, 1.0))
                    status_text.text(msg)

                need_pose = use_skeleton or use_hip_analysis or use_step_analysis or use_max_hip_height
                s_f = start_frame if not force_full_video else 0
                p_f = plant_frame if not force_full_video else total_f
                e_f = end_frame if not force_full_video else total_f

                manual_frames = {
                    'phase1': st.session_state.manual_p1_frame,
                    'phase2': st.session_state.manual_p2_frame,
                    'plant': st.session_state.manual_plant_frame,
                    'max_bend': (st.session_state.manual_bend_start, st.session_state.manual_bend_end),
                }

                results = run_unified_pipeline(
                    st.session_state.video_path,
                    pole_model_path=pole_model_path,
                    pole_conf=pole_conf,
                    enable_pose=need_pose,
                    enable_hip=use_hip_analysis,
                    enable_step=use_step_analysis,
                    enable_max_hip_height=use_max_hip_height,
                    enable_pole=use_pole_analysis,
                    start_frame=s_f, plant_frame=p_f, end_frame=e_f,
                    athlete_height_m=athlete_height_m,
                    progress_callback=_prog_cb,
                    step_min_lift=step_min_lift,
                    step_min_dist=step_min_dist,
                    pole_length_m=pole_length_m,
                    enable_ml_steps=use_ml_steps,
                    skip_pole_metrics=False,
                    manual_pole_frames=manual_frames,
                    precomputed_pose=st.session_state.pass1_pose_results,
                    precomputed_pole=st.session_state.pass1_pole_results,
                )

                _unpack_and_store_results(results, is_pass1=False)
                status_text.text("Final Analysis Complete!")
                st.rerun()

            except Exception as e:
                st.error(f"Final analysis failed: {e}")
                logger.exception("Pass 2 pipeline failed")

    # "Run Analysis" Button
    elif st.button("🚀 Run Analysis", type="primary"):

        # Determine if we need pose at all
        need_pose = use_skeleton or use_hip_analysis or use_step_analysis or use_max_hip_height

        if not (need_pose or use_pole_analysis):
            st.warning("Please select at least one analysis/overlay type from the sidebar.")
        else:
            try:
                progress_bar = st.progress(0.0)
                status_text = st.empty()

                def _prog_cb(pct, msg=""):
                    progress_bar.progress(min(pct, 1.0))
                    status_text.text(msg)

                # Determine window
                s_f = start_frame if not force_full_video else 0
                p_f = plant_frame if not force_full_video else total_f
                e_f = end_frame if not force_full_video else total_f

                # If manual pole mode, run pass 1 (skip pole metrics)
                is_pass1 = use_manual_pole_frames and use_pole_analysis

                results = run_unified_pipeline(
                    st.session_state.video_path,
                    pole_model_path=pole_model_path,
                    pole_conf=pole_conf,
                    enable_pose=need_pose,
                    enable_hip=use_hip_analysis,
                    enable_step=use_step_analysis,
                    enable_max_hip_height=use_max_hip_height,
                    enable_pole=use_pole_analysis,
                    start_frame=s_f, plant_frame=p_f, end_frame=e_f,
                    athlete_height_m=athlete_height_m,
                    progress_callback=_prog_cb,
                    step_min_lift=step_min_lift,
                    step_min_dist=step_min_dist,
                    pole_length_m=pole_length_m,
                    enable_ml_steps=use_ml_steps,
                    skip_pole_metrics=is_pass1,
                )

                _unpack_and_store_results(results, is_pass1=is_pass1)

                if is_pass1:
                    status_text.text("Pass 1 Complete! Select pole frames below.")
                else:
                    status_text.text("Processing Complete!")
                st.rerun()

            except Exception as e:
                st.error(f"Analysis failed: {e}")
                logger.exception("Pipeline failed")

# Results Area
if st.session_state.results:
    st.divider()
    st.subheader("Analysis Results")
    st.video(st.session_state.results)
    
    with open(st.session_state.results, "rb") as f:
        st.download_button("Download Output", f, file_name="analysis_output.mp4")

    # --- MAX HIP HEIGHT METRIC ---
    hip_h = getattr(st.session_state, 'max_hip_height', None)
    if hip_h and isinstance(hip_h, dict):
        st.divider()
        h_m = hip_h['height_m']
        h_in = hip_h['height_in']
        h_ft = int(h_in // 12)
        h_in_r = h_in % 12
        col_hip1, col_hip2 = st.columns(2)
        with col_hip1:
            st.metric("Max Hip Height", f"{h_ft}'{h_in_r:.0f}\" ({h_m:.2f} m)")
        if 'predicted_clear_m' in hip_h:
            pc_m = hip_h['predicted_clear_m']
            pc_in = hip_h['predicted_clear_in']
            pc_ft = int(pc_in // 12)
            pc_in_r = pc_in % 12
            with col_hip2:
                st.metric("Predicted Max Clear", f"{pc_ft}'{pc_in_r:.0f}\" ({pc_m:.2f} m)")
        st.caption(f"Peak hip height at frame {hip_h['peak_frame']}. Measured from lowest foot near plant to lower hip point at peak. Predicted clear = lower hip - 6\".")

    # Display Debug Images
    debug_dir = os.path.join(os.getcwd(), "debug_output")
    vid_path = st.session_state.video_path
    vid_base = os.path.splitext(os.path.basename(vid_path))[0]
    
    debug_tip = os.path.join(debug_dir, f"{vid_base}_debug_tip_hand.jpg")
    debug_top = os.path.join(debug_dir, f"{vid_base}_debug_top_hand.jpg")
    
    col1, col2 = st.columns(2)
    
    if os.path.exists(debug_tip):
        with col1:
            st.image(debug_tip, caption="Phase 1: Ground Tip to Bottom Hand")
            with open(debug_tip, "rb") as f:
                st.download_button("Download Tip-Hand Image", f, file_name="debug_tip_hand.jpg")
            
    if os.path.exists(debug_top):
        with col2:
            st.image(debug_top, caption="Phase 2: Sky End to Bottom Hand")
            with open(debug_top, "rb") as f:
                st.download_button("Download Top-Hand Image", f, file_name="debug_top_hand.jpg")
    
    # --- POLE CALIBRATION & BEND IMAGES ---
    debug_bend = os.path.join(debug_dir, f"{vid_base}_debug_bend.jpg")
    debug_plant = os.path.join(debug_dir, f"{vid_base}_debug_plant.jpg")
    
    if os.path.exists(debug_bend) or os.path.exists(debug_plant):
        st.divider()
        col_b1, col_b2 = st.columns(2)
        
        if os.path.exists(debug_plant):
            with col_b1:
                st.subheader("Pole Tip Projection (Plant)")
                st.image(debug_plant, caption="Plant Frame: Reconstructed Tip (Blue) from Bottom Hand (Green)")
                with open(debug_plant, "rb") as f:
                    st.download_button("Download Plant Projection", f, file_name="debug_plant.jpg")
                    
        if os.path.exists(debug_bend):
            with col_b2:
                st.subheader("Maximum Pole Bend")
                st.image(debug_bend, caption="Max-Bend Frame: Projected Tip (Blue) to Top (Red)")
                with open(debug_bend, "rb") as f:
                    st.download_button("Download Max-Bend Image", f, file_name="debug_max_bend.jpg")

    # --- POLE BEND PLOT ---
    bend_result = getattr(st.session_state, 'bend_data', None)
    if bend_result and isinstance(bend_result, dict):
        # We now have poly_max_bend and smoothed max_val
        poly_max = bend_result.get('poly_max_bend')
        max_val = bend_result.get('max_bend') 
        bend_series = bend_result.get('bend_series', [])
        poly_series = bend_result.get('poly_series', [])
        
        st.divider()
        st.subheader("Pole Bend Progression")
        if poly_max is not None:
            st.metric("Mathematical Min Chord Ratio", f"{poly_max:.1f}%", help="True minimum calculated from a quadratic curve fit over the bend window.")
        elif max_val is not None:
             st.metric("Max Pole Chord Ratio", f"{max_val:.1f}%")
             
        st.caption("Lower value = more bend. 100% = perfectly straight.")
            
        if bend_series:
            fps = getattr(st.session_state, 'fps', 30.0)
            times = [i / fps for i in range(len(bend_series))]
            raw_series = bend_result.get('raw_series', [])
            
            import pandas as pd
            plot_data = {
                'Time after Plant (s)': times,
                'Smoothed Average (%)': bend_series
            }
            if raw_series and len(raw_series) == len(times):
                plot_data['Raw Calculation (%)'] = raw_series
            if poly_series and len(poly_series) == len(times):
                plot_data['Quadratic Fit (%)'] = poly_series
                
            bend_df = pd.DataFrame(plot_data).set_index('Time after Plant (s)')
            st.line_chart(bend_df)
            st.caption("Tracking the straight-line distance (chord) between the stationary tip and moving top endpoint. "
                       "Note that polynomial fit curves provide the mathematical true minimum peak bend.")

    # --- APPROACH VELOCITY ---
    vel_data = getattr(st.session_state, 'velocity_data', [])
    if vel_data:
        st.divider()
        st.subheader("Approach Velocity")

        import pandas as pd
        vel_df = pd.DataFrame(vel_data)

        height_sc = getattr(st.session_state, 'height_scale', None)
        plant_f = getattr(st.session_state, 'p_frame', None)

        # Compute takeoff velocity (avg of 0.25s before plant)
        cap_v = cv2.VideoCapture(st.session_state.video_path)
        fps_v = cap_v.get(cv2.CAP_PROP_FPS) or 30.0
        cap_v.release()
        takeoff_window = int(0.25 * fps_v)
        if plant_f is not None:
            takeoff_entries = vel_df[vel_df['frame'].between(plant_f - takeoff_window, plant_f)]
        else:
            takeoff_entries = pd.DataFrame()

        if height_sc:
            peak_v = vel_df['velocity_m_s'].max()
            peak_mph = vel_df['velocity_mph'].max()
            avg_v = vel_df['velocity_m_s'].mean()
            avg_mph = vel_df['velocity_mph'].mean()
            takeoff_v = takeoff_entries['velocity_m_s'].mean() if not takeoff_entries.empty else avg_v
            takeoff_mph = takeoff_entries['velocity_mph'].mean() if not takeoff_entries.empty else avg_mph

            col_v1, col_v2, col_v3 = st.columns(3)
            with col_v1:
                st.metric("Peak Velocity", f"{peak_v:.2f} m/s ({peak_mph:.1f} mph)")
            with col_v2:
                st.metric("Avg Velocity", f"{avg_v:.2f} m/s ({avg_mph:.1f} mph)")
            with col_v3:
                st.metric("Takeoff Velocity", f"{takeoff_v:.2f} m/s ({takeoff_mph:.1f} mph)")

            import altair as alt
            chart_df = vel_df[['time_s', 'velocity_m_s', 'velocity_mph']].rename(
                columns={'time_s': 'Time (s)', 'velocity_m_s': 'Velocity (m/s)', 'velocity_mph': 'Velocity (mph)'}
            )
            base = alt.Chart(chart_df).encode(x='Time (s):Q')
            line_ms = base.mark_line(color='#1f77b4').encode(
                y=alt.Y('Velocity (m/s):Q', scale=alt.Scale(domainMin=0))
            )
            st.altair_chart(line_ms, use_container_width=True)
        else:
            peak_v = vel_df['velocity_m_s'].max()
            avg_v = vel_df['velocity_m_s'].mean()
            takeoff_v = takeoff_entries['velocity_m_s'].mean() if not takeoff_entries.empty else avg_v

            col_v1, col_v2, col_v3 = st.columns(3)
            with col_v1:
                st.metric("Peak Velocity", f"{peak_v:.0f} px/s")
            with col_v2:
                st.metric("Avg Velocity", f"{avg_v:.0f} px/s")
            with col_v3:
                st.metric("Takeoff Velocity", f"{takeoff_v:.0f} px/s")

            import altair as alt
            chart_df = vel_df[['time_s', 'velocity_m_s']].rename(
                columns={'time_s': 'Time (s)', 'velocity_m_s': 'Velocity (px/s)'}
            )
            chart = alt.Chart(chart_df).mark_line().encode(
                x='Time (s):Q',
                y=alt.Y('Velocity (px/s):Q', scale=alt.Scale(domainMin=0))
            ).interactive()
            st.altair_chart(chart, use_container_width=True)

        st.caption("Velocity computed from hip center horizontal displacement. Smoothed over 5 frames. "
                   "Takeoff velocity is the average over the 0.25s before the plant frame.")

        # Download CSV
        csv_vel = vel_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Velocity Data (CSV)",
            data=csv_vel,
            file_name=f"{vid_base}_approach_velocity.csv",
            mime="text/csv",
            key='download-velocity-csv'
        )

    # --- FOOT PATH CSV ---
    ankles = (getattr(st.session_state, 'l_ankles', []), getattr(st.session_state, 'r_ankles', []))
    l_ank, r_ank = ankles
    stride_data_local = getattr(st.session_state, 'stride_data', [])
    
    if l_ank or r_ank:
        st.divider()
        st.subheader("Gait Data Export")
        
        # Build CSV Data
        # We need to know the frame range. Ankle lists are within the (Start, End) window.
        start_f = getattr(st.session_state, 's_frame', 0)
        plant_f = getattr(st.session_state, 'p_frame', start_f + len(l_ank))
        
        # Detected step frames
        step_frames = {s['frame'] for s in stride_data_local}
        
        csv_rows = []
        # Restrict the loop to only output data up to plant_frame
        max_idx = min(max(len(l_ank), len(r_ank)), max(0, plant_f - start_f + 1))
        
        for i in range(max_idx):
            f_idx = start_f + i
            l_pos = (l_ank[i][0], l_ank[i][1]) if i < len(l_ank) and l_ank[i] is not None else (None, None)
            r_pos = (r_ank[i][0], r_ank[i][1]) if i < len(r_ank) and r_ank[i] is not None else (None, None)
            is_step = f_idx in step_frames
            
            csv_rows.append({
                "Frame": f_idx,
                "L_Ankle_X": l_pos[0],
                "L_Ankle_Y": l_pos[1],
                "R_Ankle_X": r_pos[0],
                "R_Ankle_Y": r_pos[1],
                "Step_Detected": is_step
            })
            
        import pandas as pd
        gait_df = pd.DataFrame(csv_rows)
        csv_data = gait_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="Download Ankle Paths CSV",
            data=csv_data,
            file_name=f"{vid_base}_gait_analysis.csv",
            mime="text/csv"
        )
        st.caption("CSV includes frame-by-frame ankle coordinates and step detection flags.")

        # --- FOOT PATH PLOT WITH STRIKES ---
        st.subheader("Detailed Foot Path")
        # We want to show Y-path with Dots on the step frames
        # We use st.scatter_chart or st.line_chart with markers if possible
        # Actually st.line_chart doesn't easily support mixed scatter. 
        # Using a combined DF:
        plot_df = pd.DataFrame({
            'Frame': gait_df['Frame'],
            'Left Ankle Y': pd.to_numeric(gait_df['L_Ankle_Y'], errors='coerce'),
            'Right Ankle Y': pd.to_numeric(gait_df['R_Ankle_Y'], errors='coerce')
        }).set_index('Frame')
        
        # Create a series for markers (non-steps = NaN)
        import numpy as np
        l_strikes = gait_df.apply(lambda r: r['L_Ankle_Y'] if r['Step_Detected'] else np.nan, axis=1)
        r_strikes = gait_df.apply(lambda r: r['R_Ankle_Y'] if r['Step_Detected'] else np.nan, axis=1)
        
        plot_df['Left Strike'] = pd.to_numeric(l_strikes, errors='coerce').values
        plot_df['Right Strike'] = pd.to_numeric(r_strikes, errors='coerce').values
        
        st.line_chart(plot_df)
        st.caption("Lines: Continuous Ankle Y-Path. Large Dots: Detected Foot Strikes.")
    # --- PLOTTING (STRIDE) ---
    st.divider()
    st.subheader("Stride Analysis Plots")
    
    stride_data = getattr(st.session_state, 'stride_data', [])
    
    if stride_data:
        import pandas as pd
        df = pd.DataFrame(stride_data)
        
        if not df.empty:
            # Get FPS for plotting
            cap_fps = cv2.VideoCapture(st.session_state.video_path)
            fps_val = cap_fps.get(cv2.CAP_PROP_FPS)
            cap_fps.release()
            st.session_state.fps = fps_val if fps_val > 0 else 30.0
            
            # Check if we have height-based calibration data (cm/inches)
            has_real_units = 'stride_cm' in df.columns and df['stride_cm'].notna().any()
            
            if has_real_units:
                st.info("Stride calibrated using athlete height reference.")
                
                # Primary plot: cm
                st.markdown("**Stride Length (cm) vs Step Number**")
                import altair as alt
                stride_chart_df = df[['stride_cm']].reset_index().rename(columns={'index': 'Step', 'stride_cm': 'Stride (cm)'})
                y_max = stride_chart_df['Stride (cm)'].max() * 1.1
                chart = alt.Chart(stride_chart_df).mark_line(point=True).encode(
                    x='Step:Q',
                    y=alt.Y('Stride (cm):Q', scale=alt.Scale(domainMin=0, domainMax=y_max))
                ).interactive()
                st.altair_chart(chart, use_container_width=True)
                
                # Show average stats
                avg_cm = df['stride_cm'].mean()
                avg_in = df['stride_in'].mean()
                st.metric("Average Stride", f"{avg_cm:.0f} cm ({avg_in:.1f} in)")
            else:
                # Fallback: pixels
                vid_cap = cv2.VideoCapture(st.session_state.video_path)
                v_width = vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                vid_cap.release()
                df['stride_px'] = df['stride_norm'] * v_width
                
                st.markdown("**Stride Length (px) vs Step Number**")
                import altair as alt
                stride_px_df = df[['stride_px']].reset_index().rename(columns={'index': 'Step', 'stride_px': 'Stride (px)'})
                y_max = stride_px_df['Stride (px)'].max() * 1.1
                chart = alt.Chart(stride_px_df).mark_line(point=True).encode(
                    x='Step:Q',
                    y=alt.Y('Stride (px):Q', scale=alt.Scale(domainMin=0, domainMax=y_max))
                ).interactive()
                st.altair_chart(chart, use_container_width=True)
            
            # Download Data
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Stride Data (CSV)",
                csv,
                "stride_data.csv",
                "text/csv",
                key='download-csv'
            )
        else:
            st.write("No strides detected.")
    else:
        st.write("No stride data available.")

    # --- FOOT PATH PLOT ---
    st.divider()
    st.subheader("Foot Path & Contact Analysis")
    
    l_ankles = getattr(st.session_state, 'l_ankles', [])
    r_ankles = getattr(st.session_state, 'r_ankles', [])
    stride_data_local = getattr(st.session_state, 'stride_data', [])
    
    if l_ankles and r_ankles:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Extract Y values from (x, y) tuples
        l_ankles_y = [pt[1] if pt is not None else np.nan for pt in l_ankles]
        r_ankles_y = [pt[1] if pt is not None else np.nan for pt in r_ankles]
        
        # Create Frame Array
        frames = np.arange(len(l_ankles_y))
        
        fig, ax = plt.subplots(figsize=(10, 4))
        
        ax.plot(frames, l_ankles_y, label="Left Ankle Y", color="blue", linewidth=1, alpha=0.7)
        ax.plot(frames, r_ankles_y, label="Right Ankle Y", color="red", linewidth=1, alpha=0.7)
        
        for s in stride_data_local:
            f = s['frame']
            if f < len(l_ankles_y):
                ax.axvline(x=f, color='green', linestyle='--', alpha=0.3)
                
                ly = l_ankles_y[f] if f < len(l_ankles_y) else np.nan
                ry = r_ankles_y[f] if f < len(r_ankles_y) else np.nan
                
                if not np.isnan(ly):
                    ax.scatter([f], [ly], color='blue', s=30, zorder=5)
                if not np.isnan(ry):
                    ax.scatter([f], [ry], color='red', s=30, zorder=5)

        ax.set_title("Ankle Vertical Path & Detected Steps")
        ax.set_xlabel("Frame Index")
        ax.set_ylabel("Vertical Position (px)")
        ax.set_xlim(left=0)
        all_y = [v for v in l_ankles_y + r_ankles_y if not np.isnan(v)]
        if all_y:
            ax.set_ylim(bottom=0, top=max(all_y) * 1.05)
        ax.invert_yaxis() # So higher Y (TOP) is higher on plot
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # Download Plot
        import io
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        st.download_button("Download Plot (PNG)", buf.getvalue(), "foot_path_plot.png", "image/png")



