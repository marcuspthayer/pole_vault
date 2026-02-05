import streamlit as st
import tempfile
import os
import logging

from pvapp.pipelines.skeleton_overlay import process_video_skeleton_overlay
from pvapp.pipelines.hip_overlay import process_video_hip_overlay
from pvapp.logging_utils import setup_logger

st.set_page_config(page_title="AlphaPeak PV Analysis", layout="wide")

# Logging
LOG_FILE = os.path.join(os.getcwd(), "logs", "pv_streamlit.log")
setup_logger("pv", log_file=LOG_FILE)
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

if "video_path" not in st.session_state:
    st.session_state.video_path = None
if "results" not in st.session_state:
    st.session_state.results = None
if "page" not in st.session_state:
    st.session_state.page = "upload"
if "last_error" not in st.session_state:
    st.session_state.last_error = None

st.markdown('<h1 class="gradient-header">AlphaForm: Pole Vault Analysis</h1>', unsafe_allow_html=True)

if st.session_state.last_error is not None:
    st.error("Last processing error:")
    st.exception(st.session_state.last_error)

# ---------------- PAGE: Upload ----------------
if st.session_state.page == "upload":
    st.subheader("Upload Your Vault Video")
    uploaded = st.file_uploader("Upload MP4 / MOV / AVI", type=["mp4", "mov", "avi"])

    if uploaded:
        # Persist uploaded file to disk so OpenCV can read it
        if st.session_state.video_path is None:
            suffix = os.path.splitext(uploaded.name)[1].lower() or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                st.session_state.video_path = tmp.name

        st.video(st.session_state.video_path)

        with st.expander("Show logs (tail)"):
            try:
                if os.path.exists(LOG_FILE):
                    with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
                        lines = f.readlines()[-200:]
                    st.code("".join(lines))
                else:
                    st.write(f"No log file yet: {LOG_FILE}")
            except Exception as _e:
                st.write("Failed to read log file:")
                st.exception(_e)
                st.write(_e)

        st.subheader("Debug Step 0: Get skeleton overlay working end-to-end")
        col1, col2, col3 = st.columns(3)
        with col1:
            yolo_model_path = st.text_input("YOLO person model", value="yolo11n.pt")
        with col2:
            conf = st.slider("YOLO confidence", 0.05, 0.80, 0.25, 0.01)
        with col3:
            margin = st.slider("ROI margin", 0.0, 0.8, 0.30, 0.05)

        draw_roi = st.checkbox("Draw YOLO ROI box", value=True)

        max_frames = st.number_input("Max frames (debug; 0 = full video)", min_value=0, value=0, step=50)
        can_process = st.session_state.video_path is not None

        if st.button("🚀 Process video (skeleton overlay)", use_container_width=True, disabled=not can_process):
            st.session_state.last_error = None
            try:
                with st.spinner("Running YOLO + MediaPipe…"):
                    mf = None if int(max_frames) == 0 else int(max_frames)
                    out_path = process_video_skeleton_overlay(
                        st.session_state.video_path,
                        output_path=None,
                        yolo_model_path=yolo_model_path,
                        conf=float(conf),
                        margin=float(margin),
                        draw_roi_box=draw_roi,
                        log_file=LOG_FILE,
                        max_frames=mf,
                    )

                if not out_path or not os.path.exists(out_path):
                    raise RuntimeError(f"Processing finished but output file not found: {out_path}")

                logger.info(f"Processing complete | out={out_path}")
                st.session_state.results = out_path
                st.session_state.page = "results"
                st.rerun()

            except Exception as e:
                logger.exception("Processing failed")
                st.session_state.last_error = e
                st.error("Processing failed (details below).")
                st.exception(e)


# ---------------- PAGE: Results ----------------
if st.session_state.page == "results":
    st.subheader("Results")
    st.video(st.session_state.results)

    with open(st.session_state.results, "rb") as f:
        st.download_button(
            "Download processed video",
            f,
            file_name=os.path.basename(st.session_state.results),
        )

    if st.button("🔁 Process another video", use_container_width=True):
        st.session_state.page = "upload"
        st.session_state.results = None
        st.session_state.video_path = None
        st.session_state.last_error = None
        st.rerun()

