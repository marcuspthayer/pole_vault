import streamlit as st
import tempfile
import cv2
import numpy as np
import os

# ---------------- Page Config ----------------
st.set_page_config(page_title="AlphaPeak PV Analysis", layout="wide")

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

.card {
    background-color: #F0F2F6;
    padding: 20px;
    border-radius: 10px;
    font-family: sans-serif;
    color: #262730;
}

.stButton>button {
    background-color: #FF3300;
    color: white;
    font-weight: bold;
    border-radius: 5px;
}

.stCheckbox>div>label {
    color: #262730;
    font-family: sans-serif;
}

body {
    background-color: #FFFFFF;
    color: #262730;
    font-family: sans-serif;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Session State ----------------
if "video_path" not in st.session_state:
    st.session_state.video_path = None
if "full_pole_frame" not in st.session_state:
    st.session_state.full_pole_frame = None
if "results" not in st.session_state:
    st.session_state.results = None
if "page" not in st.session_state:
    st.session_state.page = "upload"  # can be "upload" or "results"

st.markdown('<h1 class="gradient-header">AlphaForm: Pole Vault Analysis</h1>', unsafe_allow_html=True)

# ---------------- PAGE: Upload + Select Frame ----------------
if st.session_state.page == "upload":
    st.subheader("Step 1: Upload Your Vault Video")
    uploaded = st.file_uploader("Upload MP4 / MOV / AVI", type=["mp4", "mov", "avi"])

    if uploaded:
        if st.session_state.video_path is None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded.read())
                st.session_state.video_path = tmp.name

        st.video(st.session_state.video_path, format="video/mp4")

        # Step 2: Select full pole frame
        st.subheader("Step 2: Select Full Pole Frame")
        cap = cv2.VideoCapture(st.session_state.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_idx = st.slider("Move slider to find full pole", 0, total_frames - 1, 0)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, caption=f"Frame {frame_idx}", width=700)

        if st.button("✅ This is my full pole"):
            st.session_state.full_pole_frame = frame_idx
            st.success(f"Selected frame {frame_idx} saved!")

        # Step 3: Analysis options
        st.subheader("Step 3: Analysis Options (Stub)")
        enable_steps = st.checkbox("Label steps", value=True)
        crop_to_first = st.checkbox("Crop to first step", value=False)

        if st.button("🚀 Run Biomechanics Analysis", use_container_width=True):
            with st.spinner("Running analysis…"):
                # ----- STUB: replace with backend later -----
                stub_output = st.session_state.video_path  # pretend processed video
                st.session_state.results = stub_output
                st.session_state.page = "results"  # move to results page

# ---------------- PAGE: Results ----------------
elif st.session_state.page == "results":
    st.subheader("Step 4: Results")
    st.video(st.session_state.results, format="video/mp4")

    # Display placeholder metrics
    st.markdown("""
    **Metrics (Placeholder):**  
    - Stride rate: 3.2 steps/s  
    - Cadence: 192 spm  
    - Hip droop (worst 5%): +2.1%  
    """)

    # Example placeholder chart for hip path
    import pandas as pd
    import altair as alt

    time = np.arange(0, 10, 0.1)
    hip_y = np.sin(time) + 5  # dummy data
    df = pd.DataFrame({"Time (s)": time, "Hip Y": hip_y})
    chart = alt.Chart(df).mark_line(color="#FF3300").encode(
        x="Time (s)",
        y="Hip Y"
    ).properties(width=700)
    st.altair_chart(chart)

    # Download button
    with open(st.session_state.results, "rb") as f:
        st.download_button(
            "Download Analyzed Video",
            f,
            file_name="pv_analysis_stub.mp4"
        )

    st.info(f"Full pole frame selected: {st.session_state.full_pole_frame}")

    # Go back button
    if st.button("🔙 Go Back to Upload / Frame Selection"):
        st.session_state.page = "upload"
        st.session_state.results = None
