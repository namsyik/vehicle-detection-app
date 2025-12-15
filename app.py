import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"    

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Vehicle Detection System",
    page_icon="🚗",
    layout="centered"
)

# ---------------------------------------------------------
# Model Loading (cached for performance)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    """
    Load the trained YOLOv8 model.
    Cached to avoid reloading on every interaction.
    """
    return YOLO("best.pt")

model = load_model()

# ---------------------------------------------------------
# Application Header
# ---------------------------------------------------------
st.title("🚦 Vehicle Detection System")
st.markdown(
    """
    This application performs **automatic vehicle detection** using a  
    **YOLOv8 deep learning model** trained on a labeled vehicle dataset.

    **Supported vehicle classes:**
    - Bus
    - Car
    - Motorcycle
    - Pickup
    - Truck
    """
)

st.divider()

# ---------------------------------------------------------
# User Controls
# ---------------------------------------------------------
st.subheader("Input Configuration")

mode = st.radio(
    "Select input type:",
    ["Image", "Video"],
    help="Choose whether to run detection on a single image or a video file."
)

confidence = st.slider(
    "Detection Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.4,
    step=0.05,
    help="Lower values detect more objects but may increase false positives."
)

st.divider()

# ---------------------------------------------------------
# Image Detection
# ---------------------------------------------------------
if mode == "Image":
    st.subheader("Image-Based Vehicle Detection")
    st.markdown(
        """
        Upload an image containing vehicles.  
        The system will detect and label all recognized vehicles with bounding boxes.
        """
    )

    uploaded_image = st.file_uploader(
        "Upload an image file",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        image_np = np.array(image)

        with st.spinner("Running vehicle detection on image..."):
            results = model(image_np, conf=confidence)

        annotated_image = results[0].plot()
        st.image(
            annotated_image,
            caption="Detection Result",
            use_container_width=True
        )

# ---------------------------------------------------------
# Video Detection
# ---------------------------------------------------------
if mode == "Video":
    st.subheader("Video-Based Vehicle Detection")
    st.markdown(
        """
        Upload a short video clip.  
        The system processes the video **frame-by-frame** and displays detected vehicles.

        ⚠️ Recommended: ≤ 30 seconds, ≤ 720p resolution.
        """
    )

    uploaded_video = st.file_uploader(
        "Upload a video file",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video:
        # ---- Create temp file SAFELY (Windows compatible) ----
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, uploaded_video.name)

        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())

        cap = cv2.VideoCapture(video_path)
        frame_placeholder = st.empty()

        st.info("Processing video. This may take a moment on CPU.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=confidence)
            annotated_frame = results[0].plot()
            annotated_frame = cv2.cvtColor(
                annotated_frame, cv2.COLOR_BGR2RGB
            )

            frame_placeholder.image(
                annotated_frame,
                use_container_width=True
            )

        # ---- CLEANUP (ORDER MATTERS) ----
        cap.release()

        try:
            os.remove(video_path)
            os.rmdir(temp_dir)
        except PermissionError:
            pass  # safe fallback on Windows

        st.success("Video processing completed successfully.")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.divider()
st.markdown(
    """
    **Model:** YOLOv8 (Ultralytics)  
    **Inference:** CPU-based, optimized for deployment  
    **Deployment Platform:** Streamlit  

    This system is designed for **educational, research, and demonstration purposes**.
    """
)
