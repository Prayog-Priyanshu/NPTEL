import streamlit as st
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="YOLO Multi-Image Detection",
    layout="wide"
)

st.title("🧠 YOLOv11s Model Performance Report: Food image recognition from Indian THALI")
st.write("Upload multiple images and get predictions using your trained YOLO model.")

# ==========================
# DEVICE SETUP
# ==========================
device = "cuda" if torch.cuda.is_available() else "cpu"
st.success(f"Using device: {device}")

# ==========================
# LOAD MODEL (CACHE)
# ==========================
@st.cache_resource
def load_model():
    model = YOLO("my_model.pt")   # change if needed
    model.to(device)
    return model

model = load_model()

# ==========================
# IMAGE UPLOAD
# ==========================
uploaded_files = st.file_uploader(
    "Upload one or more images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# ==========================
# CONFIDENCE SLIDER
# ==========================
confidence = st.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.4,
    step=0.05
)

# ==========================
# PROCESS IMAGES
# ==========================
if uploaded_files:
    st.write(f"📸 {len(uploaded_files)} image(s) uploaded")

    cols = st.columns(2)

    with cols[0]:
        st.subheader("Original Images")

    with cols[1]:
        st.subheader("Predicted Images")

    for uploaded_file in uploaded_files:
        # Read image
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        # YOLO Inference
        results = model(
            image_np,
            conf=confidence,
            device=device,
            half=False,
            verbose=False
        )

        # Draw predictions
        annotated_img = results[0].plot())

        # Display side-by-side
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption=f"Original - {uploaded_file.name}", use_container_width=True)

        with col2:
            st.image(annotated_img, caption="Prediction", use_container_width=True)

else:
    st.info("👆 Upload images to start detection")


