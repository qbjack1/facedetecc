from pathlib import Path
import time

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import streamlit as st


MODEL_PATH = Path(__file__).with_name("detector.tflite")
APP_TITLE = "Face Detection: A Project by QBjack"
GITHUB_REPO_1 = "https://github.com/qbjack1/facedetecc"
GITHUB_REPO_2 = "https://github.com/qbjack1/Portfolio"

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.subheader("🔗 Please see the documentation located in the GitHub Repository")
st.markdown(GITHUB_REPO_1)
st.subheader("🔗 Check out my portfolio!")
st.markdown(GITHUB_REPO_2)

@st.cache_resource
def load_mp_detector(model_path: str, min_confidence: float):
    base_options = python.BaseOptions(model_asset_path=model_path)

    options = vision.FaceDetectorOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        min_detection_confidence=min_confidence
    )

    mp_detector = vision.FaceDetector.create_from_options(options)

    return mp_detector


def run_detection(image_np, detector):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)

    start = time.perf_counter()
    results = detector.detect(mp_image)
    elapsed = time.perf_counter() - start

    return results, elapsed


def extract_detections(results, image_size):
    width, height = image_size
    detections = []

    for detection in results.detections or []:
        bbox = detection.bounding_box
        score = detection.categories[0].score

        x1 = max(0, bbox.origin_x)
        y1 = max(0, bbox.origin_y)
        x2 = min(width, bbox.origin_x + bbox.width)
        y2 = min(height, bbox.origin_y + bbox.height)

        detections.append({
            "x": x1,
            "y": y1,
            "w": max(0, x2 - x1),
            "h": max(0, y2 - y1),
            "score": score,
        })

    return detections


def draw_detections(image, detections):
    result = image.copy()
    draw = ImageDraw.Draw(result)

    for detection in detections:
        x = detection["x"]
        y = detection["y"]
        w = detection["w"]
        h = detection["h"]
        score = detection["score"]

        draw.rectangle((x, y, x + w, y + h), outline="red", width=4)
        draw.text((x, max(0, y - 18)), f"{score:.2f}", fill="red")

    return result

if not MODEL_PATH.exists():
    st.error("Model file not found. Please ensure 'detector.tflite' is in the same directory as this script.")
    st.stop()

confidence_threshold = st.sidebar.slider(
    "Minimum Detection Confidence",
    min_value=0.1,
    max_value=0.95,
    value=0.3,
    step=0.05
)

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is None:
    st.info("Please upload an image to proceed.")
    st.stop()

image = PILImage.open(uploaded_file).convert('RGB')
image_np = np.array(image)

detector = load_mp_detector(str(MODEL_PATH), confidence_threshold)
results, elapsed = run_detection(image_np, detector)
detections = extract_detections(results, image.size)
result_image = draw_detections(image, detections)

metric_cols = st.columns(2)
metric_cols[0].metric("Faces Detected", len(detections))
metric_cols[1].metric("Detection Time (s)", f"{elapsed:.4f}s")

left_col, right_col = st.columns(2)
left_col.subheader("Original Image")
left_col.image(image, use_container_width=True)

right_col.subheader("Detected Faces")
right_col.image(result_image, use_container_width=True)
