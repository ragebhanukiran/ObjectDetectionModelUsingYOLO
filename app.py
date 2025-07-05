import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="🚗 YOLOv8 Vehicle Detector", layout="wide")

# Load YOLOv8 model once
@st.cache_resource
def load_model():
    model = YOLO("runs/detect/train/weights/best.pt")
    model.conf = 0.4  # Confidence threshold
    return model

model = load_model()
class_names = ['car', 'emv', 'htv']

st.title("🚗 YOLOv8 Vehicle Detector")
st.markdown("Detect vehicles (`car`, `emv`, `htv`) from uploaded images or live webcam.")

# Tabs: One for Upload, One for Webcam
tab1, tab2 = st.tabs(["📸 Image Upload", "🎥 Live Detection"])

# --------------- 📸 Upload Tab ---------------
with tab1:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)
        
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.subheader("🔍 Running Detection...")

        results = model(img_array)[0]

        for box in results.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{class_names[cls_id]} {conf:.2f}"
            cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_array, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        st.image(img_array, caption="🖼️ Detection Result", use_column_width=True)

# --------------- 🎥 Real-Time Detection Tab ---------------
with tab2:
    st.markdown("Use your webcam for real-time detection. Press `Q` in the live feed to quit.")

    class YOLOTransformer(VideoTransformerBase):
        def transform(self, frame):
            image = frame.to_ndarray(format="bgr24")
            results = model(image)[0]

            for box in results.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                if cls_id >= len(class_names): continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{class_names[cls_id]} {conf:.2f}"
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            return image

    webrtc_streamer(
        key="realtime",
        video_transformer_factory=YOLOTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True,
    )

    
