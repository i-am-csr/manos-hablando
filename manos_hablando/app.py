import tempfile
from pathlib import Path

import av
import cv2
import torch
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

from manos_hablando.config import MODELS_DIR
from manos_hablando.dataset import load_keypoints
from manos_hablando.data.mediapipe_handler import extract_keypoints
from manos_hablando.modeling.predict import load_model, predict_top_k

# ─────────────────────────────────────────
# Hand connections (MediaPipe topology)
# ─────────────────────────────────────────
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),           # thumb
    (0,5),(5,6),(6,7),(7,8),           # index
    (0,9),(9,10),(10,11),(11,12),      # middle
    (0,13),(13,14),(14,15),(15,16),    # ring
    (0,17),(17,18),(18,19),(19,20),    # pinky
    (5,9),(9,13),(13,17),              # palm
]


def draw_landmarks(img: np.ndarray, keypoints: list) -> np.ndarray:
    """Draw 21 hand landmarks and connections on the frame."""
    h, w, _ = img.shape

    # Draw connections
    for start, end in CONNECTIONS:
        x0 = int(keypoints[start][0] * w)
        y0 = int(keypoints[start][1] * h)
        x1 = int(keypoints[end][0] * w)
        y1 = int(keypoints[end][1] * h)
        cv2.line(img, (x0, y0), (x1, y1), (0, 200, 0), 2)

    # Draw points
    for i, (x, y, z) in enumerate(keypoints):
        cx = int(x * w)
        cy = int(y * h)
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), -1)

    return img


# ─────────────────────────────────────────
# Setup
# ─────────────────────────────────────────
st.set_page_config(page_title="manos-hablando", layout="centered")
st.title("🤟 Manos Hablando")
st.caption("LSM hand sign recognition — show a sign to your webcam")


@st.cache_resource
def setup():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, encoder = load_keypoints()
    model = load_model(MODELS_DIR / "lsm_transformer.pt", len(encoder.classes_), device)
    return model, encoder, device


model, encoder, device = setup()


# ─────────────────────────────────────────
# Video processor
# ─────────────────────────────────────────
class HandSignProcessor(VideoProcessorBase):
    def __init__(self):
        self.result = None
        self.keypoints = None
        self.frame_count = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Only process every 10 frames to avoid lag
        self.frame_count += 1
        if self.frame_count % 10 == 0:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                cv2.imwrite(f.name, img)
                tmp_path = Path(f.name)

            self.keypoints = extract_keypoints(tmp_path)
            result = predict_top_k(tmp_path, model, encoder, device, k=1)
            tmp_path.unlink()

            if result:
                letter, conf = result[0]
                self.result = (letter, conf)
                print(f"Detected: {letter} ({conf*100:.1f}%)")

        # Draw landmarks
        if self.keypoints:
            img = draw_landmarks(img, self.keypoints)

        # Draw prediction
        if self.result:
            letter, conf = self.result
            cv2.putText(
                img, f"{letter} {conf*100:.1f}%",
                (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                2, (0, 255, 0), 3
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ─────────────────────────────────────────
# UI
# ─────────────────────────────────────────
ctx = webrtc_streamer(
    key="manos-hablando",
    video_processor_factory=HandSignProcessor,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
)