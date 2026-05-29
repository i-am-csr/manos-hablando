import tempfile
import time
import threading
from pathlib import Path

import av
import cv2
import torch
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

from manos_hablando.config import MODELS_DIR
from manos_hablando.dataset import load_keypoints, normalize_keypoints
from manos_hablando.dataset_video import load_video_keypoints
from manos_hablando.data.mediapipe_handler import extract_keypoints
from manos_hablando.modeling.predict import load_model as load_static_model
from manos_hablando.modeling.predict import predict_top_k
from manos_hablando.modeling.train_video import LSMVideoTransformer
from manos_hablando.text_processing import LetterBuffer, LLMProcessor

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

RECORD_SECONDS = 3


def draw_landmarks(img: np.ndarray, keypoints: list) -> np.ndarray:
    """Draw 21 hand landmarks and connections on the frame."""
    h, w, _ = img.shape

    for start, end in CONNECTIONS:
        x0 = int(keypoints[start][0] * w)
        y0 = int(keypoints[start][1] * h)
        x1 = int(keypoints[end][0] * w)
        y1 = int(keypoints[end][1] * h)
        cv2.line(img, (x0, y0), (x1, y1), (0, 200, 0), 2)

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
def setup_static():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, encoder = load_keypoints()
    model = load_static_model(
        MODELS_DIR / "lsm_transformer.pt", len(encoder.classes_), device,
    )
    return model, encoder, device


@st.cache_resource
def setup_video():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, encoder = load_video_keypoints()
    num_classes = len(encoder.classes_)
    model = LSMVideoTransformer(num_classes=num_classes).to(device)
    model_path = MODELS_DIR / "lsm_video_transformer.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, encoder, device


@st.cache_resource
def get_llm_processor() -> LLMProcessor:
    """Single shared LLM processor across reruns."""
    return LLMProcessor()


# ─────────────────────────────────────────
# Session state
# ─────────────────────────────────────────
if "sentence" not in st.session_state:
    st.session_state.sentence = ""
if "last_reconstructed" not in st.session_state:
    st.session_state.last_reconstructed = ""


# ─────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────
with st.sidebar:
    st.header("Mode")
    mode = st.radio(
        "Select sign type:",
        ["Static Signs (A-Y)", "Dynamic Signs (J, K, Ñ, Q, X, Z)"],
        index=0,
    )
    is_dynamic = "Dynamic" in mode

    if is_dynamic:
        st.divider()
        st.markdown(
            f"**How to use:**\n"
            f"1. Click **Record** below the video\n"
            f"2. Perform the sign ({RECORD_SECONDS}s window)\n"
            f"3. See the prediction"
        )
    else:
        st.divider()
        st.markdown(
            "**How to use:**\n"
            "1. Fingerspell a word letter by letter\n"
            "2. Hold each letter steady for ~0.5s\n"
            "3. Pause (lower your hand) to complete the word\n"
            "4. The LLM reconstructs the Spanish word"
        )


# ─────────────────────────────────────────
# Static mode processor (with letter buffering)
# ─────────────────────────────────────────
class StaticSignProcessor(VideoProcessorBase):
    def __init__(self):
        self.result = None
        self.keypoints = None
        self.frame_count = 0
        self.buffer = LetterBuffer(
            stability_window=15,
            confidence_threshold=0.7,
            word_boundary_frames=30,
        )

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        # Only run inference every 3rd frame for responsiveness
        if self.frame_count % 3 == 0:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                cv2.imwrite(f.name, img)
                tmp_path = Path(f.name)

            model, encoder, device = setup_static()
            self.keypoints = extract_keypoints(tmp_path)

            if self.keypoints is not None:
                result = predict_top_k(tmp_path, model, encoder, device, k=1)
                if result:
                    letter, conf = result[0]
                    self.result = (letter, conf)
                    self.buffer.add_prediction(letter, conf)
            else:
                self.buffer.add_no_hand()

            tmp_path.unlink()

        if self.keypoints:
            img = draw_landmarks(img, self.keypoints["keypoints"])

        if self.result:
            letter, conf = self.result
            cv2.putText(
                img, f"{letter} {conf*100:.1f}%",
                (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                2, (0, 255, 0), 3,
            )

        # Overlay the in-progress word on the video
        current_word = self.buffer.get_current_word()
        if current_word:
            cv2.putText(
                img, current_word,
                (30, img.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (255, 255, 0), 3,
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ─────────────────────────────────────────
# Dynamic mode processor (unchanged behavior)
# ─────────────────────────────────────────
class DynamicSignProcessor(VideoProcessorBase):
    def __init__(self):
        self.keypoints = None
        self.frame_count = 0
        self.lock = threading.Lock()
        self.recording = False
        self.record_start = 0.0
        self.frame_buffer: list[list] = []
        self.prediction = None

    def start_recording(self):
        with self.lock:
            self.recording = True
            self.record_start = time.time()
            self.frame_buffer = []
            self.prediction = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        current_kp = None
        if self.frame_count % 3 == 0:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                cv2.imwrite(f.name, img)
                tmp_path = Path(f.name)

            result = extract_keypoints(tmp_path)
            tmp_path.unlink()

            if result:
                current_kp = result["keypoints"]
                self.keypoints = current_kp

        with self.lock:
            if self.recording:
                elapsed = time.time() - self.record_start

                if current_kp is not None:
                    self.frame_buffer.append(current_kp)

                if elapsed >= RECORD_SECONDS:
                    self.recording = False
                    self._predict_from_buffer()

                remaining = max(0, RECORD_SECONDS - elapsed)
                cv2.putText(
                    img, f"REC {remaining:.1f}s",
                    (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 0, 255), 3,
                )
                cv2.rectangle(
                    img, (0, 0),
                    (img.shape[1]-1, img.shape[0]-1), (0, 0, 255), 4,
                )

            elif self.prediction:
                letter, conf = self.prediction
                cv2.putText(
                    img, f"{letter} {conf*100:.1f}%",
                    (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 255, 0), 3,
                )

        if self.keypoints:
            img = draw_landmarks(img, self.keypoints)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def _predict_from_buffer(self):
        """Run video transformer on the buffered keypoint sequence."""
        if len(self.frame_buffer) < 5:
            return

        model, encoder, device = setup_video()

        kp = np.array(self.frame_buffer, dtype=np.float32)
        kp_flat = kp.reshape(kp.shape[0], -1)
        kp_norm = normalize_keypoints(kp_flat)

        X_tensor = torch.tensor(
            kp_norm, dtype=torch.float32,
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(X_tensor)
            probs = torch.softmax(logits, dim=-1)
            confidence, idx = probs.max(dim=-1)

        letter = encoder.classes_[idx.item()]
        self.prediction = (letter, confidence.item())


# ─────────────────────────────────────────
# UI
# ─────────────────────────────────────────
if is_dynamic:
    ctx = webrtc_streamer(
        key="manos-hablando-dynamic",
        video_processor_factory=DynamicSignProcessor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
        },
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔴 Record", use_container_width=True, type="primary"):
            if ctx.video_processor:
                ctx.video_processor.start_recording()
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            if ctx.video_processor:
                with ctx.video_processor.lock:
                    ctx.video_processor.prediction = None
                    ctx.video_processor.frame_buffer = []

    if ctx.video_processor:
        with ctx.video_processor.lock:
            pred = ctx.video_processor.prediction
            n_frames = len(ctx.video_processor.frame_buffer)
            is_rec = ctx.video_processor.recording

        if pred:
            letter, conf = pred
            st.success(
                f"Predicted: **{letter}** ({conf*100:.1f}% confidence) "
                f"— {n_frames} frames captured"
            )
        elif is_rec:
            st.info("Recording... perform the sign now")

else:
    ctx = webrtc_streamer(
        key="manos-hablando-static",
        video_processor_factory=StaticSignProcessor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
        },
    )

    if st.button("🗑️ Clear all", use_container_width=True):
        st.session_state.sentence = ""
        st.session_state.last_reconstructed = ""
        if ctx.video_processor:
            ctx.video_processor.buffer.reset()

    # Live updating section — polls the processor every 0.5s
    @st.fragment(run_every=0.5)
    def live_text_panel():
        if ctx.video_processor is None:
            st.info("Start the webcam to begin fingerspelling.")
            return

        buf: LetterBuffer = ctx.video_processor.buffer
        llm = get_llm_processor()
        current_word = buf.get_current_word()

        # Detect word completion → call LLM → append to sentence
        if buf.word_complete():
            letters = list(current_word)
            if letters:
                reconstructed = llm.reconstruct(letters)
                st.session_state.last_reconstructed = reconstructed
                st.session_state.last_raw = "".join(letters)
                st.session_state.last_status = llm.last_status
                st.session_state.last_error = llm.last_error
                st.session_state.last_latency_ms = llm.last_latency_ms
                if st.session_state.sentence:
                    st.session_state.sentence += " " + reconstructed
                else:
                    st.session_state.sentence = reconstructed
            buf.reset()
            current_word = ""

        # Letter buffer
        st.subheader("Letter buffer")
        if current_word:
            spaced = " ".join(current_word)
            st.markdown(
                f"<div style='font-size: 32px; "
                f"font-family: monospace; letter-spacing: 4px;'>"
                f"{spaced}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='color: #888;'>"
                "Waiting for letters...</div>",
                unsafe_allow_html=True,
            )

        # Last LLM call — status + result
        if st.session_state.get("last_status"):
            st.subheader("Last LLM call")
            status = st.session_state.last_status
            raw = st.session_state.get("last_raw", "")
            recon = st.session_state.get("last_reconstructed", "")
            latency = st.session_state.get("last_latency_ms", 0)
            err = st.session_state.get("last_error", "")

            badge = {
                "success":  "🟢 SUCCESS",
                "fallback": "🔴 FALLBACK",
                "no_key":   "⚪ NO KEY",
                "calling":  "🟡 CALLING",
            }.get(status, status)

            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.markdown(f"**Status:** {badge}")
                st.markdown(f"**Latency:** {latency:.0f}ms")
                st.markdown(f"**Calls:** {llm.call_count}")
            with col_b:
                st.markdown(f"**Raw letters:** `{raw}`")
                st.markdown(f"**Reconstructed:** `{recon}`")
                if err:
                    st.markdown(
                        f"<div style='color:#c00; font-family: monospace; "
                        f"font-size: 12px;'>{err}</div>",
                        unsafe_allow_html=True,
                    )

        # Full sentence
        st.subheader("Full sentence")
        st.text_area(
            "sentence",
            value=st.session_state.sentence,
            height=120,
            label_visibility="collapsed",
        )

    live_text_panel()
