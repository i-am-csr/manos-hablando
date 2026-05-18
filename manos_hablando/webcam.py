import cv2
import torch
import numpy as np
from pathlib import Path
import tempfile

from manos_hablando.config import MODELS_DIR
from manos_hablando.dataset import load_keypoints, normalize_keypoints
from manos_hablando.data.mediapipe_handler import extract_keypoints
from manos_hablando.modeling.predict import load_model, predict_top_k

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_, _, encoder = load_keypoints()
model = load_model(MODELS_DIR / "lsm_transformer.pt", len(encoder.classes_), device)

# Webcam loop
cap = cv2.VideoCapture(0)
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Save frame to temp file for MediaPipe
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        cv2.imwrite(f.name, frame)
        tmp_path = Path(f.name)

    result = predict_top_k(tmp_path, model, encoder, device, k=1)
    tmp_path.unlink()

    if result:
        letter, conf = result[0]
        print(f"{letter} {conf*100:.1f}%", end="\r")  # overwrite same line

        # Draw on frame
        cv2.putText(frame, f"{letter} {conf*100:.1f}%",
                    (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 255, 0), 3)

    cv2.imshow("manos-hablando", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()