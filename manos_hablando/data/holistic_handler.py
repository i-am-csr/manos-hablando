"""
MediaPipe Tasks API holistic extractor (face + pose + hands).

Replaces the legacy mediapipe.solutions.holistic API by running the three
Tasks-API landmarkers (FaceLandmarker, PoseLandmarker, HandLandmarker)
side by side on the same frame and concatenating their outputs into a
single per-frame feature vector.

Feature layout (per frame), matching references/mediapipe_holistic_reference.ipynb:

    pose        33 × 4 (x, y, z, visibility)   = 132
    face       478 × 3                          = 1434
    left hand   21 × 3                          =   63
    right hand  21 × 3                          =   63
                                       FEATURE_DIM = 1692

Missing detections are zero-padded so the feature dim stays constant.
"""

from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
FACE_MODEL_PATH = MODELS_DIR / "face_landmarker.task"
POSE_MODEL_PATH = MODELS_DIR / "pose_landmarker.task"
HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"

POSE_LANDMARKS = 33
FACE_LANDMARKS = 478
HAND_LANDMARKS = 21

POSE_DIMS = POSE_LANDMARKS * 4
FACE_DIMS = FACE_LANDMARKS * 3
HAND_DIMS = HAND_LANDMARKS * 3
FEATURE_DIM = POSE_DIMS + FACE_DIMS + HAND_DIMS * 2  # 1692

# Offsets of each section inside the flat per-frame feature vector.
POSE_OFFSET = 0
FACE_OFFSET = POSE_OFFSET + POSE_DIMS              # 132
LEFT_HAND_OFFSET = FACE_OFFSET + FACE_DIMS         # 1566
RIGHT_HAND_OFFSET = LEFT_HAND_OFFSET + HAND_DIMS   # 1629

# Pose landmark indices used as the body anchor for normalization.
LEFT_SHOULDER_IDX = 11
RIGHT_SHOULDER_IDX = 12

# Face landmark subset for the holistic words pipeline.
# The full 478-point mesh is overkill for sign-language vocabulary and
# starves training of signal on a small dataset (~2k samples) by drowning
# the 126-dim hand block in 1434 mostly-redundant face dims. This subset
# covers the regions LSM expressions actually live in: lip shape (open/closed,
# rounded/wide), eye openness, and brow position. Canonical mesh indices.
FACE_KEEP_INDICES: list[int] = sorted(
    {
        # Outer lip contour
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
        185, 40, 39, 37, 0, 267, 269, 270, 409,
        # Inner lip contour
        78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
        191, 80, 81, 82, 13, 312, 311, 310, 415,
        # Right eye
        33, 7, 163, 144, 145, 153, 154, 155, 133,
        173, 157, 158, 159, 160, 161, 246,
        # Left eye
        263, 249, 390, 373, 374, 380, 381, 382, 362,
        398, 384, 385, 386, 387, 388, 466,
        # Right eyebrow
        70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
        # Left eyebrow
        300, 293, 334, 296, 336, 285, 295, 282, 283, 276,
    }
)
TRIMMED_FACE_LANDMARKS = len(FACE_KEEP_INDICES)
TRIMMED_FACE_DIMS = TRIMMED_FACE_LANDMARKS * 3
TRIMMED_FEATURE_DIM = POSE_DIMS + TRIMMED_FACE_DIMS + HAND_DIMS * 2


def _check_models() -> None:
    """Raise a clear error if any landmarker model file is missing."""
    missing = [p for p in (FACE_MODEL_PATH, POSE_MODEL_PATH, HAND_MODEL_PATH) if not p.exists()]
    if not missing:
        return
    raise FileNotFoundError(
        "Missing MediaPipe landmarker model file(s):\n"
        + "\n".join(f"  - {p}" for p in missing)
        + "\n\nDownload with:\n"
        "  curl -o models/face_landmarker.task "
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task\n"
        "  curl -o models/pose_landmarker.task "
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    )


def _build_image_landmarkers() -> tuple[
    vision.FaceLandmarker, vision.PoseLandmarker, vision.HandLandmarker,
]:
    """Build the three landmarkers in IMAGE mode (single-frame)."""
    _check_models()
    face = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(FACE_MODEL_PATH)),
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
    )
    pose = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
    )
    hand = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )
    return face, pose, hand


def _build_video_landmarkers() -> tuple[
    vision.FaceLandmarker, vision.PoseLandmarker, vision.HandLandmarker,
]:
    """Build the three landmarkers in VIDEO mode."""
    _check_models()
    face = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(FACE_MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
    )
    pose = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )
    hand = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )
    return face, pose, hand


def _pose_to_vec(pose_result) -> np.ndarray:
    if not pose_result.pose_landmarks:
        return np.zeros(POSE_DIMS, dtype=np.float32)
    return np.array(
        [[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_result.pose_landmarks[0]],
        dtype=np.float32,
    ).flatten()


def _face_to_vec(face_result) -> np.ndarray:
    if not face_result.face_landmarks:
        return np.zeros(FACE_DIMS, dtype=np.float32)
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in face_result.face_landmarks[0]],
        dtype=np.float32,
    ).flatten()


def _hands_to_vecs(hand_result) -> tuple[np.ndarray, np.ndarray]:
    """Returns (left_vec, right_vec). Zeros for any hand not present."""
    left = np.zeros(HAND_DIMS, dtype=np.float32)
    right = np.zeros(HAND_DIMS, dtype=np.float32)
    if not hand_result.hand_landmarks:
        return left, right
    for i, lms in enumerate(hand_result.hand_landmarks):
        vec = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32).flatten()
        label = hand_result.handedness[i][0].display_name
        if label == "Left":
            left = vec
        else:
            right = vec
    return left, right


def build_feature_vector(face_result, pose_result, hand_result) -> np.ndarray:
    """Concatenate (pose, face, left_hand, right_hand) → (FEATURE_DIM,) float32."""
    pose_v = _pose_to_vec(pose_result)
    face_v = _face_to_vec(face_result)
    left_v, right_v = _hands_to_vecs(hand_result)
    return np.concatenate([pose_v, face_v, left_v, right_v])


def extract_holistic_video(video_path: str | Path) -> dict | None:
    """
    Run face + pose + hand landmarkers across every frame of a video and
    return the stacked (N_frames, FEATURE_DIM) feature tensor.

    A frame is counted as "detected" if either pose OR at least one hand
    was found — face alone is too forgiving (often hallucinated on noise).

    Returns:
        {
            "fps": float,
            "total_frames": int,
            "detected_frames": int,
            "features": np.ndarray of shape (total_frames, FEATURE_DIM),
            "modality_counts": {"face": int, "pose": int, "hands": int},
        }
        or None if no frames had a useful detection.
    """
    import cv2

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    rows: list[np.ndarray] = []
    n_face = n_pose = n_hand = n_useful = 0

    face_lm, pose_lm, hand_lm = _build_video_landmarkers()
    try:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            ts = int(frame_idx * 1000 / fps)

            face_r = face_lm.detect_for_video(mp_img, ts)
            pose_r = pose_lm.detect_for_video(mp_img, ts)
            hand_r = hand_lm.detect_for_video(mp_img, ts)

            if face_r.face_landmarks:
                n_face += 1
            if pose_r.pose_landmarks:
                n_pose += 1
            if hand_r.hand_landmarks:
                n_hand += 1
            if pose_r.pose_landmarks or hand_r.hand_landmarks:
                n_useful += 1

            rows.append(build_feature_vector(face_r, pose_r, hand_r))
            frame_idx += 1
    finally:
        face_lm.close()
        pose_lm.close()
        hand_lm.close()
        cap.release()

    if n_useful == 0 or not rows:
        return None

    features = np.stack(rows).astype(np.float32)
    return {
        "fps": float(fps),
        "total_frames": features.shape[0],
        "detected_frames": n_useful,
        "features": features,
        "modality_counts": {"face": n_face, "pose": n_pose, "hands": n_hand},
    }
