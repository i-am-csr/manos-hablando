from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from loguru import logger

# Landmark names for reference
LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "hand_landmarker.task"


def _build_landmarker() -> vision.HandLandmarker:
    """Initialize and return a MediaPipe HandLandmarker instance."""
    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def extract_keypoints(image_path: str | Path) -> dict | None:
    """
    Extract all available data from MediaPipe Hand Landmarker.

    Args:
        image_path: Path to the image file.

    Returns:
        Dictionary with all MediaPipe outputs, or None if no hand detected.
        {
            "keypoints": [[x, y, z], ...],          # 21 points, image coordinates (normalized 0-1)
            "world_keypoints": [[x, y, z], ...],    # 21 points, world coordinates (meters)
            "handedness": "Left" or "Right",
            "handedness_confidence": float           # 0.0 to 1.0
        }
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    mp_image = mp.Image.create_from_file(str(image_path))

    with _build_landmarker() as landmarker:
        result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return None

    handedness = result.handedness[0][0]

    return {
        "keypoints": [[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[0]],
        "world_keypoints": [[lm.x, lm.y, lm.z] for lm in result.hand_world_landmarks[0]],
        "handedness": handedness.display_name,
        "handedness_confidence": float(handedness.score),
    }


def _build_video_landmarker() -> vision.HandLandmarker:
    """Initialize a MediaPipe HandLandmarker in VIDEO mode."""
    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def extract_keypoints_video(video_path: str | Path) -> dict | None:
    """
    Extract per-frame hand landmarks from a video using MediaPipe VIDEO mode.

    Args:
        video_path: Path to the video file.

    Returns:
        Dictionary with per-frame keypoints, or None if no hand detected
        in any frame.
        {
            "fps": float,
            "total_frames": int,
            "detected_frames": int,
            "keypoints": [[[x,y,z], ...21], ...N frames],
            "world_keypoints": [[[x,y,z], ...21], ...N frames],
        }
    """
    import cv2

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    keypoints_seq = []
    world_keypoints_seq = []

    with _build_video_landmarker() as landmarker:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=frame_rgb,
            )
            timestamp_ms = int(frame_idx * 1000 / fps)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                keypoints_seq.append(
                    [[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[0]]
                )
                world_keypoints_seq.append(
                    [[lm.x, lm.y, lm.z] for lm in result.hand_world_landmarks[0]]
                )

            frame_idx += 1

    cap.release()

    if not keypoints_seq:
        return None

    return {
        "fps": fps,
        "total_frames": total_frames,
        "detected_frames": len(keypoints_seq),
        "keypoints": keypoints_seq,
        "world_keypoints": world_keypoints_seq,
    }


def extract_keypoints_video_per_frame(
    video_path: str | Path,
) -> dict | None:
    """
    Extract per-frame hand landmarks across the entire video timeline,
    preserving frame positions where no hand was detected.

    Unlike extract_keypoints_video (which drops no-detection frames),
    this returns one slot per source frame so callers can segment on the
    no-hand gaps. Used by predict_full to split a multi-sign video into
    one segment per letter.

    Returns:
        {
            "fps": float,
            "total_frames": int,
            "detected_frames": int,
            "keypoints_per_frame": [ [[x,y,z], ...21]  or  None, ...total_frames ],
        }
    """
    import cv2

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    timeline: list[list | None] = []
    detected = 0

    with _build_video_landmarker() as landmarker:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=frame_rgb,
            )
            timestamp_ms = int(frame_idx * 1000 / fps)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                timeline.append(
                    [[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[0]]
                )
                detected += 1
            else:
                timeline.append(None)

            frame_idx += 1

    cap.release()

    if detected == 0:
        return None

    return {
        "fps": fps,
        "total_frames": frame_idx,
        "detected_frames": detected,
        "keypoints_per_frame": timeline,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        logger.error("Usage: python mediapipe_handler.py <image_path>")
        sys.exit(1)

    path = sys.argv[1]
    result = extract_keypoints(path)

    if result is None:
        logger.warning("No hand detected in image.")
    else:
        logger.success(f"Handedness: {result['handedness']} ({result['handedness_confidence']*100:.1f}%)")
        logger.info(f"Detected {len(result['keypoints'])} landmarks:")
        for i, (x, y, z) in enumerate(result['keypoints']):
            logger.info(f"  [{i:2d}] {LANDMARK_NAMES[i]:15s} x={x:.4f}  y={y:.4f}  z={z:.4f}")
        logger.info("World landmarks:")
        for i, (x, y, z) in enumerate(result['world_keypoints']):
            logger.info(f"  [{i:2d}] {LANDMARK_NAMES[i]:15s} x={x:.4f}  y={y:.4f}  z={z:.4f}")