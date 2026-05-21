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