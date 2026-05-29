# MediaPipe Hand Landmarker — Video Processing

MediaPipe's Hand Landmarker supports three running modes. This document explains how **VIDEO mode** works and how it differs from single-image detection.

## Running Modes

| Mode | Method | Use Case |
| ---- | ------ | -------- |
| `IMAGE` | `detect()` | Single image, no temporal context |
| `VIDEO` | `detect_for_video()` | Pre-recorded video, frame-by-frame |
| `LIVE_STREAM` | `detect_async()` | Real-time webcam, callback-based |

## VIDEO Mode

In VIDEO mode, MediaPipe processes frames **sequentially** and uses temporal information from previous frames to improve tracking consistency. This is the mode used for processing pre-recorded sign language videos.

### Key Differences from IMAGE Mode

| Aspect | IMAGE | VIDEO |
| ------ | ----- | ----- |
| Running mode | `vision.RunningMode.IMAGE` | `vision.RunningMode.VIDEO` |
| Detection method | `detect(mp_image)` | `detect_for_video(mp_image, timestamp_ms)` |
| Timestamp required | No | Yes (monotonically increasing) |
| Temporal smoothing | None | Uses previous frames for tracking |
| Frame order | Any | Must be chronological |

### How It Works

1. **Frame extraction:** Read each frame from the video using OpenCV.
2. **Color conversion:** Convert BGR (OpenCV default) to RGB (MediaPipe requirement).
3. **Wrap as MediaPipe Image:** Create an `mp.Image` object from the RGB array.
4. **Detect with timestamp:** Call `detect_for_video(mp_image, timestamp_ms)` with a monotonically increasing timestamp derived from the frame index and FPS.
5. **Collect results:** Each call returns a `HandLandmarkerResult` with landmarks, handedness, and world landmarks (or empty lists if no hand is detected).

### Timestamps

VIDEO mode requires a timestamp in milliseconds for each frame. The timestamp must be **strictly increasing** — passing a timestamp equal to or less than the previous one will cause an error.

The timestamp is computed from the frame index and the video's FPS:

```
timestamp_ms = int(frame_index * 1000 / fps)
```

### Temporal Tracking

Unlike IMAGE mode (which runs full hand detection on every frame), VIDEO mode uses a **detection + tracking** pipeline:

- **Detection** runs on the first frame and whenever tracking is lost.
- **Tracking** uses the previous frame's landmarks to predict where the hand is in the current frame, which is faster and produces smoother results.
- The `min_tracking_confidence` parameter (default 0.5) controls when tracking falls back to full detection. Lower values keep tracking longer; higher values trigger re-detection more often.

## Configuration

```python
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path="models/hand_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
```

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `num_hands` | 1 | Maximum number of hands to detect per frame |
| `min_hand_detection_confidence` | 0.5 | Minimum confidence for the hand detection model |
| `min_hand_presence_confidence` | 0.5 | Minimum confidence for hand presence in the frame |
| `min_tracking_confidence` | 0.5 | Minimum confidence to keep tracking (vs re-detecting) |

## Processing Loop

```python
import cv2
import mediapipe as mp

cap = cv2.VideoCapture("video.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)

with vision.HandLandmarker.create_from_options(options) as landmarker:
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(frame_idx * 1000 / fps)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]       # 21 NormalizedLandmarks
            world = result.hand_world_landmarks[0]      # 21 Landmarks (meters)
            handedness = result.handedness[0][0]         # Category

        frame_idx += 1

cap.release()
```

## Output Format

Each frame produces a `HandLandmarkerResult` with the same structure as IMAGE mode:

- **`hand_landmarks`** — 21 normalized landmarks (x, y, z in 0-1 image space)
- **`hand_world_landmarks`** — 21 landmarks in real-world meters
- **`handedness`** — Left or Right with confidence score

For a complete description of the 21 landmarks and their anatomical names, see [MediaPipe Hand Landmarks](mediapipe_points_mapped.md).

### Video-Specific Output Shape

For a video with `N` detected frames, the extracted keypoints form a sequence:

| Data | Shape | Description |
| ---- | ----- | ----------- |
| Image keypoints | `(N, 21, 3)` | Normalized coordinates per frame |
| World keypoints | `(N, 21, 3)` | Metric coordinates per frame |

This temporal sequence is what enables classification of **dynamic signs** (like J and Z) that require motion to distinguish from static poses.

## Official Documentation

- [Hand Landmarker overview](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
- [Python guide](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python)
