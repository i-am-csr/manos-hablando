# Video Pipeline — Dynamic Sign Classification

This document describes the end-to-end pipeline for classifying **dynamic LSM signs** (letters that require hand movement) from video.

## Dynamic Letters

| Letter | Why dynamic |
| ------ | ----------- |
| J | Pinky traces a downward arc |
| K | Fingers move apart |
| Ñ | Wrist twist motion |
| Q | Downward pointing motion |
| X | Index finger hooks inward |
| Z | Index finger traces a Z shape in the air |

These letters cannot be classified from a single static image — they require a **temporal sequence** of hand landmarks.

## Data Format

### Source

Videos are stored in `data/raw/{LETTER}/`, one folder per letter:

```
data/raw/
├── J/   → 102 videos (S1-J-frontal-1.mp4, ...)
├── K/   → 105 videos
├── Ñ/   → 107 videos
├── Q/   → 101 videos
├── X/   → 104 videos
└── Z/   → 103 videos
```

Naming convention: `S{subject}-{letter}-{view}-{take}.mp4`

### Processed

After keypoint extraction, video data is saved to `data/processed/video_keypoints.json`. Each entry represents one video:

```json
{
  "letter": "J",
  "file": "S1-J-frontal-1.mp4",
  "fps": 30.0,
  "total_frames": 62,
  "detected_frames": 61,
  "keypoints": [[[x,y,z], ...21 landmarks], ...N frames],
  "world_keypoints": [[[x,y,z], ...21 landmarks], ...N frames]
}
```

The keypoints array has shape `(N, 21, 3)` where N varies per video.

## Pipeline Steps

### 1. Extract keypoints from videos

Processes all videos in the dynamic letter folders using MediaPipe Hand Landmarker in VIDEO mode. Creates `data/processed/video_keypoints.json`.

```bash
uv run python -m manos_hablando.data.extract_keypoints_video 2>/dev/null
```

The `2>/dev/null` suppresses MediaPipe's internal C++ logging (TensorFlow Lite delegate messages).

### 2. Train the video model

Trains an `LSMVideoTransformer` on the extracted sequences. Logs metrics to MLflow under the `lsm-video-transformer` experiment. Saves the best checkpoint to `models/lsm_video_transformer.pt`.

```bash
uv run python -m manos_hablando.modeling.train_video 2>/dev/null
```

Default hyperparameters:

| Parameter | Default |
| --------- | ------- |
| epochs | 50 |
| batch_size | 16 |
| learning_rate | 1e-3 |
| d_model | 128 |
| nhead | 4 |
| num_layers | 3 |
| dropout | 0.1 |

Override any parameter via CLI flags:

```bash
uv run python -m manos_hablando.modeling.train_video --epochs 100 --learning-rate 5e-4 2>/dev/null
```

### 3. Predict on a video

Run inference on a single video file:

```bash
uv run python -m manos_hablando.modeling.predict_video path/to/video.mp4 2>/dev/null
```

Show top-k predictions:

```bash
uv run python -m manos_hablando.modeling.predict_video path/to/video.mp4 --top-k 3 2>/dev/null
```

## Architecture

### Static vs Video Transformer

| Aspect | LSMTransformer (static) | LSMVideoTransformer (video) |
| ------ | ----------------------- | --------------------------- |
| Input | `(batch, 63)` — single frame | `(batch, seq_len, 63)` — frame sequence |
| Sequence dim | Dummy (unsqueezed to 1) | Real temporal sequence |
| Positional encoding | Not meaningful | Encodes frame order |
| Pooling | Mean over 1 token | Masked mean over non-padded frames |
| Classes | 26 (static letters) | 6 (dynamic letters) |
| Checkpoint | `models/lsm_transformer.pt` | `models/lsm_video_transformer.pt` |

### Handling Variable-Length Sequences

Videos produce different numbers of detected frames. The pipeline handles this with:

1. **Padding:** Shorter sequences are zero-padded to the longest sequence in the batch.
2. **Padding mask:** A boolean mask marks padded positions so the Transformer ignores them.
3. **Masked mean pooling:** The classifier averages only over real frames, not padding.

## Project Files

| File | Role |
| ---- | ---- |
| `manos_hablando/data/mediapipe_handler.py` | MediaPipe wrapper — `extract_keypoints_video()` for VIDEO mode |
| `manos_hablando/data/extract_keypoints_video.py` | Batch extraction script for all dynamic letter folders |
| `manos_hablando/dataset_video.py` | Data loading, normalization, splitting, DataLoaders |
| `manos_hablando/modeling/train_video.py` | `LSMVideoTransformer` model + training loop |
| `manos_hablando/modeling/predict_video.py` | Inference on single video files |

## Normalization

The same wrist-centering strategy from the static pipeline is applied **per frame**:

1. Subtract wrist position (landmark 0) from all 21 landmarks
2. Divide by the max distance from wrist in that frame

This makes each frame invariant to hand position and scale in the image, while preserving the temporal motion pattern across frames.
