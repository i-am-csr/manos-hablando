## 2026-05-13

### Data Pipeline
- Built `manos_hablando/data/mediapipe_handler.py` — modular keypoint extractor
- Built `manos_hablando/data/extract_keypoints.py` — processes full dataset and outputs `data/processed/keypoints.json`
- Built `manos_hablando/data/augment.py` — balances dataset to 100 images per letter using `albumentations`
- Removed J and Z from static dataset (dynamic signs requiring video)
- Final dataset: 2,057 samples across 26 LSM letters

### Model
- Designed and implemented `LSMTransformer` — Transformer-based classifier for hand sign recognition
- Input: 63 normalized keypoint coordinates (21 points × 3 coords)
- Output: letter classification across 26 LSM classes
- Training tracked with [MLflow](https://mlflow.org)
- First training run: **val_acc=80.26%** | **test_acc=78.64%** in 50 epochs

### Inference & Demo
- Built `manos_hablando/modeling/predict.py` — single image inference with top-k predictions
- Built `manos_hablando/app.py` — real-time Streamlit demo using `streamlit-webrtc`
- Features: live webcam feed, MediaPipe landmark overlay (21 points + skeleton), letter prediction with confidence score

### Notebooks & Docs
- `notebooks/1.0-cb-data-exploration.ipynb` — dataset analysis, class distribution, MediaPipe detection rates

---

## 2026-05-12

### Project Setup
- Initialized project structure using [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org)
- Configured `uv` as package manager, `ruff` for linting, `pytest` for testing, `mkdocs` for documentation

### Data Pipeline
- Tested MediaPipe Hand Landmarker on sample image

### Notebooks & Docs
- Created [reference notebook](notebooks/mediapipe_reference.ipynb) and [MediaPipe points map](docs/docs/mediapipe_points_mapped.md)