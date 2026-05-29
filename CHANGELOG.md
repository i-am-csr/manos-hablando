## 2026-05-29

### Unified Full Pipeline (29 letters)
- Built `manos_hablando/data/extract_keypoints_full.py` — routes static letter folders → IMAGE mode and dynamic folders (J, K, Ñ, Q, X, Z) → VIDEO mode, emits a single `data/processed/full_keypoints.json` (static letters as 1-frame entries, dynamic as N-frame). 2450 samples extracted, 472 failed.
- Built `manos_hablando/dataset_full.py` — replicates single-frame static entries to 30 frames at load time (closes train/inference distribution gap), reuses `LSMVideoDataset` + `collate_video_batch` from the video pipeline. Exposes `prepare_inference_sequence()` helper for predict.
- Built `manos_hablando/modeling/train_full.py` — wraps `LSMVideoTransformer` + `train_epoch`/`evaluate` from the video pipeline against the unified loader. New MLflow experiment `lsm-full-transformer`, checkpoint `models/lsm_full_transformer.pt`.
- First training run: 29 classes (A–Z + Ñ + LL + RR), **best val_acc=90.2%** | **test_acc=87.2%** in 50 epochs

### Multi-Sign Video Analyzer (`predict_full.py`)
- Takes a single video with multiple fingerspelled signs and returns the reconstructed sentence
- Two segmentation modes:
  - `sliding` (default): trailing-window classifier + `LetterBuffer` debouncer — handles natural fluid signing without hand-down pauses
  - `gaps`: cuts segments at runs of no-hand frames — cleanest when user deliberately pauses between signs
- Added `extract_keypoints_video_per_frame()` to `data/mediapipe_handler.py` — preserves no-hand positions in the timeline so segmentation has something to cut on
- `--slowdown N` flag — replicates each timeline frame N times. At N=3 a 10-frame natural held pose becomes 30 effective frames, matching the static training distribution. Sweet spot 3-4×; beyond that, transitions get debounced as spurious letters
- `--context "..."` flag — injects a hint into the LLM call (e.g. "a Spanish person's name")
- `--verbose` flag — prints the per-step event log with per-window predictions and confidence

### LLM Prompt Overhaul
- Rewrote `SYSTEM_PROMPT` in `text_processing/llm_processor.py`:
  - Now explicitly allows proper nouns (names, places) as valid output — no longer biased toward dictionary words
  - Documents the systematic recognition confusions observed in practice: closed-fist family (A↔M↔N↔S↔T) and pointing-finger family (R↔V↔U↔K)
  - Supports an optional context hint via `LLMProcessor.reconstruct(letters, context=...)`
- Empirical result on `references/spelling.mov` (truth: KARLA MERINO ANGELES): old prompt → `REALIZARSE`; new prompt no-context → `García Reyes`; new prompt with name context → structured Mexican 3-part names

### Docs
- New top-level `CLAUDE.md` with project overview, pipeline architecture, and CLI command reference (including all `predict_full` flags)

---

## 2026-05-25

### Baseline Notebook
- Built `notebooks/baseline.ipynb` — establishes reference performance for the static pipeline
- Models compared: DummyClassifier (random-chance floor), Logistic Regression, Random Forest (primary baseline)
- Per-class F1, confusion matrix, learning curves, and feature importance by finger/coordinate
- Random Forest baseline: **test_acc=76.7%** (vs LSMTransformer 78.6%)

### Dynamic Sign Pipeline (J, K, Ñ, Q, X, Z)
- Migrated raw dataset: cleared static images for K/Q/X and added video folders for J, Ñ, Z — **622 videos total** from MSL-dynamic-signs (S1–S21, frontal view)
- Added `extract_keypoints_video()` to `data/mediapipe_handler.py` — VIDEO-mode landmark extraction with monotonic timestamps
- Built `manos_hablando/data/extract_keypoints_video.py` — batch pipeline → `data/processed/video_keypoints.json`
- Built `manos_hablando/dataset_video.py` — per-frame normalization, padded variable-length sequences, masked DataLoaders
- Built `manos_hablando/modeling/train_video.py` — `LSMVideoTransformer` with real temporal sequence dimension and masked mean pooling
- Built `manos_hablando/modeling/predict_video.py` — inference on single video files
- Reference notebook: `references/mediapipe_video_reference.ipynb` (uses `references/video-J.mp4`)
- Docs: `docs/docs/mediapipe_video_processing.md`, `docs/docs/video_pipeline.md`

### Streamlit App Enhancements
- Added sidebar mode toggle: **Static Signs (A-Y)** vs **Dynamic Signs (J, K, Ñ, Q, X, Z)**
- Dynamic mode: Record button + 3-second capture window + LSMVideoTransformer prediction overlay
- Static mode: live fingerspelling buffer overlaid on the video

### Fingerspelling → LLM Reconstruction
- New package `manos_hablando/text_processing/`:
  - `buffer.py` — `LetterBuffer` with debouncing (15-frame stability window, 0.7 confidence threshold, 30-frame word boundary). Thread-safe via `threading.Lock`.
  - `llm_processor.py` — `LLMProcessor` using **LangChain** (`init_chat_model`) with OpenAI provider. Falls back to raw letters on any error.
- Loads `OPENAI_API_KEY` via `python-dotenv` (added entry in `.env`)
- App integration: live letter buffer, last LLM call status (🟢 SUCCESS / 🔴 FALLBACK / ⚪ NO KEY) with latency and error details, accumulating sentence text area, Clear button
- Added `truststore` dependency — fixes SSL `CERTIFICATE_VERIFY_FAILED` when running behind corporate proxies (e.g. Cisco Secure Access) by delegating cert verification to the OS trust store

### Dependencies
- Added: `openai`, `langchain`, `langchain-openai`, `truststore`

---

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