## 2026-06-03

### Raw-data restructure (`data/raw/letters/`, `data/raw/words/`)
- Moved the per-letter folders from `data/raw/{LETTER}/` to `data/raw/letters/{LETTER}/` to make room for a parallel `data/raw/words/` dataset of whole-word sign clips (42 word classes, ~50 videos each, ~2,100 videos total).
- `config.py`: added `RAW_LETTERS_DIR = RAW_DATA_DIR / "letters"` and `RAW_WORDS_DIR = RAW_DATA_DIR / "words"`; kept `RAW_DATA_DIR` pointing at `data/raw/` so future datasets can branch off the same root.
- Updated all three letter extractors (`extract_keypoints.py`, `extract_keypoints_video.py`, `extract_keypoints_full.py`) to default to `RAW_LETTERS_DIR`. CLAUDE.md and `docs/docs/video_pipeline.md` paths updated to match.

### MediaPipe holistic reference notebook + docs
- New `references/mediapipe_holistic_reference.ipynb` — replaces the deprecated `mediapipe.solutions.holistic` solution by running the three Tasks-API landmarkers (`FaceLandmarker`, `PoseLandmarker`, `HandLandmarker`) side by side. Walks through IMAGE and VIDEO modes, the combined per-frame feature layout (`pose 132 + face 1434 + hands 126 = 1692`), zero-padding for missing modalities, and integration notes (shoulder-centered normalization, handedness flip, mesh trimming).
- New `docs/docs/mediapipe_face_pose_mapped.md` — companion to the existing hand-landmarks doc. Documents the 478 face points (468 mesh + 10 iris) with canonical index lists for face oval, lips (outer/inner), eyes, eyebrows, and irises; the 52 blendshape coefficients grouped by region; and all 33 pose landmarks split into face (0-10), upper body (11-22, the ones LSM cares about), and lower body (23-32). Includes the combined-feature-vector table and pointers to download `face_landmarker.task` and `pose_landmarker.task`.

### Holistic words pipeline — 5 new files
- `manos_hablando/data/holistic_handler.py` — Tasks-API holistic extractor. Builds the three landmarkers in IMAGE or VIDEO mode, exposes `build_feature_vector(face, pose, hand)` → `(FEATURE_DIM=1692,) float32`, and `extract_holistic_video(path)` → per-frame tensor plus modality detection counts. `_check_models()` raises a clear download instruction when `face_landmarker.task` / `pose_landmarker.task` are missing from `models/`. Section offset constants (`POSE_OFFSET=0`, `FACE_OFFSET=132`, `LEFT_HAND_OFFSET=1566`, `RIGHT_HAND_OFFSET=1629`) plus shoulder indices (`LEFT_SHOULDER_IDX=11`, `RIGHT_SHOULDER_IDX=12`) live here so downstream normalization can address the right slots.
- `manos_hablando/data/extract_keypoints_holistic.py` — walks `data/raw/words/{Word}/*.mp4`, writes one `.npz` per video at `data/processed/holistic/{Word}/{stem}.npz` (single key `features`, shape `(N_frames, 1692)`) plus a `data/processed/holistic_keypoints.json` manifest indexing them. **Storage choice:** inline JSON like the letter pipelines would balloon to multi-GB and parse painfully slowly; npz + a small manifest keeps the bulk binary and the index readable.
- `manos_hablando/dataset_holistic.py` — manifest-driven loader. Lazily reads each `.npz`, reuses `LSMVideoDataset` + `collate_video_batch` from `dataset_video.py`. Exposes `get_holistic_dataloaders()` and `prepare_inference_sequence()`. First cut shipped **without normalization** — flagged as TODO; addressed the next day.
- `manos_hablando/modeling/train_holistic.py` — thin wrapper around `LSMVideoTransformer` with `input_dim=FEATURE_DIM` (1692) and a wider `d_model=256` default to absorb the bigger input. Default `batch_size=8` (vs 16 in `train_full`) because each sample is ~27× larger. MLflow experiment `lsm-holistic-transformer`; checkpoint `models/lsm_holistic_transformer.pt`.
- `manos_hablando/modeling/predict_holistic.py` — single video → top-K predicted word. **No segmentation, no LLM**: words are direct labels, the whole clip is classified once.
- CLAUDE.md: new "Holistic words pipeline" architecture section + four CLI lines (extract / dataset sanity / train / predict).

### Holistic v1 training → chance; shoulder-centered normalization added
- Extraction across `data/raw/words/`: 2,159 videos succeeded (sequence lengths min=133, max=179, mean=150.4 frames). The +52 over the 2,107 raw `.mp4` count comes from a handful of clips that produced a second extraction after a hand re-detection — investigate later.
- First training pass (50 epochs, ~6 min CPU): **collapsed to uniform predictions** — `train_loss` plateaued at 3.7356 (log(42) ≈ 3.7377), `val_acc` oscillated between 8/324 and 9/324, `test_acc=0.0247` (chance = 1/42 ≈ 0.0238).
- Root cause: features went straight from MediaPipe's 0-1 image-space into the model. The 1434-dim face block dominates the 126-dim hand block by sheer volume, and absolute screen positions depend on where the signer stood — there's no consistent signal across samples.
- Fix: `normalize_holistic_sequence(seq)` in `dataset_holistic.py`. Per-frame: anchor = midpoint of pose 11/12, scale = euclidean distance between them. Recenter all `x,y` (pose, face, both hands) on the anchor; divide all `x,y,z` by the scale. Leave pose `visibility` alone (it's already 0-1 semantic). Hands that weren't detected stay all-zero; frames where shoulders are missing get zeroed entirely. Verified on a real sample: post-norm shoulders sit exactly at `(±0.5, ≈0)`, shoulder distance `1.0`.
- Wired into both `get_holistic_dataloaders()` (training path) and `prepare_inference_sequence()` (predict path) so train and inference stay in sync. No re-extraction needed — normalization happens at load time.

---

## 2026-06-02

### Alternative-models notebook (`notebooks/alternative_models.ipynb`)
- Built the Week 5 / Activity 4 deliverable: 7 non-ensemble individual classifiers on the unified 29-letter dataset, with mean-pooled per-sample features `(63,)` so classical sklearn models can train on the same data the transformer sees.
- Pipeline: balanced subsample → 60/20/20 train/val/test split → `StandardScaler` + classifier, evaluated with F1-Macro as the primary metric.
- Baseline leaderboard (val): MLP **0.9425**, RandomForest **0.8971**, LogReg 0.8781, SVM-RBF 0.8768, KNN 0.8407, DecisionTree 0.7802, GaussianNB 0.6447.
- Hyperparameter tuning (RandomizedSearchCV, 20 combos) on the top two: MLP flat on val (−0.001) but **+0.0044 F1 on test**; RandomForest +0.006 on val, flat on test.
- **Final selection: tuned MLP** — `hidden_layer_sizes=(128, 64)`, `activation='tanh'`, `alpha=0.001`, `lr_init=0.001`. Test: **F1-Macro=0.9617, acc=95.93%** — within ~1.5 pts of `LSMVideoTransformer` (test_acc=97.5%) on mean-pooled features that throw away the temporal axis.

### Bulk-extraction speedup (`data/extract_keypoints_full.py`, `data/mediapipe_handler.py`)
- `extract_keypoints()` now accepts an optional pre-built `landmarker`; `extract_keypoints_full.py` builds **one** IMAGE-mode `HandLandmarker` and reuses it across all static images, amortizing the ~300 ms Metal/GL context warm-up that previously ran per image. Dynamic letters still build their own VIDEO-mode landmarker per clip.
- Added `DEFAULT_MAX_STATIC_SAMPLES = 1500` cap with deterministic `random.Random(SAMPLE_SEED=42)` subsampling — larger per-letter sets showed diminishing returns and inflated extraction time linearly. Cap is opt-out via `max_static_samples=None`.

### `predict_full.py` — per-user bias correction
- Added `--correct-biases` flag with a hardcoded substitution table `KNOWN_BIAS_SUBSTITUTIONS = {P→K, RR→R, LL→L, T→S}` for one signer's systematic confusions against the current MSL-ABC-trained checkpoint. Documented as **per-user, not universal** — re-derive after any retrain/fine-tune.
- Added `tokenize_word()` (greedy 2-then-1-char split honoring `MULTI_CHAR_LETTERS = {"LL", "RR"}`) and `apply_bias_corrections()` so multi-char classes survive the `LetterBuffer`'s "".join() concatenation before substitution.
- Applies corrections **before** the LLM call so the prompt sees the cleaned sequence; raw sequence is still logged for diagnostics.
- Rest of the file is `ruff format` cleanup (line wrapping, trailing newline) — no behavior change.

---

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