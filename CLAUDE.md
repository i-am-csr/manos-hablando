# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`manos-hablando` is an LSM (Lengua de Señas Mexicanas) → Spanish text translation system. It runs hand-sign recognition on a webcam stream and reconstructs whole Spanish words using an LLM. Project scaffolding follows [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org); package manager is `uv`; linter/formatter is `ruff`; experiment tracking is MLflow.

## Common commands

Dependency management uses `uv` (NOT pip/poetry). `uv sync` reads `pyproject.toml` + `uv.lock`. Python is pinned to `~=3.11.0`.

```bash
make requirements              # uv sync
make create_environment        # uv venv --python 3.11

make lint                      # ruff format --check && ruff check
make format                    # ruff check --fix && ruff format
make test                      # python -m pytest tests   (note: tests/ is currently a stub)
make clean                     # delete __pycache__ and *.pyc

# Run the Streamlit demo (this is the main user-facing entry point)
uv run streamlit run manos_hablando/app.py

# Browse MLflow runs (uses ./mlflow.db sqlite + ./mlruns artifacts)
uv run mlflow ui
```

CLI entry points use `typer`. Run modules with `-m`, not as scripts:

```bash
# Static pipeline (letters A–Y, single image)
uv run python -m manos_hablando.data.extract_keypoints       # raw images → data/processed/keypoints.json
uv run python -m manos_hablando.data.augment                 # balance to 100 imgs/letter via albumentations
uv run python -m manos_hablando.dataset                      # sanity-check load/normalize/split
uv run python -m manos_hablando.modeling.train               # train LSMTransformer
uv run python -m manos_hablando.modeling.predict <image.jpg> # inference with top-k

# Dynamic pipeline (letters J, K, Ñ, Q, X, Z, video)
uv run python -m manos_hablando.data.extract_keypoints_video 2>/dev/null   # raw videos → data/processed/video_keypoints.json
uv run python -m manos_hablando.modeling.train_video 2>/dev/null
uv run python -m manos_hablando.modeling.predict_video <video.mp4> 2>/dev/null

# Unified full pipeline (all 29 letters: A–Z + Ñ + LL + RR) — preferred for new work
uv run python -m manos_hablando.data.extract_keypoints_full 2>/dev/null    # raw → data/processed/full_keypoints.json
uv run python -m manos_hablando.dataset_full                               # sanity-check load + split
uv run python -m manos_hablando.modeling.train_full 2>/dev/null            # → models/lsm_full_transformer.pt
uv run python -m manos_hablando.modeling.predict_full <video.mp4> 2>/dev/null

# Holistic words pipeline (whole signed words from data/raw/words/, face + pose + hands features)
uv run python -m manos_hablando.data.extract_keypoints_holistic 2>/dev/null  # raw → data/processed/holistic/*.npz + holistic_keypoints.json
uv run python -m manos_hablando.dataset_holistic                             # sanity-check load + split
uv run python -m manos_hablando.modeling.train_holistic 2>/dev/null          # → models/lsm_holistic_transformer.pt
uv run python -m manos_hablando.modeling.predict_holistic <video.mp4> 2>/dev/null
```

The `2>/dev/null` for video commands suppresses noisy MediaPipe/TFLite C++ logging — keep it when documenting examples.

### Running `predict_full` (the multi-sign analyzer)

Takes a single video that may contain multiple fingerspelled signs, segments it, classifies each segment with `LSMVideoTransformer`, then pipes the letter sequence to `LLMProcessor` for Spanish reconstruction.

```bash
# Default: sliding-window mode, LLM reconstruction on
uv run python -m manos_hablando.modeling.predict_full clip.mp4 2>/dev/null

# Just the raw letter sequence, no LLM call
uv run python -m manos_hablando.modeling.predict_full clip.mp4 --no-llm 2>/dev/null

# Fast natural fingerspelling: slow the timeline 3× and tell the LLM what to expect
uv run python -m manos_hablando.modeling.predict_full clip.mp4 \
    --slowdown 3 \
    --context "a Spanish person's first name followed by paternal and maternal surnames" \
    2>/dev/null

# Deliberate-pause fingerspelling (user lowers hand between letters)
uv run python -m manos_hablando.modeling.predict_full clip.mp4 --mode gaps 2>/dev/null

# Diagnostic view: print every per-window prediction
uv run python -m manos_hablando.modeling.predict_full clip.mp4 --verbose --no-llm 2>/dev/null
```

All flags:

| Flag | Default | Purpose |
| ---- | ------- | ------- |
| `--mode sliding\|gaps` | `sliding` | Segmentation strategy. `sliding` debounces a per-frame classifier — robust to fluid signing. `gaps` cuts at no-hand pauses — cleanest when the user deliberately lowers their hand. |
| `--slowdown N` | `1` | Replicate each timeline frame N times. Use 3-4 for natural fingerspelling (~10 frames/letter). Beyond 5× the transitions become long enough to be debounced as spurious letters. |
| `--context "..."` | `""` | Hint passed to the LLM about what was signed (e.g. `"a Spanish person's name"`, `"a common Spanish noun"`). Without it the LLM defaults to dictionary vocabulary and will avoid proper nouns. |
| `--no-llm` | off | Skip the LLM call entirely. Useful when iterating on segmentation knobs. |
| `--verbose` | off | (sliding only) Print the per-step event log: frame index, time, top-1 letter, confidence, status. |
| `--window-frames N` | `30` | (sliding) Frames per classifier window. 30 ≈ 1s at 30fps and matches the static replication length. |
| `--stride N` | `3` | (sliding) Step between window evaluations, in frames. |
| `--stability N` | `4` | (sliding) Consecutive matching predictions required to commit a letter. At stride=3, 4 → ~12 frames of agreement. |
| `--confidence X` | `0.5` | (sliding) Minimum confidence for a prediction to count toward stability. |
| `--word-boundary N` | `15` | (sliding) No-hand frames that close a word and flush the buffer. |
| `--gap-frames N` | `8` | (gaps) Consecutive no-hand frames that close a sign segment. |
| `--min-segment N` | `6` | (gaps) Minimum frames in a segment; shorter ones are dropped as noise. |
| `--min-confidence X` | `0.4` | (gaps) Drop segment predictions below this confidence. |
| `--top-k N` | `3` | (gaps) Top-k candidates shown per segment in the report table. |

## Architecture

The codebase contains **three pipelines** that share normalization but have separate datasets, models, and checkpoints — all following the shape `extract_keypoints → dataset (load/normalize/split/DataLoader) → modeling (train/predict)`:

1. **Static** (24 letters A–Y, single images) — original baseline.
2. **Video / dynamic** (6 letters J/K/Ñ/Q/X/Z, multi-frame clips) — for letters that require motion.
3. **Full / unified** (29 letters A–Z + Ñ + LL + RR) — newer, replaces the two above for inference. The Streamlit app still uses 1 and 2 separately; new work should target 3.

### Shared foundation

- **`data/mediapipe_handler.py`** is the only place that talks to MediaPipe Hand Landmarker. It exposes `extract_keypoints()` for single images (IMAGE mode) and `extract_keypoints_video()` for videos (VIDEO mode with monotonically increasing timestamps `int(frame_idx * 1000 / fps)`). Both load the model from `models/hand_landmarker.task`. Output is always 21 landmarks × (x, y, z), each normalized to 0-1 image space.
- **Normalization** is wrist-centering: subtract landmark 0 (wrist) from all 21 points, divide by max distance from wrist. Implemented once in `dataset.normalize_keypoints(X: (N, 63))`. The video pipeline reuses it per-frame via `dataset_video.normalize_video_sequences()`. Output shape and dtype must stay `(N, 63) float32` — downstream PyTorch code depends on this.
- **`config.py`** computes all path constants from `PROJ_ROOT = Path(__file__).resolve().parents[1]` and calls `load_dotenv()`. Always import paths from there rather than reconstructing them.

### Static pipeline (24 letters, A–Y except J and Z)

- Dataset: `data/raw/letters/{LETTER}/*.jpg|png` → `data/processed/keypoints.json` (one entry per image, `keypoints` shape `(21, 3)`).
- Model: `LSMTransformer` in `modeling/train.py` — Transformer encoder over a **dummy** sequence of length 1 (input is unsqueezed to `(batch, 1, 63)`). The transformer architecture is mostly a stand-in for an MLP here; the temporal dimension is meaningless.
- Checkpoint: `models/lsm_transformer.pt`. MLflow experiment: `lsm-transformer`.

### Video / dynamic pipeline (6 letters: J, K, Ñ, Q, X, Z)

- Dataset: `data/raw/letters/{LETTER}/*.mp4` (naming `S{subject}-{letter}-{view}-{take}.mp4`) → `data/processed/video_keypoints.json`. Each entry has `keypoints` of shape `(N_frames, 21, 3)`, variable N per video.
- `dataset_video.py`: variable-length sequences are kept as a `list[np.ndarray]` until the DataLoader. The custom `collate_video_batch` zero-pads to the longest sequence in the batch and returns `(X_padded, y, lengths)`.
- Model: `LSMVideoTransformer` in `modeling/train_video.py` — same building blocks but takes a **real** temporal sequence `(batch, seq_len, 63)` and accepts an `src_key_padding_mask`. Pooling is **masked mean** over non-padded frames; do NOT replace with plain `.mean(dim=1)` or padding will contaminate the embedding.
- The set of dynamic letters is hardcoded as `DYNAMIC_LETTERS = {"J", "K", "Ñ", "Q", "X", "Z"}` in `data/extract_keypoints_video.py`. Keep this in sync with `data/raw/letters/`.
- Checkpoint: `models/lsm_video_transformer.pt`. MLflow experiment: `lsm-video-transformer`.

### Full / unified pipeline (29 letters: A–Z + Ñ + LL + RR)

- Dataset: every letter folder under `data/raw/letters/` — static letter folders contribute images, the 6 dynamic letters contribute videos. `extract_keypoints_full.py` routes each folder to the right MediaPipe mode and emits a single `data/processed/full_keypoints.json` where every entry has shape `(N_frames, 21, 3)`. Static letters have `N=1`; dynamic letters have `N=20-100+`. Each entry carries a `source` field (`"image"` or `"video"`) so downstream code can branch on it.
- `dataset_full.py`: at load time, single-frame entries (static letters) are replicated to `STATIC_REPLICATE_LENGTH = 30` via `np.repeat`. This is **load-bearing** — it closes the train/inference distribution gap so that a real video of someone holding "A" for ~30 frames matches what the model saw in training. Reuses `LSMVideoDataset` + `collate_video_batch` + `normalize_video_sequences` from `dataset_video.py`. Also exposes `prepare_inference_sequence()` used by `predict_full.py`.
- Model: reuses `LSMVideoTransformer` from `train_video.py` unchanged — the architecture and training loop are identical, only the data and class count differ. `train_full.py` is a thin wrapper.
- Checkpoint: `models/lsm_full_transformer.pt`. MLflow experiment: `lsm-full-transformer`. Current run: `val_acc=90.2%`, `test_acc=87.2%` (on the isolated training distribution — natural connected fingerspelling is substantially harder).
- Inference (`predict_full.py`):
  - Uses `extract_keypoints_video_per_frame()` (a new helper in `mediapipe_handler.py`) which preserves no-hand positions in the timeline so segmentation has gaps to cut on.
  - Two modes:
    - **sliding** (default): walks the timeline at `stride` step, builds a trailing `window_frames` window of detected frames, classifies the window, feeds the top prediction into `LetterBuffer`. Words flush when the buffer's `word_complete()` fires (on `word_boundary` no-hand frames).
    - **gaps**: walks the timeline once, cuts segments at runs of `gap_frames` no-hand frames, classifies each segment.
  - `--slowdown N` replicates each timeline entry N times before analysis. At N=3-4 a 10-frame natural held pose becomes 30-40 effective frames, matching what `STATIC_REPLICATE_LENGTH=30` trained the model to see. **Diminishing returns at 4×, harmful at 5×+** because letter transitions also get stretched and start getting debounced as spurious letters (the U/V family is most prone to this).
  - Known systematic confusions in the trained model — useful for prompt design: closed-fist family (A↔M↔N↔S↔T) and pointing-finger family (R↔V↔U↔K). These are documented in `SYSTEM_PROMPT` so the LLM can compensate.

### Holistic words pipeline (whole signed words, face + pose + hands)

- **Purpose:** classify whole Spanish words signed in LSM (not letter-by-letter spelling), like `Abrir`, `Bailar`, `Comer`. Lives entirely separately from the letter pipelines — different data, different MediaPipe stack, different model checkpoint.
- **Replaces deprecated `mediapipe.solutions.holistic`** by running the three Tasks-API landmarkers side-by-side: `FaceLandmarker`, `PoseLandmarker`, `HandLandmarker`. Per-frame feature is the concatenation `pose(33×4) + face(478×3) + left_hand(21×3) + right_hand(21×3) = 1692` dims; missing detections become zero-padded slots. Constants live in `data/holistic_handler.py` (`FEATURE_DIM`, `POSE_DIMS`, etc.).
- **Requires two extra MediaPipe model files** not currently in `models/`: `face_landmarker.task` and `pose_landmarker.task`. `holistic_handler._check_models()` raises a clear download instruction if either is missing. URLs are in the docstring and in `references/mediapipe_holistic_reference.ipynb`.
- Dataset: every word folder under `data/raw/words/` (e.g. `Abrir/`, `Bailar/`) → one `.npz` per video at `data/processed/holistic/{Word}/{file_stem}.npz` (key `features`, shape `(N_frames, 1692)`), plus a single `data/processed/holistic_keypoints.json` manifest indexing them. **JSON-only storage like the letter pipelines would balloon to ~GB-scale** because of the 1692-dim per-frame features — npz keeps the manifest readable and the bulk binary.
- `dataset_holistic.py`: walks the manifest, lazily loads each `.npz`, reuses `LSMVideoDataset` + `collate_video_batch` from `dataset_video.py`. **No normalization is currently applied** — features go straight from MediaPipe's image-normalized 0-1 space into the model. Adding shoulder-centered normalization (anchor on pose 11/12 midpoint) is the natural next step.
- Model: reuses `LSMVideoTransformer` from `train_video.py` unchanged, just with `input_dim=FEATURE_DIM` (1692) instead of 63 and a larger `d_model=256` default to absorb the wider input. Checkpoint: `models/lsm_holistic_transformer.pt`. MLflow experiment: `lsm-holistic-transformer`.
- Inference (`predict_holistic.py`): assumes **one signed word per video** — no segmentation, no LLM call. The whole clip's features are classified once and top-K candidates are printed.

### Streamlit app (`manos_hablando/app.py`)

The app is the integration point. It uses `streamlit-webrtc` for live webcam capture, with separate `VideoProcessorBase` subclasses per mode selected from the sidebar:

- **Static mode (`StaticSignProcessor`)**: runs `extract_keypoints` + `LSMTransformer` every 3rd frame, feeds the predicted letter+confidence into a `LetterBuffer`. When the buffer reports `word_complete()` (after `word_boundary_frames` of no-hand), the letters are sent to `LLMProcessor.reconstruct()` and appended to the running sentence.
- **Dynamic mode (`DynamicSignProcessor`)**: arms on a "Record" button click, buffers per-frame keypoints for `RECORD_SECONDS = 3`, then runs `LSMVideoTransformer` once on the captured sequence.

The processors run on a streamlit-webrtc worker thread, so any shared mutable state (`frame_buffer`, `prediction`, `LetterBuffer`) must be accessed under `self.lock` / via the buffer's internal lock. Model loading is wrapped in `@st.cache_resource`.

### text_processing package

- **`LetterBuffer`** (`buffer.py`) — thread-safe debouncer. A letter must be predicted with confidence ≥ `confidence_threshold` for `stability_window` consecutive frames before it commits to the word. Low-confidence frames break the streak but don't reset the no-hand counter. `add_no_hand()` advances the word-boundary timer; once `word_boundary_frames` consecutive no-hand frames accumulate, `word_complete()` flips to True.
- **`LLMProcessor`** (`llm_processor.py`) — wraps `langchain.chat_models.init_chat_model` (default `provider=openai`, `model=gpt-4o`). Reconstructs Spanish content (words, phrases, **and proper nouns**) from noisy fingerspelled letter sequences via a hardcoded `SYSTEM_PROMPT` that documents the model's known confusion families. `reconstruct(letters, context=...)` accepts an optional context hint (e.g. `"a Spanish person's name"`) that is injected into the human message — without it the LLM defaults to dictionary vocabulary and won't reach for proper nouns. **Always falls back** to the raw concatenated letters on missing `OPENAI_API_KEY`, empty response, or any exception — never raises. Exposes `last_status` (`idle`/`calling`/`success`/`fallback`/`no_key`) and timing/error fields for the UI to display.
- **`truststore.inject_into_ssl()`** is called at import time in `llm_processor.py` **before any other SSL-using import**. This swaps `certifi`'s CA bundle for the OS trust store. It is required when running behind corporate SSL inspection (e.g. Cisco Secure Access) where the proxy CA is in the OS keychain but not in certifi. Do not move or remove this — preserving the import order is load-bearing.

## Conventions & gotchas

- **`uv` only.** Do not invoke `pip install` or add `requirements.txt` files. Add dependencies via `uv add <pkg>`; they belong in `pyproject.toml` `[project].dependencies`, and `uv.lock` is committed.
- **Ruff config** (in `pyproject.toml`: line length 99, import sorting on (`extend-select = ["I"]`), `manos_hablando` is known-first-party, and `force-sort-within-sections = true`. Run `make format` before committing.
- **`data/`, `mlruns/`, and `mlflow.db` are gitignored**, along with `models/*.pt` and `models/hand_landmarker.task`. Do not commit dataset files, trained checkpoints, the MediaPipe model, or MLflow's local SQLite run index. `mlflow.db` is gitignored because GitHub's push-protection scanner produces false-positive "Lob Test API Key" matches on the `test_acc<run_uuid>` byte sequences MLflow writes when logging metrics — and binary DBs grow noisily anyway.
- **OpenAI key** lives in `.env` as `OPENAI_API_KEY=...` (loaded by `config.py` via `python-dotenv`). The `.env` file is gitignored.
- **Logging**: use `loguru.logger` (NOT stdlib `logging`). `config.py` already wires loguru to play nicely with `tqdm`.
- **CLI**: every script uses `typer.Typer()` with `@app.command()` and `if __name__ == "__main__": app()`. Follow this pattern when adding new entry points.
- **`tests/test_data.py` is currently a stub** (`assert False`). `make test` will fail until tests are written — don't interpret a failing test run as a regression you introduced.
- **MLflow tracking** uses the default file-based backend rooted at the project (`mlflow.db` + `./mlruns/`). Running `mlflow ui` without arguments from the project root picks these up automatically.
- **Notebooks** in `notebooks/` follow the CCDS naming convention (`{order}-{initials}-{slug}.ipynb`); `references/` holds reference notebooks and sample media (`hand.jpg`, `video-J.mp4`) used by docs and ad-hoc experimentation.