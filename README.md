# manos-hablando

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

**Manos Hablando** ("Speaking Hands") is an LSM (Lengua de Señas Mexicanas / Mexican Sign Language) fingerspelling-to-text translation system. It uses [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) to extract hand landmarks from a webcam stream or video file, classifies each signed letter with a Transformer model, and then feeds the raw letter sequence to an LLM (GPT-4o) to reconstruct complete Spanish words.

---

## Features

- Real-time webcam sign recognition via a **Streamlit** web app.
- Supports **29 LSM letters**: A–Z + Ñ + LL + RR (static poses and motion-based signs).
- LLM-powered word reconstruction that compensates for classifier confusions using context hints.
- Three recognition pipelines: static (images), dynamic (video), and a unified full pipeline.
- Experiment tracking with **MLflow**.

---

## Prerequisites

- Python **3.11** (pinned; other versions are untested)
- [`uv`](https://docs.astral.sh/uv/) package manager
- An **OpenAI API key** (for LLM word reconstruction; falls back to raw letters without it)
- The MediaPipe Hand Landmarker model file placed at `models/hand_landmarker.task`
  (download from [Google's MediaPipe Models](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker))

---

## Installation

```bash
# 1. Create a virtual environment (Python 3.11)
make create_environment
# or: uv venv --python 3.11

# 2. Activate the environment
source .venv/bin/activate       # Unix/macOS
# .\.venv\Scripts\activate      # Windows

# 3. Install all dependencies
make requirements
# or: uv sync
```

### Environment variables

Create a `.env` file in the project root (it is gitignored):

```
OPENAI_API_KEY=sk-...
```

---

## Running the Streamlit app

The Streamlit app is the main user-facing interface. It streams webcam video, recognizes signed letters in real time, and reconstructs Spanish words.

```bash
uv run streamlit run manos_hablando/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser. Use the sidebar to switch between **Static** mode (letters A–Y) and **Dynamic** mode (letters J, K, Ñ, Q, X, Z).

---

## CLI pipelines

### Full / unified pipeline (recommended — 29 letters)

```bash
# 1. Extract keypoints from raw data
uv run python -m manos_hablando.data.extract_keypoints_full 2>/dev/null

# 2. Sanity-check the dataset
uv run python -m manos_hablando.dataset_full

# 3. Train the model
uv run python -m manos_hablando.modeling.train_full 2>/dev/null

# 4. Run inference on a video
uv run python -m manos_hablando.modeling.predict_full <video.mp4> 2>/dev/null
```

#### `predict_full` options

| Flag | Default | Description |
|------|---------|-------------|
| `--mode sliding\|gaps` | `sliding` | Segmentation strategy. `sliding` uses a per-frame debounced classifier; `gaps` cuts at no-hand pauses. |
| `--slowdown N` | `1` | Replicate each frame N times (use 3–4 for natural fingerspelling). |
| `--context "..."` | `""` | Hint to the LLM (e.g. `"a Spanish person's name"`). Without it, proper nouns may be missed. |
| `--no-llm` | off | Skip the LLM call; return the raw letter sequence. |
| `--verbose` | off | Print per-step event log (sliding mode only). |
| `--window-frames N` | `30` | Frames per classifier window (sliding). |
| `--stride N` | `3` | Step between window evaluations in frames (sliding). |
| `--stability N` | `4` | Consecutive matching predictions required to commit a letter (sliding). |
| `--confidence X` | `0.5` | Minimum confidence to count toward stability (sliding). |
| `--word-boundary N` | `15` | No-hand frames that flush the word buffer (sliding). |
| `--gap-frames N` | `8` | Consecutive no-hand frames that close a segment (gaps). |
| `--min-segment N` | `6` | Minimum frames per segment; shorter ones are dropped as noise (gaps). |
| `--min-confidence X` | `0.4` | Drop segment predictions below this confidence (gaps). |
| `--top-k N` | `3` | Top-k candidates shown per segment (gaps). |

### Static pipeline (A–Y, single images)

```bash
uv run python -m manos_hablando.data.extract_keypoints
uv run python -m manos_hablando.dataset
uv run python -m manos_hablando.modeling.train
uv run python -m manos_hablando.modeling.predict <image.jpg>
```

### Dynamic pipeline (J, K, Ñ, Q, X, Z — motion signs)

```bash
uv run python -m manos_hablando.data.extract_keypoints_video 2>/dev/null
uv run python -m manos_hablando.modeling.train_video 2>/dev/null
uv run python -m manos_hablando.modeling.predict_video <video.mp4> 2>/dev/null
```

---

## Development

```bash
make lint      # Check formatting and linting (ruff)
make format    # Auto-fix and format (ruff)
make test      # Run tests (pytest)
make clean     # Remove __pycache__ and *.pyc files
```

Browse MLflow experiment runs:

```bash
uv run mlflow ui
```

---

## Architecture

The project contains **three recognition pipelines** that share the same MediaPipe-based keypoint extraction and wrist-centering normalization:

| Pipeline | Letters | Input | Model checkpoint |
|----------|---------|-------|-----------------|
| **Static** | 24 (A–Y, no J/Z) | Images | `models/lsm_transformer.pt` |
| **Dynamic** | 6 (J, K, Ñ, Q, X, Z) | Video clips | `models/lsm_video_transformer.pt` |
| **Full / unified** | 29 (A–Z + Ñ + LL + RR) | Images + videos | `models/lsm_full_transformer.pt` |

All pipelines follow the same shape: `extract_keypoints → dataset → modeling (train / predict)`.

### Key components

- **`data/mediapipe_handler.py`** — single interface to MediaPipe Hand Landmarker; outputs 21 landmarks × (x, y, z) normalized to image space.
- **`dataset*.py`** — load, normalize (wrist-centering), and split data; static images are replicated to 30 frames to match video-model expectations.
- **`modeling/`** — `LSMTransformer` (static) and `LSMVideoTransformer` (video/full) Transformer encoder models.
- **`text_processing/buffer.py`** — thread-safe `LetterBuffer` that debounces noisy per-frame predictions.
- **`text_processing/llm_processor.py`** — wraps LangChain + GPT-4o to reconstruct Spanish words from letter sequences; always falls back to raw letters on missing API key or errors.
- **`app.py`** — Streamlit web app integrating all components with live webcam capture via `streamlit-webrtc`.

---

## Project Organization

```
├── LICENSE
├── Makefile               <- Convenience commands (make requirements, make lint, …)
├── README.md
├── pyproject.toml         <- Project metadata and tool configuration (ruff, uv)
├── uv.lock                <- Locked dependency versions
│
├── data/                  <- (gitignored) Datasets
│   ├── raw/               <- Original images and videos per letter
│   └── processed/         <- Extracted keypoints JSON files
│
├── docs/                  <- MkDocs documentation source
├── models/                <- (gitignored) Trained model checkpoints and MediaPipe task file
├── notebooks/             <- Jupyter notebooks (CCDS naming: {order}-{initials}-{slug}.ipynb)
├── references/            <- Reference notebooks and sample media
├── reports/               <- Generated figures and analysis outputs
│   └── figures/
│
└── manos_hablando/        <- Python package (source code)
    ├── app.py             <- Streamlit web app (main entry point)
    ├── config.py          <- Path constants and environment variable loading
    ├── dataset.py         <- Static pipeline: load/normalize/split
    ├── dataset_video.py   <- Dynamic pipeline: variable-length sequence handling
    ├── dataset_full.py    <- Full pipeline: unified load + static replication
    ├── features.py        <- Feature utilities
    ├── plots.py           <- Visualization helpers
    ├── webcam.py          <- Webcam utilities
    ├── data/
    │   ├── mediapipe_handler.py         <- MediaPipe Hand Landmarker interface
    │   ├── extract_keypoints.py         <- Static: images → keypoints.json
    │   ├── extract_keypoints_video.py   <- Dynamic: videos → video_keypoints.json
    │   └── extract_keypoints_full.py    <- Full: all letters → full_keypoints.json
    ├── modeling/
    │   ├── train.py           <- Train LSMTransformer (static)
    │   ├── predict.py         <- Inference with LSMTransformer
    │   ├── train_video.py     <- Train LSMVideoTransformer (dynamic)
    │   ├── predict_video.py   <- Inference with LSMVideoTransformer
    │   ├── train_full.py      <- Train LSMVideoTransformer (full, 29 letters)
    │   └── predict_full.py    <- Multi-sign inference + LLM reconstruction
    └── text_processing/
        ├── buffer.py          <- LetterBuffer: thread-safe letter debouncer
        └── llm_processor.py   <- LLMProcessor: LangChain + GPT-4o word reconstruction
```

---

## Authors

Cesar Barrientos, Alan Rodriguez

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

