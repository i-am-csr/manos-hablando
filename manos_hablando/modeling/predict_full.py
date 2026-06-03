from enum import Enum
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.table import Table
import torch
import typer

from manos_hablando.config import MODELS_DIR
from manos_hablando.data.mediapipe_handler import extract_keypoints_video_per_frame
from manos_hablando.dataset_full import load_full_keypoints, prepare_inference_sequence
from manos_hablando.modeling.train_video import LSMVideoTransformer
from manos_hablando.text_processing import LetterBuffer

app = typer.Typer()
console = Console()

MODEL_PATH = MODELS_DIR / "lsm_full_transformer.pt"

# Per-user systematic confusion fixes for the current trained model. Toggle
# with --correct-biases. These are *not* universal — they encode one signer's
# observed substitutions against the MSL-ABC-trained checkpoint and will be
# wrong for any user whose hand shapes don't share these biases. Re-derive
# the table after fine-tuning or retraining.
KNOWN_BIAS_SUBSTITUTIONS: dict[str, str] = {
    "P": "K",
    "RR": "R",
    "LL": "L",
    "T": "S",
}

# Multi-character classes in the trained label set. Used to re-tokenize a
# concatenated word string back into letter tokens before bias correction,
# since the buffer joins everything with "".
MULTI_CHAR_LETTERS: set[str] = {"LL", "RR"}


def tokenize_word(word: str) -> list[str]:
    """Greedy 2-then-1-char split honoring MULTI_CHAR_LETTERS."""
    tokens: list[str] = []
    i = 0
    while i < len(word):
        if i + 1 < len(word) and word[i : i + 2] in MULTI_CHAR_LETTERS:
            tokens.append(word[i : i + 2])
            i += 2
        else:
            tokens.append(word[i])
            i += 1
    return tokens


def apply_bias_corrections(words: list[str]) -> list[str]:
    """Tokenize each word, map letters through the substitution table, rejoin."""
    return [
        "".join(KNOWN_BIAS_SUBSTITUTIONS.get(tok, tok) for tok in tokenize_word(w)) for w in words
    ]


class Mode(str, Enum):
    """Segmentation strategy."""

    # Slide a classifier window across the video and debounce per-frame
    # predictions through LetterBuffer. Robust to natural signing without
    # explicit hand-down pauses.
    sliding = "sliding"
    # Cut segments at runs of no-hand frames. Cleanest when the user
    # deliberately pauses between signs (e.g. the J reference video).
    gaps = "gaps"


# ─────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────


def load_model(
    model_path: Path,
    num_classes: int,
    device: torch.device,
) -> LSMVideoTransformer:
    """Load the trained full LSMVideoTransformer from disk."""
    model = LSMVideoTransformer(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info(f"Model loaded from {model_path}")
    return model


def classify_segment(
    frames: list[list],
    model: LSMVideoTransformer,
    encoder,
    device: torch.device,
    k: int = 3,
) -> list[tuple[str, float]]:
    """Top-k classification of a single sign segment."""
    X_tensor = prepare_inference_sequence(
        frames,
        is_static_image=False,
    ).to(device)

    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=-1)[0]
        topk = torch.topk(probs, min(k, probs.shape[0]))

    return [
        (encoder.classes_[idx.item()], score.item())
        for idx, score in zip(topk.indices, topk.values)
    ]


# ─────────────────────────────────────────
# Mode: gaps — cut on no-hand boundaries
# ─────────────────────────────────────────


def segment_timeline(
    timeline: list[list | None],
    gap_frames: int = 8,
    min_segment: int = 6,
) -> list[tuple[int, int, list[list]]]:
    """Walk a per-frame timeline and return one segment per detected sign."""
    segments: list[tuple[int, int, list[list]]] = []
    current: list[list] = []
    seg_start: int | None = None
    no_hand_run = 0

    for idx, frame in enumerate(timeline):
        if frame is not None:
            if seg_start is None:
                seg_start = idx
            current.append(frame)
            no_hand_run = 0
            continue

        no_hand_run += 1
        if no_hand_run >= gap_frames and current:
            seg_end = idx - no_hand_run
            if len(current) >= min_segment:
                segments.append((seg_start, seg_end, current))
            current = []
            seg_start = None
            no_hand_run = 0

    if current and len(current) >= min_segment and seg_start is not None:
        segments.append((seg_start, seg_start + len(current) - 1, current))

    return segments


# ─────────────────────────────────────────
# Mode: sliding — debounce per-frame
# ─────────────────────────────────────────


def analyze_sliding(
    timeline: list[list | None],
    fps: float,
    model: LSMVideoTransformer,
    encoder,
    device: torch.device,
    window_frames: int = 30,
    stride: int = 3,
    stability_window: int = 4,
    confidence_threshold: float = 0.5,
    word_boundary_frames: int = 15,
) -> tuple[list[str], list[tuple[int, str, float, str]]]:
    """
    Slide a classifier window across the timeline and debounce per-step
    predictions through a LetterBuffer.

    At each stride step, classify the trailing `window_frames` of detected
    frames as a single sign and feed (letter, confidence) into the buffer.
    The buffer commits a letter once `stability_window` consecutive
    predictions agree above `confidence_threshold`. Stretches of no-hand
    frames advance the buffer's word-boundary timer; when it fires, the
    current word is snapshotted and the buffer resets.

    Returns:
        words: list of committed words in order
        events: list of (frame_idx, letter, conf, status) for the verbose log.
                status ∈ {"pred", "no_hand", "too_short"}.
    """
    buffer = LetterBuffer(
        stability_window=stability_window,
        confidence_threshold=confidence_threshold,
        word_boundary_frames=word_boundary_frames,
    )

    events: list[tuple[int, str, float, str]] = []
    words: list[str] = []
    prev_word_state = ""

    for t in range(0, len(timeline), stride):
        frame = timeline[t]

        if frame is None:
            # The buffer counts each no-hand call against word_boundary_frames,
            # so at stride > 1 we replay the call `stride` times to keep the
            # timer scaled to real video time.
            for _ in range(stride):
                buffer.add_no_hand()
            events.append((t, "", 0.0, "no_hand"))
        else:
            window_start = max(0, t - window_frames + 1)
            window = [f for f in timeline[window_start : t + 1] if f is not None]
            if len(window) < 3:
                buffer.add_no_hand()
                events.append((t, "", 0.0, "too_short"))
            else:
                preds = classify_segment(window, model, encoder, device, k=1)
                letter, conf = preds[0]
                buffer.add_prediction(letter, conf)
                events.append((t, letter, conf, "pred"))

        # Snapshot a completed word and reset so the next word starts clean.
        if buffer.word_complete():
            word = buffer.get_current_word()
            if word and word != prev_word_state:
                words.append(word)
                prev_word_state = word
            buffer.reset()
            prev_word_state = ""

    # Flush any letters still in the buffer at end-of-video.
    trailing = buffer.get_current_word()
    if trailing:
        words.append(trailing)

    return words, events


# ─────────────────────────────────────────
# CLI
# ─────────────────────────────────────────


@app.command()
def main(
    video_path: Path = typer.Argument(
        ...,
        help="Path to a video containing one or more fingerspelled signs.",
    ),
    model_path: Path = MODEL_PATH,
    mode: Mode = typer.Option(
        Mode.sliding,
        "--mode",
        help="sliding (debounce per-frame predictions, good for natural "
        "signing) or gaps (split on hand-down pauses).",
    ),
    # gaps-mode knobs
    gap_frames: int = typer.Option(
        8,
        help="[gaps] consecutive no-hand frames that close a sign segment.",
    ),
    min_segment: int = typer.Option(
        6,
        help="[gaps] minimum frames in a segment (shorter ones dropped).",
    ),
    min_confidence: float = typer.Option(
        0.4,
        help="[gaps] drop segment predictions below this confidence.",
    ),
    # sliding-mode knobs
    window_frames: int = typer.Option(
        30,
        help="[sliding] frames per classifier window (default 1s @ 30fps).",
    ),
    stride: int = typer.Option(
        3,
        help="[sliding] step between classifier evaluations, in frames.",
    ),
    stability: int = typer.Option(
        4,
        help="[sliding] consecutive matching predictions required to commit "
        "a letter. At stride=3 default, 4 → ~12 frames (~0.4s) of "
        "agreement.",
    ),
    confidence: float = typer.Option(
        0.5,
        help="[sliding] minimum confidence to count toward stability.",
    ),
    word_boundary: int = typer.Option(
        15,
        help="[sliding] no-hand frames that close a word.",
    ),
    top_k: int = typer.Option(3, help="[gaps] Top K candidates per segment"),
    slowdown: int = typer.Option(
        1,
        help="Replicate each timeline frame N times before analysis. "
        "Use 2-3 for fast natural fingerspelling where letters are "
        "held briefly (<20 frames each). Identity at N=1.",
    ),
    no_llm: bool = typer.Option(
        False,
        "--no-llm",
        help="Skip the LLM reconstruction step.",
    ),
    correct_biases: bool = typer.Option(
        False,
        "--correct-biases",
        help="Apply per-user substitutions (P→K, RR→R, LL→L, T→S) before "
        "output and before the LLM call. These compensate for one "
        "signer's systematic confusions on the current model and will "
        "produce wrong reads for any other user.",
    ),
    context: str = typer.Option(
        "",
        "--context",
        help="Hint for the LLM about what was signed "
        '(e.g. "a Spanish person\'s first name and surnames"). '
        "Without this, the LLM defaults to common Spanish vocabulary.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="[sliding] print the per-step event log for diagnostics.",
    ),
):
    """
    Predict every LSM sign in a single video and pipe the letter sequence
    through the LLM for Spanish reconstruction.

    Two segmentation modes:

      sliding (default): slides a classifier window across the timeline
        and debounces predictions, so the user can fingerspell smoothly
        without dropping their hand between letters.

      gaps: splits the video at no-hand pauses; cleanest when the user
        deliberately lowers their hand between signs.

    Example:
        uv run python -m manos_hablando.modeling.predict_full clip.mp4 2>/dev/null
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, encoder = load_full_keypoints()
    num_classes = len(encoder.classes_)
    model = load_model(model_path, num_classes, device)

    logger.info(f"Extracting per-frame keypoints from {video_path.name}...")
    timeline_result = extract_keypoints_video_per_frame(video_path)
    if timeline_result is None:
        logger.error("No hand detected anywhere in the video.")
        raise typer.Exit(1)

    timeline = timeline_result["keypoints_per_frame"]
    fps = timeline_result["fps"]
    logger.info(
        f"Frames: {timeline_result['total_frames']} "
        f"(hand detected in {timeline_result['detected_frames']}) "
        f"@ {fps:.1f} fps  |  mode={mode.value}"
    )

    if slowdown > 1:
        # Frame replication: each natural frame becomes `slowdown` identical
        # frames in the timeline. Effective FPS for the sliding window scales
        # so a 10-frame natural held pose becomes 10*slowdown frames, matching
        # the 30-frame window the model was trained for.
        timeline = [f for f in timeline for _ in range(slowdown)]
        fps = fps * slowdown
        logger.info(
            f"Slowdown x{slowdown} → {len(timeline)} effective frames @ {fps:.1f} effective fps."
        )

    if mode is Mode.gaps:
        letters = _run_gaps(
            timeline,
            fps,
            model,
            encoder,
            device,
            gap_frames,
            min_segment,
            min_confidence,
            top_k,
        )
    else:
        letters = _run_sliding(
            timeline,
            fps,
            model,
            encoder,
            device,
            window_frames,
            stride,
            stability,
            confidence,
            word_boundary,
            verbose,
        )

    if not letters:
        logger.error("No letters committed. Try loosening thresholds.")
        raise typer.Exit(1)

    raw = " ".join(letters)
    logger.success(f"Raw letter sequence: {raw}")

    if correct_biases:
        letters = apply_bias_corrections(letters)
        corrected = " ".join(letters)
        logger.success(f"Bias-corrected sequence: {corrected}")

    if no_llm:
        return

    from manos_hablando.text_processing import LLMProcessor

    llm = LLMProcessor()
    # The LLM expects a flat letter list. For multi-word output we join
    # with spaces — the prompt is forgiving enough to interpret either.
    flat_letters: list[str] = list("".join(letters))
    reconstructed = llm.reconstruct(flat_letters, context=context or None)
    badge = {
        "success": "🟢 SUCCESS",
        "fallback": "🔴 FALLBACK",
        "no_key": "⚪ NO KEY",
    }.get(llm.last_status, llm.last_status)
    logger.success(f"{badge} ({llm.last_latency_ms:.0f}ms) → '{reconstructed}'")
    if llm.last_error:
        logger.warning(f"LLM error: {llm.last_error}")


def _run_gaps(
    timeline,
    fps,
    model,
    encoder,
    device,
    gap_frames,
    min_segment,
    min_confidence,
    top_k,
) -> list[str]:
    segments = segment_timeline(
        timeline,
        gap_frames=gap_frames,
        min_segment=min_segment,
    )
    logger.info(f"Found {len(segments)} sign segment(s).")
    if not segments:
        logger.error(
            "No segments long enough. Try lowering --min-segment or "
            "--gap-frames, or switch to --mode sliding."
        )
        return []

    letters: list[str] = []
    table = Table(title="Sign segments")
    for col, style in (
        ("#", "cyan"),
        ("Time", "magenta"),
        ("Frames", "white"),
        ("Prediction", "green"),
        (f"Top-{top_k}", "dim"),
    ):
        table.add_column(col, style=style)

    for i, (start, end, frames) in enumerate(segments, 1):
        predictions = classify_segment(frames, model, encoder, device, k=top_k)
        top_letter, top_conf = predictions[0]
        dropped = top_conf < min_confidence
        if not dropped:
            letters.append(top_letter)

        candidates = ", ".join(f"{l} {c * 100:.0f}%" for l, c in predictions)
        flag = "  ⚠ dropped" if dropped else ""
        table.add_row(
            str(i),
            f"{start / fps:5.2f}s–{end / fps:5.2f}s",
            str(len(frames)),
            f"{top_letter} ({top_conf * 100:.1f}%){flag}",
            candidates,
        )

    console.print(table)
    return letters


def _run_sliding(
    timeline,
    fps,
    model,
    encoder,
    device,
    window_frames,
    stride,
    stability,
    confidence,
    word_boundary,
    verbose,
) -> list[str]:
    logger.info(
        f"Sliding analysis: window={window_frames}f stride={stride}f "
        f"stability={stability} conf>={confidence} word_boundary={word_boundary}f"
    )

    words, events = analyze_sliding(
        timeline,
        fps,
        model,
        encoder,
        device,
        window_frames=window_frames,
        stride=stride,
        stability_window=stability,
        confidence_threshold=confidence,
        word_boundary_frames=word_boundary,
    )

    table = Table(title="Committed words")
    table.add_column("#", style="cyan")
    table.add_column("Word", style="green")
    table.add_column("Letters", style="white")
    for i, word in enumerate(words, 1):
        table.add_row(str(i), word, " ".join(word))
    console.print(table)

    if verbose:
        ev_table = Table(title="Per-step events")
        for col, style in (
            ("Frame", "cyan"),
            ("Time", "magenta"),
            ("Letter", "green"),
            ("Conf", "yellow"),
            ("Status", "dim"),
        ):
            ev_table.add_column(col, style=style)
        for frame_idx, letter, conf, status in events:
            ev_table.add_row(
                str(frame_idx),
                f"{frame_idx / fps:5.2f}s",
                letter or "—",
                f"{conf * 100:.0f}%" if status == "pred" else "—",
                status,
            )
        console.print(ev_table)

    return words


if __name__ == "__main__":
    app()
