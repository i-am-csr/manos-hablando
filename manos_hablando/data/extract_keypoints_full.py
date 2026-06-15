import json
from pathlib import Path
import random

from loguru import logger
from rich.console import Console
from rich.progress import track
from rich.table import Table

from manos_hablando.config import PROCESSED_DATA_DIR, RAW_LETTERS_DIR
from manos_hablando.data.mediapipe_handler import (
    _build_landmarker,
    extract_keypoints,
    extract_keypoints_video,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
DYNAMIC_LETTERS = {"J", "K", "Ñ", "Q", "X", "Z"}
# Cap for static-letter image folders. Larger sets show diminishing returns and
# inflate extraction time linearly. None disables the cap.
DEFAULT_MAX_STATIC_SAMPLES = 1500
SAMPLE_SEED = 42
console = Console()


def _extract_static_entry(image_path: Path, letter: str, landmarker=None) -> dict | None:
    """Run image-mode MediaPipe and wrap output as a 1-frame video entry."""
    result = extract_keypoints(image_path, landmarker=landmarker)
    if result is None:
        return None
    # Wrap (21, 3) → (1, 21, 3) so downstream code can treat every entry as a sequence.
    return {
        "letter": letter,
        "file": image_path.name,
        "source": "image",
        "fps": None,
        "total_frames": 1,
        "detected_frames": 1,
        "keypoints": [result["keypoints"]],
        "world_keypoints": [result["world_keypoints"]],
    }


def _extract_dynamic_entry(video_path: Path, letter: str) -> dict | None:
    """Run video-mode MediaPipe across the whole clip."""
    result = extract_keypoints_video(video_path)
    if result is None:
        return None
    return {
        "letter": letter,
        "file": video_path.name,
        "source": "video",
        "fps": result["fps"],
        "total_frames": result["total_frames"],
        "detected_frames": result["detected_frames"],
        "keypoints": result["keypoints"],
        "world_keypoints": result["world_keypoints"],
    }


def extract_full_dataset_keypoints(
    raw_path: Path = RAW_LETTERS_DIR,
    processed_path: Path = PROCESSED_DATA_DIR,
    max_static_samples: int | None = DEFAULT_MAX_STATIC_SAMPLES,
) -> None:
    """
    Walk every letter folder under raw_path and emit a single
    full_keypoints.json that mixes static (1-frame) and dynamic (N-frame)
    samples in the same schema.

    Static letter folders are expected to contain images; dynamic letter
    folders (J, K, Ñ, Q, X, Z) are expected to contain videos.
    """
    processed_path.mkdir(parents=True, exist_ok=True)

    dataset = []
    failures = []
    summary = []

    letter_folders = sorted(f for f in raw_path.iterdir() if f.is_dir())

    if not letter_folders:
        logger.warning(f"No letter folders found in {raw_path}")
        return

    # One IMAGE-mode landmarker shared across all static images. Building one
    # spins up a Metal/GL context (~300 ms); doing it per image dominates
    # extraction time. Video letters still build their own VIDEO-mode
    # landmarker per clip inside extract_keypoints_video.
    shared_image_landmarker = _build_landmarker()

    for letter_folder in letter_folders:
        letter = letter_folder.name
        is_dynamic = letter in DYNAMIC_LETTERS
        extensions = VIDEO_EXTENSIONS if is_dynamic else IMAGE_EXTENSIONS

        samples = sorted(f for f in letter_folder.iterdir() if f.suffix.lower() in extensions)

        if not is_dynamic and max_static_samples is not None and len(samples) > max_static_samples:
            rng = random.Random(SAMPLE_SEED)
            samples = sorted(rng.sample(samples, max_static_samples))
            logger.info(
                f"Letter '{letter}': capped to {max_static_samples} random samples "
                f"(seed={SAMPLE_SEED})"
            )

        extracted = 0
        failed = 0

        for sample_path in track(
            samples,
            description=f"[cyan]{letter} ({'video' if is_dynamic else 'image'})",
        ):
            if is_dynamic:
                entry = _extract_dynamic_entry(sample_path, letter)
            else:
                entry = _extract_static_entry(
                    sample_path, letter, landmarker=shared_image_landmarker,
                )

            if entry is not None:
                dataset.append(entry)
                extracted += 1
            else:
                failures.append(str(sample_path))
                failed += 1

        summary.append((letter, "video" if is_dynamic else "image", extracted, failed))
        logger.info(
            f"Letter '{letter}' ({'video' if is_dynamic else 'image'}): "
            f"{extracted} extracted, {failed} failed"
        )

    shared_image_landmarker.close()

    output_path = processed_path / "full_keypoints.json"
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    table = Table(title="Full Extraction Summary")
    table.add_column("Letter", style="cyan")
    table.add_column("Source", style="magenta")
    table.add_column("Extracted", style="green")
    table.add_column("Failed", style="red")
    for letter, source, extracted, failed in summary:
        table.add_row(letter, source, str(extracted), str(failed))
    console.print(table)

    logger.success(f"Done! Extracted: {len(dataset)} | Failed: {len(failures)}")
    logger.info(f"Saved to: {output_path}")

    if failures:
        logger.warning("Failed samples:")
        for f in failures:
            logger.warning(f"  {f}")


if __name__ == "__main__":
    extract_full_dataset_keypoints()
