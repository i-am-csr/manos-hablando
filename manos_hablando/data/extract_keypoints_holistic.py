"""
Walk every word folder under data/raw/words/ and extract holistic
(face + pose + hands) per-frame features for each video clip.

Storage layout — features are too large for one JSON, so each video gets
its own .npz and a single manifest indexes them:

    data/processed/holistic/{Word}/{file_stem}.npz   # key "features"
    data/processed/holistic_keypoints.json           # list of metadata entries
"""

import json
from pathlib import Path

from loguru import logger
import numpy as np
from rich.console import Console
from rich.progress import track
from rich.table import Table
import typer

from manos_hablando.config import PROCESSED_DATA_DIR, RAW_WORDS_DIR
from manos_hablando.data.holistic_handler import FEATURE_DIM, extract_holistic_video

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
HOLISTIC_DIR = PROCESSED_DATA_DIR / "holistic"
MANIFEST_FILE = PROCESSED_DATA_DIR / "holistic_keypoints.json"

console = Console()
app = typer.Typer()


def extract_words_dataset(
    raw_path: Path = RAW_WORDS_DIR,
    out_dir: Path = HOLISTIC_DIR,
    manifest_path: Path = MANIFEST_FILE,
) -> None:
    """Extract holistic features for every video under raw_path/{WORD}/."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    word_folders = sorted(f for f in raw_path.iterdir() if f.is_dir())
    if not word_folders:
        logger.warning(f"No word folders found in {raw_path}")
        return

    manifest: list[dict] = []
    failures: list[str] = []
    summary: list[tuple[str, int, int]] = []

    for word_folder in word_folders:
        word = word_folder.name
        videos = sorted(
            f for f in word_folder.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        extracted = 0
        failed = 0
        word_out = out_dir / word
        word_out.mkdir(parents=True, exist_ok=True)

        for video_path in track(videos, description=f"[cyan]{word}"):
            try:
                result = extract_holistic_video(video_path)
            except Exception as exc:
                logger.error(f"Failed on {video_path.name}: {exc}")
                failures.append(str(video_path))
                failed += 1
                continue

            if result is None:
                failures.append(str(video_path))
                failed += 1
                continue

            npz_path = word_out / f"{video_path.stem}.npz"
            np.savez_compressed(npz_path, features=result["features"])

            manifest.append(
                {
                    "word": word,
                    "file": video_path.name,
                    "npz": str(npz_path.relative_to(out_dir.parent)),
                    "fps": result["fps"],
                    "total_frames": result["total_frames"],
                    "detected_frames": result["detected_frames"],
                    "feature_dim": FEATURE_DIM,
                    "modality_counts": result["modality_counts"],
                }
            )
            extracted += 1

        summary.append((word, extracted, failed))
        logger.info(f"Word '{word}': {extracted} extracted, {failed} failed")

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    table = Table(title="Holistic Extraction Summary")
    table.add_column("Word", style="cyan")
    table.add_column("Extracted", style="green")
    table.add_column("Failed", style="red")
    for word, extracted, failed in summary:
        table.add_row(word, str(extracted), str(failed))
    console.print(table)

    logger.success(f"Done! Extracted: {len(manifest)} | Failed: {len(failures)}")
    logger.info(f"Manifest: {manifest_path}")
    logger.info(f"Features dir: {out_dir}")

    if failures:
        logger.warning("Failed videos:")
        for f in failures:
            logger.warning(f"  {f}")


@app.command()
def main(
    raw_path: Path = RAW_WORDS_DIR,
    out_dir: Path = HOLISTIC_DIR,
    manifest_path: Path = MANIFEST_FILE,
) -> None:
    """Extract holistic features for every video under data/raw/words/."""
    extract_words_dataset(raw_path=raw_path, out_dir=out_dir, manifest_path=manifest_path)


if __name__ == "__main__":
    app()
