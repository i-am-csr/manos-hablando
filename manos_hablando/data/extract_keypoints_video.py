import json
from pathlib import Path

from loguru import logger
from rich.progress import track
from rich.console import Console
from rich.table import Table

from manos_hablando.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from manos_hablando.data.mediapipe_handler import extract_keypoints_video

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
DYNAMIC_LETTERS = {"J", "K", "Ñ", "Q", "X", "Z"}
console = Console()


def extract_video_dataset_keypoints(
    raw_path: Path = RAW_DATA_DIR,
    processed_path: Path = PROCESSED_DATA_DIR,
) -> None:
    """
    Walk through dynamic letter folders in raw_path, extract per-frame
    keypoints from each video and save to processed_path/video_keypoints.json.

    Only processes folders whose name is in DYNAMIC_LETTERS.

    Args:
        raw_path: Path to the raw dataset folder (one subfolder per letter).
        processed_path: Path to save the processed keypoints JSON.
    """
    processed_path.mkdir(parents=True, exist_ok=True)

    dataset = []
    failures = []
    summary = []

    letter_folders = sorted([
        f for f in raw_path.iterdir()
        if f.is_dir() and f.name in DYNAMIC_LETTERS
    ])

    if not letter_folders:
        logger.warning(f"No dynamic letter folders found in {raw_path}")
        return

    for letter_folder in letter_folders:
        letter = letter_folder.name
        videos = [
            f for f in letter_folder.iterdir()
            if f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        extracted = 0
        failed = 0

        for video_path in track(videos, description=f"[cyan]Processing {letter}..."):
            result = extract_keypoints_video(video_path)

            if result is not None:
                dataset.append({
                    "letter": letter,
                    "file": video_path.name,
                    "fps": result["fps"],
                    "total_frames": result["total_frames"],
                    "detected_frames": result["detected_frames"],
                    "keypoints": result["keypoints"],
                    "world_keypoints": result["world_keypoints"],
                })
                extracted += 1
            else:
                failures.append(str(video_path))
                failed += 1

        summary.append((letter, extracted, failed))
        logger.info(f"Letter '{letter}': {extracted} extracted, {failed} failed")

    # Save dataset
    output_path = processed_path / "video_keypoints.json"
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    # Summary table
    table = Table(title="Video Extraction Summary")
    table.add_column("Letter", style="cyan")
    table.add_column("Extracted", style="green")
    table.add_column("Failed", style="red")

    for letter, extracted, failed in summary:
        table.add_row(letter, str(extracted), str(failed))

    console.print(table)

    logger.success(f"Done! Extracted: {len(dataset)} | Failed: {len(failures)}")
    logger.info(f"Saved to: {output_path}")

    if failures:
        logger.warning(f"Failed videos:")
        for f in failures:
            logger.warning(f"  {f}")


if __name__ == "__main__":
    extract_video_dataset_keypoints()
