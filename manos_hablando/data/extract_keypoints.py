import json
from pathlib import Path

from loguru import logger
from rich.progress import track
from rich.console import Console
from rich.table import Table

from manos_hablando.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from manos_hablando.data.mediapipe_handler import extract_keypoints

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
console = Console()


def extract_dataset_keypoints(
    raw_path: Path = RAW_DATA_DIR,
    processed_path: Path = PROCESSED_DATA_DIR,
) -> None:
    """
    Walk through all letter folders in raw_path, extract keypoints
    from each image and save the result to processed_path/keypoints.json.

    Args:
        raw_path: Path to the raw dataset folder (one subfolder per letter).
        processed_path: Path to save the processed keypoints JSON.
    """
    processed_path.mkdir(parents=True, exist_ok=True)

    dataset = []
    failures = []
    summary = []

    letter_folders = sorted([f for f in raw_path.iterdir() if f.is_dir()])

    if not letter_folders:
        logger.warning(f"No letter folders found in {raw_path}")
        return

    for letter_folder in letter_folders:
        letter = letter_folder.name
        images = [
            f for f in letter_folder.iterdir()
            if f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        extracted = 0
        failed = 0

        for image_path in track(images, description=f"[cyan]Processing {letter}..."):
            result = extract_keypoints(image_path)

            if result is not None:
                dataset.append({
                    "letter": letter,
                    "file": image_path.name,
                    "handedness": result["handedness"],
                    "handedness_confidence": result["handedness_confidence"],
                    "keypoints": result["keypoints"],
                    "world_keypoints": result["world_keypoints"],
                })
                extracted += 1
            else:
                failures.append(str(image_path))
                failed += 1

        summary.append((letter, extracted, failed))
        logger.info(f"Letter '{letter}': {extracted} extracted, {failed} failed")

    # Save dataset
    output_path = processed_path / "keypoints.json"
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    # Summary table
    table = Table(title="Extraction Summary")
    table.add_column("Letter", style="cyan")
    table.add_column("Extracted", style="green")
    table.add_column("Failed", style="red")

    for letter, extracted, failed in summary:
        table.add_row(letter, str(extracted), str(failed))

    console.print(table)

    logger.success(f"Done! Extracted: {len(dataset)} | Failed: {len(failures)}")
    logger.info(f"Saved to: {output_path}")


if __name__ == "__main__":
    extract_dataset_keypoints()