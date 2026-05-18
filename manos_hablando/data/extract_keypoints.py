import json
import logging
from pathlib import Path

from manos_hablando.data.mediapipe_handler import extract_keypoints

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def extract_dataset_keypoints(
    raw_path: Path = RAW_DATA_PATH,
    processed_path: Path = PROCESSED_DATA_PATH,
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

        logger.info(f"Processing letter '{letter}': {len(images)} images")

        for image_path in images:
            keypoints = extract_keypoints(image_path)

            if keypoints is not None:
                dataset.append({
                    "letter": letter,
                    "file": image_path.name,
                    "keypoints": keypoints,
                })
            else:
                logger.warning(f"No hand detected: {image_path}")
                failures.append(str(image_path))

    # Save dataset
    output_path = processed_path / "keypoints.json"
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    # Summary
    logger.info(f"Done!")
    logger.info(f"  ✅ Extracted: {len(dataset)}")
    logger.info(f"  ❌ Failed:    {len(failures)}")
    logger.info(f"  💾 Saved to:  {output_path}")

    if failures:
        logger.warning("Failed images:")
        for f in failures:
            logger.warning(f"  {f}")


if __name__ == "__main__":
    extract_dataset_keypoints()