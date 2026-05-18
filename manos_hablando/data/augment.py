import logging
import random
from pathlib import Path

import albumentations as A
import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "raw"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
TARGET_COUNT = 100
SKIP_LETTERS = {"Ñ"}

AUGMENTATION_PIPELINE = A.Compose([
    A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.8),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    A.RandomScale(scale_limit=0.15, p=0.7),
])


def _get_images(folder: Path) -> list[Path]:
    """Return sorted list of image files in a folder."""
    return sorted(
        f for f in folder.iterdir()
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _get_original_images(folder: Path) -> list[Path]:
    """Return only non-augmented images (no aug_ prefix)."""
    return [f for f in _get_images(folder) if not f.name.startswith("aug_")]


def augment_letter(
    letter_folder: Path,
    target: int = TARGET_COUNT,
    pipeline: A.Compose = AUGMENTATION_PIPELINE,
) -> tuple[int, int]:
    """
    Augment images for a single letter folder to reach the target count.

    For folders with more than target images, randomly removes excess originals.
    For folders below target, generates augmented copies with 'aug_' prefix.

    Args:
        letter_folder: Path to the letter's image folder.
        target: Desired number of images after augmentation.
        pipeline: Albumentations augmentation pipeline.

    Returns:
        Tuple of (before_count, after_count).
    """
    originals = _get_original_images(letter_folder)
    before_count = len(originals)

    if before_count == 0:
        return 0, 0

    # Downsample: remove random originals to reach target
    if before_count > target:
        to_remove = random.sample(originals, before_count - target)
        for img_path in to_remove:
            img_path.unlink()
        return before_count, target

    # Augment to reach target
    needed = target - before_count
    aug_index = 0

    while aug_index < needed:
        source_path = originals[aug_index % len(originals)]
        image = cv2.imread(str(source_path))
        if image is None:
            logger.warning(f"Could not read image: {source_path}")
            originals.remove(source_path)
            if not originals:
                break
            continue

        augmented = pipeline(image=image)["image"]
        aug_name = f"aug_{aug_index:04d}{source_path.suffix}"
        cv2.imwrite(str(letter_folder / aug_name), augmented)
        aug_index += 1

    return before_count, before_count + aug_index


def augment_dataset(raw_path: Path = RAW_DATA_PATH, target: int = TARGET_COUNT) -> None:
    """
    Augment all letter folders in the dataset to reach the target image count.

    Skips letters in SKIP_LETTERS. Prints a summary table when done.

    Args:
        raw_path: Path to the raw dataset folder.
        target: Desired number of images per letter.
    """
    letter_folders = sorted(f for f in raw_path.iterdir() if f.is_dir())

    if not letter_folders:
        logger.warning(f"No letter folders found in {raw_path}")
        return

    # Clean previous augmentations
    for folder in letter_folders:
        for img in folder.iterdir():
            if img.name.startswith("aug_"):
                img.unlink()

    results: list[tuple[str, int, int]] = []

    for folder in letter_folders:
        letter = folder.name
        if letter in SKIP_LETTERS:
            logger.info(f"Skipping letter '{letter}' (no images available)")
            results.append((letter, 0, 0))
            continue

        before, after = augment_letter(folder, target=target)
        results.append((letter, before, after))
        logger.info(f"Letter '{letter}': {before} -> {after}")

    # Print summary table
    print("\n" + "=" * 45)
    print(f"{'Letter':<10} {'Before':>8} {'After':>8} {'Action':<15}")
    print("-" * 45)

    for letter, before, after in results:
        if before == 0:
            action = "skipped"
        elif before > target:
            action = "downsampled"
        elif before < target:
            action = "augmented"
        else:
            action = "unchanged"
        print(f"{letter:<10} {before:>8} {after:>8} {action:<15}")

    print("=" * 45)


if __name__ == "__main__":
    augment_dataset()
