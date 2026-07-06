"""
AgriDoctor AI — Kaggle dataset downloader & organizer.

Downloads the PlantVillage and Rice Leaf Disease datasets, maps their class
folder names to AgriDoctor's DISEASE_LABELS, splits into train/val, and writes
a labels.csv ready for train_image_model.py.

Prerequisites:
    pip install kaggle
    Place your kaggle.json in ~/.kaggle/ (get it from kaggle.com/settings).

Usage:
    python scripts/download_kaggle_data.py            # downloads + organizes
    python scripts/download_kaggle_data.py --skip-download  # organize only
"""

import argparse
import logging
import os
import random
import shutil
import csv
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Project root is one level above scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw_kaggle"
DATASET_DIR = PROJECT_ROOT / "data" / "dataset"
LABELS_CSV = DATASET_DIR / "labels.csv"

# --------------------------------------------------------------------------- #
# PlantVillage folder name → AgriDoctor label mapping
# --------------------------------------------------------------------------- #
PLANTVILLAGE_MAP = {
    # Tomato
    "Tomato___Early_blight": "TOM_EARLY_BLIGHT",
    "Tomato___Late_blight": "TOM_LATE_BLIGHT",
    "Tomato___Leaf_Mold": "TOM_LEAF_MOLD",
    "Tomato___Septoria_leaf_spot": "TOM_SEPTORIA",
    "Tomato___Spider_mites Two-spotted_spider_mite": "TOM_SPIDER_MITES",
    "Tomato___Tomato_mosaic_virus": "TOM_MOSAIC",
    "Tomato___Bacterial_spot": "TOM_BACL_SPOT",
    "Tomato___healthy": "TOM_HEALTHY",
    "Tomato___Target_Spot": "TOM_UNKNOWN",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "TOM_UNKNOWN",
    # Potato
    "Potato___Early_blight": "POT_EARLY_BLIGHT",
    "Potato___Late_blight": "POT_LATE_BLIGHT",
    "Potato___healthy": "POT_HEALTHY",
    # Corn (Maize)
    "Corn_(maize)___Northern_Leaf_Blight": "MAIZE_NLB",
    "Corn_(maize)___Common_rust_": "MAIZE_RUST",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "MAIZE_GLS",
    "Corn_(maize)___healthy": "MAIZE_HEALTHY",
}

# Rice Leaf Disease dataset folder → AgriDoctor label
RICE_MAP = {
    "Leaf Blast": "RICE_BLAST",
    "leaf blast": "RICE_BLAST",
    "Brown Spot": "RICE_BROWN_SPOT",
    "brown spot": "RICE_BROWN_SPOT",
    "Bacterial Leaf Blight": "RICE_BACT_BLIGHT",
    "bacterial leaf blight": "RICE_BACT_BLIGHT",
    "Healthy": "RICE_HEALTHY",
    "healthy": "RICE_HEALTHY",
    "Hispa": "RICE_UNKNOWN",
    "hispa": "RICE_UNKNOWN",
}


def download_datasets(skip_download: bool = False) -> None:
    """Download PlantVillage + Rice datasets from Kaggle."""
    if skip_download:
        logger.info("--skip-download: skipping Kaggle download")
        return

    try:
        import kaggle  # noqa: F401
    except ImportError:
        logger.error(
            "kaggle package not installed. Run: pip install kaggle\n"
            "Then place your kaggle.json in ~/.kaggle/"
        )
        raise SystemExit(1)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    plantvillage_dir = RAW_DIR / "plantvillage"
    rice_dir = RAW_DIR / "rice"

    if not plantvillage_dir.exists():
        logger.info("Downloading PlantVillage dataset...")
        os.system(
            f"kaggle datasets download -d abdallahalidev/plantvillage-dataset "
            f"-p {plantvillage_dir} --unzip"
        )
    else:
        logger.info("PlantVillage already downloaded, skipping")

    if not rice_dir.exists():
        logger.info("Downloading Rice Leaf Disease dataset...")
        os.system(
            f"kaggle datasets download -d vbookshelf/rice-leaf-diseases "
            f"-p {rice_dir} --unzip"
        )
    else:
        logger.info("Rice dataset already downloaded, skipping")


def find_image_folders(base: Path) -> dict[str, Path]:
    """Recursively find folders that contain image files (leaf directories)."""
    folders = {}
    for dirpath, dirnames, filenames in os.walk(base):
        image_files = [
            f for f in filenames
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]
        if image_files and not dirnames:  # leaf directory with images
            folder_name = Path(dirpath).name
            folders[folder_name] = Path(dirpath)
    return folders


def organize_dataset(val_ratio: float = 0.2) -> None:
    """
    Map raw Kaggle downloads into AgriDoctor's label structure and split
    into train/val.
    """
    logger.info("Organizing dataset into %s ...", DATASET_DIR)

    # Clean previous output
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    train_dir = DATASET_DIR / "train"
    val_dir = DATASET_DIR / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    all_records = []  # (image_path, primary_label, severity_score)
    stats = {}  # label → count

    # --- Process PlantVillage ---
    plantvillage_dir = RAW_DIR / "plantvillage"
    if plantvillage_dir.exists():
        pv_folders = find_image_folders(plantvillage_dir)
        logger.info("Found %d PlantVillage folders", len(pv_folders))

        for folder_name, folder_path in pv_folders.items():
            label = PLANTVILLAGE_MAP.get(folder_name)
            if label is None:
                logger.warning("Unmapped PlantVillage folder: %s — skipping", folder_name)
                continue

            images = sorted([
                f for f in folder_path.iterdir()
                if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            ])

            random.seed(42)
            random.shuffle(images)
            split_idx = int(len(images) * (1 - val_ratio))

            for i, img in enumerate(images):
                split = "train" if i < split_idx else "val"
                dest_dir = (train_dir if split == "train" else val_dir) / label
                dest_dir.mkdir(parents=True, exist_ok=True)

                dest_path = dest_dir / img.name
                # Avoid name collisions
                if dest_path.exists():
                    dest_path = dest_dir / f"{img.stem}_{i}{img.suffix}"

                shutil.copy2(img, dest_path)

                severity = 0.0 if label.endswith("_HEALTHY") else 0.5
                rel_path = f"{split}/{label}/{dest_path.name}"
                all_records.append((rel_path, label, severity))

            stats[label] = stats.get(label, 0) + len(images)
            logger.info("  %s → %s: %d images", folder_name, label, len(images))
    else:
        logger.warning("PlantVillage directory not found at %s", plantvillage_dir)

    # --- Process Rice ---
    rice_dir = RAW_DIR / "rice"
    if rice_dir.exists():
        rice_folders = find_image_folders(rice_dir)
        logger.info("Found %d Rice folders", len(rice_folders))

        for folder_name, folder_path in rice_folders.items():
            label = RICE_MAP.get(folder_name)
            if label is None:
                logger.warning("Unmapped Rice folder: %s — skipping", folder_name)
                continue

            images = sorted([
                f for f in folder_path.iterdir()
                if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            ])

            random.seed(42)
            random.shuffle(images)
            split_idx = int(len(images) * (1 - val_ratio))

            for i, img in enumerate(images):
                split = "train" if i < split_idx else "val"
                dest_dir = (train_dir if split == "train" else val_dir) / label
                dest_dir.mkdir(parents=True, exist_ok=True)

                dest_path = dest_dir / img.name
                if dest_path.exists():
                    dest_path = dest_dir / f"{img.stem}_{i}{img.suffix}"

                shutil.copy2(img, dest_path)

                severity = 0.0 if label.endswith("_HEALTHY") else 0.5
                rel_path = f"{split}/{label}/{dest_path.name}"
                all_records.append((rel_path, label, severity))

            stats[label] = stats.get(label, 0) + len(images)
            logger.info("  %s → %s: %d images", folder_name, label, len(images))
    else:
        logger.warning("Rice directory not found at %s", rice_dir)

    # --- Write labels.csv ---
    logger.info("Writing %s (%d records) ...", LABELS_CSV, len(all_records))
    with open(LABELS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "primary_label", "severity_score"])
        for row in all_records:
            writer.writerow(row)

    # --- Summary ---
    logger.info("\n=== Dataset Summary ===")
    total = 0
    for label in sorted(stats):
        count = stats[label]
        total += count
        logger.info("  %-25s  %5d images", label, count)
    logger.info("  %-25s  %5d images", "TOTAL", total)

    train_count = len(list(train_dir.rglob("*.*")))
    val_count = len(list(val_dir.rglob("*.*")))
    logger.info("  Train: %d  |  Val: %d", train_count, val_count)
    logger.info("Labels CSV: %s", LABELS_CSV)
    logger.info("Done! ✅")


def main():
    parser = argparse.ArgumentParser(description="Download & organize Kaggle datasets for AgriDoctor")
    parser.add_argument("--skip-download", action="store_true", help="Skip Kaggle download, only organize")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio (default: 0.2)")
    args = parser.parse_args()

    download_datasets(skip_download=args.skip_download)
    organize_dataset(val_ratio=args.val_ratio)


if __name__ == "__main__":
    main()
