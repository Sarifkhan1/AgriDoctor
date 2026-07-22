"""
AgriDoctor — build the on-disk training corpus for the local CNN.

Produces a stratified train/val/test image folder tree from two public datasets:

  1. `minhhungg/plant-disease-dataset` (70,295 images, 38 classes)
       * 19 classes map onto AgriDoctor's taxonomy   -> in-scope disease labels
       * the other 19 are leaves of crops we don't
         cover (apple, grape, orange, soybean, ...)  -> OOS_OTHER_PLANT
  2. `zh-plus/tiny-imagenet` (100k object photos)    -> OOS_NOT_PLANT

The two out-of-scope classes are what make a local-first pipeline safe. A plain
19-class softmax must assign *some* crop disease to a photo of a car; with an
explicit reject head the model can say "not a plant" and we never hit the API.

Note the two rejects mean different things downstream (see backend/ai/analyzer.py):
OOS_OTHER_PLANT is "a leaf, but not one I was trained on" -> escalate to the
hosted vision model, because AgriDoctor's taxonomy also covers rice, chili,
cucumber, eggplant and livestock that PlantVillage has no images for.
OOS_NOT_PLANT is "not vegetation at all" -> reject locally, no API call.

Images are decoded once and cached as 256px JPEGs so training epochs are I/O
bound on small files rather than re-decoding parquet every pass.

Usage:
    .venv/bin/python scripts/prepare_dataset.py                  # full corpus
    .venv/bin/python scripts/prepare_dataset.py --per-class 60 --out data/dataset_smoke
"""

import argparse
import json
import logging
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("prepare")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OOS_OTHER_PLANT = "OOS_OTHER_PLANT"
OOS_NOT_PLANT = "OOS_NOT_PLANT"

# Raw PlantVillage class string -> AgriDoctor taxonomy label. Matched by requiring
# every substring to appear (case-insensitively) in the raw class name.
MAP_RULES = [
    (("tomato", "early blight"), "TOM_EARLY_BLIGHT"),
    (("tomato", "late blight"), "TOM_LATE_BLIGHT"),
    (("tomato", "leaf mold"), "TOM_LEAF_MOLD"),
    (("tomato", "septoria"), "TOM_SEPTORIA"),
    (("tomato", "spider mites"), "TOM_SPIDER_MITES"),
    (("tomato", "mosaic"), "TOM_MOSAIC"),
    (("tomato", "yellow leaf curl"), "TOM_YELLOW_LEAF_CURL"),
    (("tomato", "target spot"), "TOM_TARGET_SPOT"),
    (("tomato", "bacterial spot"), "TOM_BACL_SPOT"),
    (("tomato", "healthy"), "TOM_HEALTHY"),
    (("potato", "early blight"), "POT_EARLY_BLIGHT"),
    (("potato", "late blight"), "POT_LATE_BLIGHT"),
    (("potato", "healthy"), "POT_HEALTHY"),
    (("pepper", "bacterial spot"), "PEP_BACT_SPOT"),
    (("pepper", "healthy"), "PEP_HEALTHY"),
    (("corn", "common rust"), "MAIZE_RUST"),
    (("corn", "northern leaf blight"), "MAIZE_NLB"),
    (("corn", "gray leaf spot"), "MAIZE_GLS"),
    (("corn", "healthy"), "MAIZE_HEALTHY"),
]

# Additional crops that PlantVillage does not cover, pulled from the Project-AgML
# collection (one consistent source, which matters: mixing many corpora with
# different backgrounds invites the model to learn "which dataset" instead of
# "which disease"). Each entry maps the dataset's own class name to an AgriDoctor
# taxonomy label. Classes we deliberately DROP are set to None: fruit-only images
# (this is a leaf diagnostic) and vague labels like "nutritional" that are not a
# named disease. A dropped class is skipped, not folded into another.
AGML_SOURCES = {
    "Project-AgML/rice_leaf_disease_classification": {
        "Leaf_Blast": "RICE_BLAST",
        "Brown_Spot": "RICE_BROWN_SPOT",
        "Bacterial_Leaf_Blight": "RICE_BACT_BLIGHT",
        "Sheath_Blight": "RICE_SHEATH_BLIGHT",
        "Healthy_Rice_Leaf": "RICE_HEALTHY",
        "Leaf_Scald": None,  # no clean taxonomy match; drop rather than mislabel
    },
    "Project-AgML/COLD_chili_leaf_disease_classification": {
        "healthy": "CHILI_HEALTHY",
        "powdery mildew": "CHILI_POWDERY",
        "mites_and_trips": "CHILI_THRIPS",
        "cercospora": "CHILI_CERCOSPORA",
        "nutritional": None,  # deficiency, not a disease
    },
    "Project-AgML/cucumber_disease_classification": {
        "Anthracnose": "CUC_ANTHRAC",
        "Downy_Mildew": "CUC_DOWNY",
        "Fresh_Leaf": "CUC_HEALTHY",
        "Bacterial_Wilt": "CUC_BACT_WILT",
        "Gummy_Stem_Blight": "CUC_GUMMY",
        "Belly_Rot": None,          # fruit
        "Pythium_Fruit_Rot": None,  # fruit
        "Fresh_Cucumber": None,     # fruit
    },
    "Project-AgML/eggplant_leaf_disease_classification": {
        "Cercospora Leaf Spot": "EGG_LEAF_SPOT",
        "Flea Beetles": "EGG_FLEA_BEETLE",
        "Fresh Eggplant Leaf": "EGG_HEALTHY",
        "Leaf Wilt": "EGG_VERTICILLIUM",
        "Powdery Mildew": "EGG_POWDERY",
        "Tobacco Mosaic Virus": "EGG_MOSAIC",
        "Aphids": None,               # limit class count
        "Phytophthora Blight": None,  # limit class count
        "Defect Eggplant": None,      # fruit
        "Fresh Eggplant": None,       # fruit
    },
}


def map_class(raw: str):
    """Return the in-scope label for a raw class string, or None if out of scope."""
    s = (raw or "").lower()
    for subs, label in MAP_RULES:
        if all(x in s for x in subs):
            return label
    return None


def save_jpeg(img, dest: Path, size: int) -> bool:
    """Resize (shortest side -> `size`, capped) and write a quality-85 JPEG."""
    try:
        img = img.convert("RGB")
        w, h = img.size
        scale = size / min(w, h)
        if scale < 1.0:  # only downscale; never invent detail
            img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))))
        img.save(dest, "JPEG", quality=85)
        return True
    except Exception:
        log.debug("skipped an undecodable image", exc_info=True)
        return False


def write_split(buckets: dict, out_root: Path, ratios, size: int, rng) -> Counter:
    """Stratify each label independently so every split sees every class."""
    counts = Counter()
    for label, items in sorted(buckets.items()):
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        parts = {
            "train": items[:n_train],
            "val": items[n_train : n_train + n_val],
            "test": items[n_train + n_val :],
        }
        for split, rows in parts.items():
            dest_dir = out_root / split / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            for i, img in enumerate(rows):
                if save_jpeg(img, dest_dir / f"{i:06d}.jpg", size):
                    counts[f"{split}/{label}"] += 1
        log.info("%-22s n=%5d -> %s", label, n, {k: len(v) for k, v in parts.items()})
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/dataset", help="output root")
    ap.add_argument(
        "--per-class",
        type=int,
        default=0,
        help="cap images per in-scope class (0 = use all)",
    )
    ap.add_argument(
        "--oos-per-class",
        type=int,
        default=2600,
        help="images for EACH of the two out-of-scope classes",
    )
    ap.add_argument("--size", type=int, default=256, help="cached JPEG short side")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--force", action="store_true", help="overwrite existing output")
    args = ap.parse_args()

    from datasets import load_dataset

    rng = random.Random(args.seed)
    out_root = PROJECT_ROOT / args.out
    if out_root.exists():
        if not args.force:
            raise SystemExit(f"{out_root} exists — pass --force to rebuild")
        shutil.rmtree(out_root)

    buckets = defaultdict(list)

    # --- 1. PlantVillage: in-scope labels + other-plant rejects ----------------
    log.info("loading minhhungg/plant-disease-dataset ...")
    plants = load_dataset("minhhungg/plant-disease-dataset", split="train")
    plants = plants.shuffle(seed=args.seed)

    oos_plant_cap = args.oos_per_class
    all_in_scope = {lbl for _, lbl in MAP_RULES}
    for row in plants:
        label = map_class(row["class"])
        if label is None:
            if len(buckets[OOS_OTHER_PLANT]) >= oos_plant_cap:
                continue
            label = OOS_OTHER_PLANT
        elif args.per_class and len(buckets[label]) >= args.per_class:
            continue
        buckets[label].append(row["image"])

        # With every cap satisfied there is nothing left to collect — stop early
        # instead of decoding the remaining rows (matters for smoke runs).
        if args.per_class and len(buckets[OOS_OTHER_PLANT]) >= oos_plant_cap:
            if all(len(buckets[l]) >= args.per_class for l in all_in_scope):
                log.info("all caps satisfied — stopping scan early")
                break

    in_scope = sum(v for k, v in ((k, len(v)) for k, v in buckets.items()) if k != OOS_OTHER_PLANT)
    log.info(
        "plant dataset -> %d in-scope images across %d labels, %d other-plant rejects",
        in_scope,
        len([k for k in buckets if k != OOS_OTHER_PLANT]),
        len(buckets[OOS_OTHER_PLANT]),
    )

    # --- 2. Project-AgML: crops PlantVillage does not cover --------------------
    # rice, chili, cucumber, eggplant. Loaded by full download (streaming these
    # was unreliable), mapped per AGML_SOURCES, fruit/vague classes dropped.
    for name, mapping in AGML_SOURCES.items():
        crop_tag = name.split("/")[-1].split("_")[0]
        log.info("loading %s ...", name)
        try:
            ds = load_dataset(name, split="train")
        except Exception:
            log.warning("could not load %s — skipping this crop", name, exc_info=True)
            continue
        label_names = None
        try:
            label_names = ds.features["label"].names
        except Exception:
            pass
        added = 0
        for row in ds:
            raw = row.get("label")
            if label_names is not None and isinstance(raw, int):
                raw = label_names[raw]
            target = mapping.get(raw, "__unmapped__")
            if target is None:
                continue  # deliberately dropped (fruit / vague)
            if target == "__unmapped__":
                continue  # class we didn't anticipate; ignore rather than guess
            if args.per_class and len(buckets[target]) >= args.per_class:
                continue
            buckets[target].append(row["image"])
            added += 1
        log.info("  %s -> %d images across %d classes", crop_tag, added,
                 len({v for v in mapping.values() if v}))

    # --- 3. tiny-imagenet: not-a-plant rejects --------------------------------
    log.info("loading zh-plus/tiny-imagenet ...")
    objects = load_dataset("zh-plus/tiny-imagenet", split="train", streaming=True)
    objects = objects.shuffle(seed=args.seed, buffer_size=10_000)
    for row in objects:
        if len(buckets[OOS_NOT_PLANT]) >= args.oos_per_class:
            break
        buckets[OOS_NOT_PLANT].append(row["image"])
    log.info("tiny-imagenet -> %d not-a-plant rejects", len(buckets[OOS_NOT_PLANT]))

    # --- 3. Stratified split + JPEG cache -------------------------------------
    counts = write_split(buckets, out_root, (0.70, 0.15, 0.15), args.size, rng)

    labels = sorted(buckets.keys())
    manifest = {
        "sources": {
            "tomato/potato/pepper/maize + OOS_OTHER_PLANT": "minhhungg/plant-disease-dataset",
            "rice/chili/cucumber/eggplant": "Project-AgML/* (see AGML_SOURCES)",
            "OOS_NOT_PLANT": "zh-plus/tiny-imagenet",
        },
        "num_classes": len(labels),
        "classes": labels,
        "in_scope_classes": [
            x for x in labels if x not in (OOS_OTHER_PLANT, OOS_NOT_PLANT)
        ],
        "reject_classes": [OOS_OTHER_PLANT, OOS_NOT_PLANT],
        "split_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
        "image_short_side": args.size,
        "seed": args.seed,
        "counts": dict(sorted(counts.items())),
        "total_images": sum(counts.values()),
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info(
        "DONE — %d images, %d classes -> %s",
        manifest["total_images"],
        len(labels),
        out_root,
    )


if __name__ == "__main__":
    main()
