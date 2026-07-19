"""
AgriDoctor — real (honest) image-classifier training run.

Trains a compact transfer-learning CNN on the public
`minhhungg/plant-disease-dataset` (HuggingFace), restricted to the classes that map
to AgriDoctor's crop taxonomy. Produces GENUINE, reproducible metrics — this is a
small proof-of-concept, NOT the deployed engine (which is the Groq vision model).

Outputs:
  data/models/best_model.pt      - checkpoint (state_dict + class list + metrics)
  data/models/metrics.json       - real validation accuracy / macro-F1 / per-class

Usage:
  .venv/bin/python scripts/train_real.py --per-class 250 --epochs 6
"""

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_real")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "data" / "models"


# --- Map raw PlantVillage class strings -> AgriDoctor taxonomy label ids -------
# Rules are (required substrings) -> label_id, matched case-insensitively against
# the raw class string (which looks like "tomato early blight", "potato late
# blight", "pepper bell bacterial spot", "corn (maize) common rust", ...).
MAP_RULES = [
    (("tomato", "early blight"), "TOM_EARLY_BLIGHT"),
    (("tomato", "late blight"), "TOM_LATE_BLIGHT"),
    (("tomato", "leaf mold"), "TOM_LEAF_MOLD"),
    (("tomato", "septoria"), "TOM_SEPTORIA"),
    (("tomato", "spider mites"), "TOM_SPIDER_MITES"),
    (("tomato", "mosaic"), "TOM_MOSAIC"),
    (("tomato", "bacterial spot"), "TOM_BACL_SPOT"),
    (("tomato", "healthy"), "TOM_HEALTHY"),
    (("potato", "early blight"), "POT_EARLY_BLIGHT"),
    (("potato", "late blight"), "POT_LATE_BLIGHT"),
    (("potato", "healthy"), "POT_HEALTHY"),
    (("pepper", "bacterial spot"), "PEP_BACT_SPOT"),
    (("pepper", "healthy"), "PEP_HEALTHY"),
    (("corn", "common rust"), "MAIZE_RUST"),
    (("maize", "common rust"), "MAIZE_RUST"),
    (("corn", "northern leaf blight"), "MAIZE_NLB"),
    (("maize", "northern leaf blight"), "MAIZE_NLB"),
    (("corn", "gray leaf spot"), "MAIZE_GLS"),
    (("corn", "cercospora"), "MAIZE_GLS"),
    (("corn", "healthy"), "MAIZE_HEALTHY"),
    (("maize", "healthy"), "MAIZE_HEALTHY"),
]


def map_class(raw: str):
    s = (raw or "").lower()
    for subs, label in MAP_RULES:
        if all(x in s for x in subs):
            return label
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=250, help="max images per class")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-per-class", type=int, default=40)
    args = ap.parse_args()

    import numpy as np
    import torch
    import torch.nn as nn
    from datasets import load_dataset
    from sklearn.metrics import accuracy_score, f1_score
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    log.info("device=%s", device)

    log.info("loading dataset (cached)...")
    ds = load_dataset("minhhungg/plant-disease-dataset", split="train")

    # --- build a balanced, mapped subset ---
    raw_counts = Counter(ds["class"])
    log.info("raw classes: %d", len(raw_counts))
    buckets = defaultdict(list)  # label_id -> list of dataset indices
    unmapped = set()
    for i, raw in enumerate(ds["class"]):
        label = map_class(raw)
        if label is None:
            unmapped.add(raw)
            continue
        if len(buckets[label]) < args.per_class:
            buckets[label].append(i)
    # keep only classes with enough samples
    classes = sorted(k for k, v in buckets.items() if len(v) >= args.min_per_class)
    log.info("mapped classes kept (%d): %s", len(classes), classes)
    log.info("per-class counts: %s", {k: len(buckets[k]) for k in classes})
    if len(classes) < 2:
        raise SystemExit("Not enough mapped classes with data; aborting.")

    label_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_label = {i: c for c, i in label_to_idx.items()}

    # train/val split (80/20) per class
    rng = np.random.default_rng(42)
    train_idx, val_idx = [], []
    for c in classes:
        idxs = buckets[c][:]
        rng.shuffle(idxs)
        cut = int(0.8 * len(idxs))
        train_idx += [(i, label_to_idx[c]) for i in idxs[:cut]]
        val_idx += [(i, label_to_idx[c]) for i in idxs[cut:]]
    rng.shuffle(train_idx)
    log.info("train=%d val=%d", len(train_idx), len(val_idx))

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            norm,
        ]
    )
    eval_tf = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), norm]
    )

    class HFSubset(Dataset):
        def __init__(self, pairs, tf):
            self.pairs = pairs
            self.tf = tf

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, k):
            ds_i, y = self.pairs[k]
            img = ds[ds_i]["image"].convert("RGB")
            return self.tf(img), y

    train_dl = DataLoader(
        HFSubset(train_idx, train_tf), batch_size=args.batch_size, shuffle=True
    )
    val_dl = DataLoader(HFSubset(val_idx, eval_tf), batch_size=args.batch_size)

    # --- compact transfer-learning model ---
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_f = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_f, len(classes))
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_state = None
    best_report = {}
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
        # validate
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                out = model(xb.to(device))
                preds += out.argmax(1).cpu().tolist()
                gts += yb.tolist()
        acc = accuracy_score(gts, preds)
        macro_f1 = f1_score(gts, preds, average="macro")
        log.info(
            "epoch %d loss=%.4f val_acc=%.4f val_macroF1=%.4f",
            epoch,
            running / max(1, len(train_idx)),
            acc,
            macro_f1,
        )
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            per_class_f1 = f1_score(gts, preds, average=None, labels=list(range(len(classes))))
            best_report = {
                "val_accuracy": round(float(acc), 4),
                "val_macro_f1": round(float(macro_f1), 4),
                "epoch": epoch,
                "per_class_f1": {
                    idx_to_label[i]: round(float(f), 4)
                    for i, f in enumerate(per_class_f1)
                },
            }

    # --- save ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "arch": "mobilenet_v3_small",
        "model_state_dict": best_state,
        "classes": classes,
        "idx_to_label": idx_to_label,
        "num_classes": len(classes),
        "best_f1": best_f1,
        "input_size": 224,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
    }
    torch.save(ckpt, MODELS_DIR / "best_model.pt")
    metrics = {
        "dataset": "minhhungg/plant-disease-dataset (HuggingFace)",
        "backbone": "mobilenet_v3_small (ImageNet transfer learning)",
        "device": device,
        "num_classes": len(classes),
        "classes": classes,
        "train_images": len(train_idx),
        "val_images": len(val_idx),
        "epochs": args.epochs,
        **best_report,
        "note": "Small honest proof-of-concept. NOT the deployed engine (Groq VLM). "
        "PlantVillage-style images have plain backgrounds, so field accuracy is lower.",
    }
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("SAVED best_f1=%.4f -> %s", best_f1, MODELS_DIR / "metrics.json")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
