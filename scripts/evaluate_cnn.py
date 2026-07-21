"""
AgriDoctor — held-out evaluation of the local CNN.

Runs the test split (never seen during training or model selection) and reports
four things, in increasing order of how much they actually matter:

  1. Classification quality — accuracy, macro-F1, per-class precision/recall,
     confusion matrix.
  2. Calibration — reliability of the confidence number the server thresholds on.
  3. Safety gate — how often an out-of-scope image is wrongly accepted as a
     confident crop diagnosis. This is the number that decides whether the CNN is
     safe to run as the primary engine, and no accuracy figure substitutes for it.
  4. Routing mix — what share of traffic is answered locally vs escalated, i.e.
     how many API calls the local model actually saves.

Also benchmarks single-image latency, since "does this beat waiting on a
rate-limited API" is the practical claim being made.

Usage:
    .venv/bin/python scripts/evaluate_cnn.py
    .venv/bin/python scripts/evaluate_cnn.py --model data/models/smoke_cnn.pt --data data/dataset_smoke
"""

import argparse
import json
import logging
import time
from functools import partial
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("evaluate")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "data" / "models"

OOS_OTHER_PLANT = "OOS_OTHER_PLANT"
OOS_NOT_PLANT = "OOS_NOT_PLANT"
REJECT_CLASSES = {OOS_OTHER_PLANT, OOS_NOT_PLANT}


# --------------------------------------------------------------------------- #
# Corruption suite
#
# PlantVillage images are studio-clean, so clean-test accuracy overstates field
# performance — the well-documented weakness of every model trained on it, and
# the one the project report already flags. These corruptions approximate what a
# phone camera in a field actually produces, and the gap between clean and
# corrupted accuracy is an honest measure of that generalisation gap.
# --------------------------------------------------------------------------- #
def _blur(img, radius=2.0):
    from PIL import ImageFilter

    return img.filter(ImageFilter.GaussianBlur(radius))


def _noise(img, sigma=0.06):
    import numpy as np
    from PIL import Image

    arr = np.asarray(img).astype(np.float32) / 255.0
    rng = np.random.default_rng(0)  # fixed so the benchmark is reproducible
    arr = np.clip(arr + rng.normal(0, sigma, arr.shape), 0, 1)
    return Image.fromarray((arr * 255).astype("uint8"))


def _brightness(img, factor):
    from PIL import ImageEnhance

    return ImageEnhance.Brightness(img).enhance(factor)


def _jpeg(img, quality=25):
    import io

    from PIL import Image

    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _occlude(img, frac=0.25):
    """Black box over part of the leaf — a finger, shadow, or another leaf."""
    from PIL import ImageDraw

    img = img.copy()
    w, h = img.size
    bw, bh = int(w * frac), int(h * frac)
    ImageDraw.Draw(img).rectangle(
        [(w - bw) // 2, (h - bh) // 2, (w + bw) // 2, (h + bh) // 2], fill=(0, 0, 0)
    )
    return img


def _lowres(img, scale=0.25):
    from PIL import Image

    w, h = img.size
    small = img.resize((max(8, int(w * scale)), max(8, int(h * scale))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


# functools.partial rather than lambdas: DataLoader workers pickle the dataset,
# and a lambda defined at module scope cannot be pickled.
CORRUPTIONS = {
    "blur": partial(_blur, radius=2.0),
    "heavy_blur": partial(_blur, radius=4.0),
    "noise": partial(_noise, sigma=0.06),
    "dark": partial(_brightness, factor=0.5),
    "bright": partial(_brightness, factor=1.6),
    "jpeg_artifacts": partial(_jpeg, quality=25),
    "occlusion": partial(_occlude, frac=0.25),
    "low_resolution": partial(_lowres, scale=0.25),
}


class CorruptedDataset:
    """Wraps an ImageFolder, applying a corruption before the eval transform."""

    def __init__(self, samples, corruption, transform):
        self.samples = samples
        self.corruption = corruption
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        from PIL import Image

        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.transform(self.corruption(img)), label


def plot_confusion(cm, classes, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Row-normalise: with uneven class counts, raw counts hide the error pattern.
    cm = cm.astype(float)
    cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1e-9)

    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=7)
    ax.set_yticklabels(classes, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix (row-normalised) — held-out test split")

    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j] >= 0.01:
                ax.text(
                    j, i, f"{cm[i, j]:.2f}",
                    ha="center", va="center", fontsize=5.5,
                    color="white" if cm[i, j] > 0.5 else "black",
                )

    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="data/models/agridoctor_cnn.pt")
    ap.add_argument("--data", default="data/dataset")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--latency-samples", type=int, default=30)
    ap.add_argument("--skip-robustness", action="store_true",
                    help="skip the corruption suite (it re-runs the test set 8x)")
    ap.add_argument("--out", default="eval_report.json")
    args = ap.parse_args()

    import numpy as np
    import torch
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )
    from torch.utils.data import DataLoader
    from torchvision import transforms as T
    from torchvision.datasets import ImageFolder

    import sys

    sys.path.insert(0, str(PROJECT_ROOT))
    from backend.ai.cnn_predictor import (
        DECISION_DIAGNOSIS,
        DECISION_ESCALATE,
        DECISION_NOT_PLANT,
        CNNPredictor,
    )

    model_path = PROJECT_ROOT / args.model
    split_dir = PROJECT_ROOT / args.data / args.split
    if not model_path.exists():
        raise SystemExit(f"no checkpoint at {model_path} — train first")
    if not split_dir.exists():
        raise SystemExit(f"no {args.split} split at {split_dir}")

    # Load through the serving predictor so we evaluate exactly what ships,
    # including temperature scaling and the decision thresholds.
    predictor = CNNPredictor(str(model_path))
    if not predictor._load():  # noqa: SLF001
        raise SystemExit("failed to load checkpoint")
    classes = predictor.classes
    device = predictor._device  # noqa: SLF001
    model = predictor._model  # noqa: SLF001
    log.info("loaded %s | %d classes | device=%s", model_path.name, len(classes), device)

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    size = int(ckpt.get("input_size", 224))
    temperature = float(ckpt.get("temperature", 1.0)) or 1.0
    eval_tf = T.Compose([
        T.Resize(int(size * 1.14)),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(ckpt.get("normalize_mean", [0.485, 0.456, 0.406]),
                    ckpt.get("normalize_std", [0.229, 0.224, 0.225])),
    ])

    ds = ImageFolder(split_dir, transform=eval_tf)
    if ds.classes != classes:
        log.warning("class order mismatch between checkpoint and %s", split_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers)
    log.info("%s split: %d images", args.split, len(ds))

    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device)) / temperature
            all_probs.append(torch.softmax(logits, dim=1).cpu())
            all_labels.append(labels)
    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    preds = probs.argmax(axis=1)
    conf = probs.max(axis=1)

    # --- 1. Classification quality --------------------------------------------
    acc = float(accuracy_score(labels, preds))
    macro_f1 = float(f1_score(labels, preds, average="macro"))
    report = classification_report(labels, preds, target_names=classes,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(labels, preds, labels=list(range(len(classes))))
    plot_confusion(cm, classes, MODELS_DIR / "confusion_matrix.png")
    log.info("accuracy=%.4f  macro_f1=%.4f", acc, macro_f1)

    # --- 2. Calibration -------------------------------------------------------
    correct = (preds == labels).astype(float)
    edges = np.linspace(0, 1, 16)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            ece += (m.sum() / len(conf)) * abs(correct[m].mean() - conf[m].mean())
    log.info("ECE=%.4f  (temperature=%.4f)", ece, temperature)

    # --- 3. Safety gate -------------------------------------------------------
    idx_of = {c: i for i, c in enumerate(classes)}
    reject_idx = {idx_of[c] for c in REJECT_CLASSES if c in idx_of}
    in_scope_idx = set(range(len(classes))) - reject_idx

    thr = predictor.confidence_threshold
    margin_thr = predictor.margin_threshold
    sorted_probs = np.sort(probs, axis=1)
    margins = sorted_probs[:, -1] - sorted_probs[:, -2]
    would_diagnose = (
        np.isin(preds, list(in_scope_idx)) & (conf >= thr) & (margins >= margin_thr)
    )

    oos_mask = np.isin(labels, list(reject_idx))
    not_plant_mask = labels == idx_of.get(OOS_NOT_PLANT, -1)
    safety = {
        "threshold_used": thr,
        "margin_threshold_used": margin_thr,
        "oos_test_images": int(oos_mask.sum()),
        # THE key number: out-of-scope images that would receive a confident
        # crop diagnosis instead of being rejected or escalated.
        "false_accept_count": int(would_diagnose[oos_mask].sum()),
        "false_accept_rate": round(float(would_diagnose[oos_mask].mean()), 4)
        if oos_mask.sum() else None,
        "not_plant_recall": round(
            float((preds[not_plant_mask] == idx_of[OOS_NOT_PLANT]).mean()), 4
        ) if not_plant_mask.sum() else None,
        "in_scope_answered_locally": round(
            float(would_diagnose[~oos_mask].mean()), 4
        ) if (~oos_mask).sum() else None,
    }
    log.info("safety gate: false-accept rate on out-of-scope = %s",
             safety["false_accept_rate"])

    # --- 3b. OOD detection quality (threshold-free) ---------------------------
    # AUROC over the reject-class probability mass answers "can the model
    # separate in-scope from out-of-scope at all", independent of where the
    # threshold happens to sit. A weak AUROC means no threshold choice will save
    # the safety gate; a strong one means the gate is just a tuning question.
    from sklearn.metrics import roc_auc_score

    reject_mass = probs[:, sorted(reject_idx)].sum(axis=1) if reject_idx else None
    ood_metrics = {}
    if reject_mass is not None and 0 < oos_mask.sum() < len(oos_mask):
        ood_metrics["auroc_oos_vs_in_scope"] = round(
            float(roc_auc_score(oos_mask.astype(int), reject_mass)), 4
        )
        # Split the two reject types: "not a plant" should be far easier to spot
        # than "a leaf of a crop I don't know", and reporting one number hides that.
        for name, cls in ((("not_plant"), OOS_NOT_PLANT), ("other_plant", OOS_OTHER_PLANT)):
            if cls not in idx_of:
                continue
            m = labels == idx_of[cls]
            in_scope_m = ~oos_mask
            sel = m | in_scope_m
            if m.sum():
                ood_metrics[f"auroc_{name}_vs_in_scope"] = round(
                    float(roc_auc_score(m[sel].astype(int), reject_mass[sel])), 4
                )
    log.info("OOD detection: %s", ood_metrics)

    # --- 3c. Threshold sweep --------------------------------------------------
    # The operating point is a real trade-off: a higher bar means fewer wrong
    # local answers but more escalations (and more API calls). This table is the
    # evidence for whichever threshold ships.
    sweep = []
    for t in [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        answered = (
            np.isin(preds, list(in_scope_idx)) & (conf >= t) & (margins >= margin_thr)
        )
        in_answered = answered & ~oos_mask
        sweep.append({
            "threshold": t,
            # Share of in-scope images answered locally (no API call).
            "coverage_in_scope": round(float(answered[~oos_mask].mean()), 4)
            if (~oos_mask).sum() else None,
            # Accuracy among those answered locally — what the farmer actually sees.
            "accuracy_when_answered": round(
                float((preds[in_answered] == labels[in_answered]).mean()), 4
            ) if in_answered.sum() else None,
            # Out-of-scope images wrongly given a confident diagnosis.
            "false_accept_rate": round(float(answered[oos_mask].mean()), 4)
            if oos_mask.sum() else None,
        })
    log.info("threshold sweep computed for %d operating points", len(sweep))

    # --- 4. Routing mix -------------------------------------------------------
    routing = {DECISION_DIAGNOSIS: 0, DECISION_NOT_PLANT: 0, DECISION_ESCALATE: 0}
    for i in range(len(preds)):
        d, _ = predictor._decide(  # noqa: SLF001
            classes[preds[i]], float(conf[i]), float(margins[i])
        )
        routing[d] += 1
    total = max(len(preds), 1)
    routing_pct = {k: round(v / total, 4) for k, v in routing.items()}
    log.info("routing: %s", routing_pct)

    # --- 5. Robustness to field-like corruptions ------------------------------
    robustness = {}
    if not args.skip_robustness:
        in_scope_names = [c for c in classes if c not in REJECT_CLASSES]
        log.info("running corruption suite (%d corruptions)...", len(CORRUPTIONS))
        for name, fn in CORRUPTIONS.items():
            c_ds = CorruptedDataset(ds.samples, fn, eval_tf)
            c_loader = DataLoader(c_ds, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.workers)
            c_logits, c_labels = [], []
            with torch.no_grad():
                for images, lbls in c_loader:
                    c_logits.append(model(images.to(device)).cpu())
                    c_labels.append(lbls)
            c_probs = torch.softmax(torch.cat(c_logits) / temperature, dim=1).numpy()
            c_labels = torch.cat(c_labels).numpy()
            c_preds = c_probs.argmax(axis=1)
            c_acc = float(accuracy_score(c_labels, c_preds))
            c_f1 = float(f1_score(c_labels, c_preds, average="macro"))
            # Also track whether corruption pushes out-of-scope images into a
            # confident diagnosis — degradation that breaks the safety gate is
            # worse than degradation that merely loses accuracy.
            c_margins = np.sort(c_probs, axis=1)[:, -1] - np.sort(c_probs, axis=1)[:, -2]
            c_answered = (
                np.isin(c_preds, list(in_scope_idx))
                & (c_probs.max(axis=1) >= thr)
                & (c_margins >= margin_thr)
            )
            robustness[name] = {
                "accuracy": round(c_acc, 4),
                "macro_f1": round(c_f1, 4),
                "accuracy_drop": round(acc - c_acc, 4),
                "false_accept_rate": round(float(c_answered[oos_mask].mean()), 4)
                if oos_mask.sum() else None,
            }
            log.info("  %-16s acc=%.4f (drop %.4f)  macroF1=%.4f",
                     name, c_acc, acc - c_acc, c_f1)
        if robustness:
            worst = min(robustness.items(), key=lambda kv: kv[1]["accuracy"])
            robustness["_summary"] = {
                "clean_accuracy": round(acc, 4),
                "mean_corrupted_accuracy": round(
                    float(np.mean([v["accuracy"] for k, v in robustness.items()
                                   if not k.startswith("_")])), 4
                ),
                "worst_corruption": worst[0],
                "worst_accuracy": worst[1]["accuracy"],
                "in_scope_classes": len(in_scope_names),
            }
            log.info("robustness summary: %s", robustness["_summary"])

    # --- Latency --------------------------------------------------------------
    sample_paths = [p for p, _ in ds.samples[: args.latency_samples]]
    # Warm up first: the initial call pays lazy model load and MPS kernel
    # compilation, which would otherwise dominate the mean.
    if sample_paths:
        predictor.predict(Path(sample_paths[0]).read_bytes())
    latencies = []
    for p in sample_paths:
        raw = Path(p).read_bytes()
        t0 = time.perf_counter()
        predictor.predict(raw)
        latencies.append((time.perf_counter() - t0) * 1000)
    latency = {
        "samples": len(latencies),
        "tta_enabled": predictor.use_tta,
        "mean_ms": round(float(np.mean(latencies)), 1),
        "p50_ms": round(float(np.percentile(latencies, 50)), 1),
        "p95_ms": round(float(np.percentile(latencies, 95)), 1),
    }
    log.info("latency: %s", latency)

    out = {
        "model": str(args.model),
        "checkpoint_arch": ckpt.get("arch"),
        "split": args.split,
        "images": len(ds),
        "num_classes": len(classes),
        "classes": classes,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "calibration": {"temperature": round(temperature, 4), "ece": round(ece, 4)},
        "safety_gate": safety,
        "ood_detection": ood_metrics,
        "threshold_sweep": sweep,
        "routing_mix": routing_pct,
        "robustness": robustness,
        "latency": latency,
        "per_class": {
            c: {
                "precision": round(report[c]["precision"], 4),
                "recall": round(report[c]["recall"], 4),
                "f1": round(report[c]["f1-score"], 4),
                "support": int(report[c]["support"]),
            }
            for c in classes if c in report
        },
        "confusion_matrix": cm.tolist(),
    }
    (MODELS_DIR / args.out).write_text(json.dumps(out, indent=2))
    log.info("wrote %s and confusion_matrix.png", MODELS_DIR / args.out)


if __name__ == "__main__":
    main()
