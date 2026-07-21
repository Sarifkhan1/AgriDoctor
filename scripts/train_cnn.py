"""
AgriDoctor — train the local crop-disease CNN.

Transfer-learns an ImageNet backbone (EfficientNet-B0 by default) over the corpus
built by `scripts/prepare_dataset.py`: 19 in-scope disease/healthy classes plus
the two out-of-scope reject classes that let the model decline an image instead of
forcing it into a disease bucket.

What this run does beyond a plain fine-tune, and why:

  * Two-stage schedule — head-only warmup (frozen backbone) then full fine-tune at
    a lower LR. Stops the randomly-initialised head from wrecking pretrained
    features with large early gradients.
  * Heavy augmentation (RandAugment, flips, rotation, colour jitter, random
    resized crop, random erasing). PlantVillage images are shot on plain lab
    backgrounds; without this the model learns the background, not the lesion, and
    collapses on real field photos.
  * Resolution jitter on EVERY class. OOS_NOT_PLANT comes from 64px tiny-imagenet
    while the leaves are 256px, so sharpness alone would separate them — the model
    would "reject" by detecting blur rather than by understanding content. Random
    downscale-then-upscale destroys that shortcut.
  * Class-balanced sampling + label smoothing, since class counts are uneven.
  * Temperature scaling fitted on the validation split. Raw softmax from a
    fine-tuned CNN is badly overconfident, and serving thresholds a miscalibrated
    confidence is how you ship a wrong diagnosis at "97%".
  * Selects the best epoch by macro-F1, not accuracy — the reject classes matter
    as much as the common diseases.

Outputs:
    data/models/agridoctor_cnn.pt     checkpoint (weights + classes + calibration)
    data/models/train_metrics.json    per-epoch history + final validation report
    data/models/training_curves.png   loss / macro-F1 curves

Usage:
    .venv/bin/python scripts/train_cnn.py                       # full run
    .venv/bin/python scripts/train_cnn.py --data data/dataset_smoke --epochs 2
"""

import argparse
import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "data" / "models"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ResolutionJitter:
    """Randomly degrade then restore resolution.

    Removes the sharpness shortcut between the 256px leaf sources and the 64px
    tiny-imagenet rejects. Applied to all classes so it carries no label signal.
    """

    def __init__(self, p=0.5, min_scale=0.25):
        self.p = p
        self.min_scale = min_scale

    def __call__(self, img):
        import random

        from PIL import Image

        if random.random() > self.p:
            return img
        w, h = img.size
        s = random.uniform(self.min_scale, 1.0)
        small = img.resize((max(8, int(w * s)), max(8, int(h * s))), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)


def build_transforms(size: int):
    from torchvision import transforms as T

    train = T.Compose(
        [
            ResolutionJitter(p=0.5, min_scale=0.25),
            T.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(p=0.2),
            T.RandomRotation(25),
            T.RandAugment(num_ops=2, magnitude=7),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ]
    )
    # Eval: deterministic resize + centre crop, no jitter.
    evaluate = T.Compose(
        [
            T.Resize(int(size * 1.14)),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train, evaluate


def build_model(arch: str, num_classes: int, dropout: float):
    import torch.nn as nn
    from torchvision import models

    if arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_f = m.classifier[-1].in_features
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
        head, backbone_last = m.classifier, "features"
    elif arch == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, num_classes))
        head, backbone_last = m.fc, "layer4"
    elif arch == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        )
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
        head, backbone_last = m.classifier, "features"
    else:
        raise ValueError(f"unsupported arch: {arch}")
    return m, head, backbone_last


def set_backbone_trainable(model, head, trainable: bool):
    head_params = {id(p) for p in head.parameters()}
    for p in model.parameters():
        if id(p) not in head_params:
            p.requires_grad = trainable


def make_balanced_sampler(dataset):
    """Sample inversely to class frequency so rare classes aren't drowned out."""
    from collections import Counter

    import torch
    from torch.utils.data import WeightedRandomSampler

    targets = [s[1] for s in dataset.samples]
    freq = Counter(targets)
    weights = [1.0 / freq[t] for t in targets]
    return WeightedRandomSampler(
        torch.DoubleTensor(weights), num_samples=len(targets), replacement=True
    )


def run_epoch(model, loader, device, criterion, optimizer=None):
    """One pass. Training when `optimizer` is given, else evaluation."""
    import torch

    training = optimizer is not None
    model.train(training)
    total_loss, seen = 0.0, 0
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        seen += images.size(0)
        all_preds.append(logits.argmax(1).detach().cpu())
        all_labels.append(labels.detach().cpu())
        if not training:
            all_probs.append(logits.detach().cpu())

    preds = torch.cat(all_preds).numpy()
    labels_np = torch.cat(all_labels).numpy()
    logits_cat = torch.cat(all_probs) if all_probs else None
    return total_loss / max(seen, 1), preds, labels_np, logits_cat


def fit_temperature(logits, labels):
    """Optimise a single scalar T minimising NLL of softmax(logits / T).

    Standard temperature scaling (Guo et al., 2017). Monotone in the logits, so it
    changes no prediction — only the confidence we threshold on at serve time.
    """
    import torch
    import torch.nn as nn

    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=60)
    nll = nn.CrossEntropyLoss()
    labels_t = torch.as_tensor(labels, dtype=torch.long)

    def closure():
        optimizer.zero_grad()
        loss = nll(logits / log_t.exp(), labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().item())


def expected_calibration_error(probs, labels, bins=15):
    """Mean gap between confidence and accuracy across confidence bins."""
    import numpy as np

    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(conf)) * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def plot_curves(history, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax1.plot(epochs, [h["val_loss"] for h in history], label="val")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, [h["val_macro_f1"] for h in history], color="green", label="val macro-F1")
    ax2.plot(epochs, [h["val_accuracy"] for h in history], color="orange", label="val accuracy")
    ax2.set_xlabel("epoch")
    ax2.set_title("Validation performance")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset")
    ap.add_argument("--arch", default="efficientnet_b0",
                    choices=["efficientnet_b0", "resnet50", "mobilenet_v3_large"])
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--warmup-epochs", type=int, default=2,
                    help="head-only epochs before unfreezing the backbone")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4, help="LR for the fine-tune stage")
    ap.add_argument("--head-lr", type=float, default=1e-3, help="LR for the warmup stage")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=5, help="early-stop patience")
    ap.add_argument("--out", default="agridoctor_cnn.pt")
    args = ap.parse_args()

    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder

    data_root = PROJECT_ROOT / args.data
    if not (data_root / "train").exists():
        raise SystemExit(f"no training data at {data_root} — run scripts/prepare_dataset.py first")

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    log.info("device=%s arch=%s", device, args.arch)

    train_tf, eval_tf = build_transforms(args.size)
    train_ds = ImageFolder(data_root / "train", transform=train_tf)
    val_ds = ImageFolder(data_root / "val", transform=eval_tf)
    classes = train_ds.classes
    log.info("%d classes | train=%d val=%d", len(classes), len(train_ds), len(val_ds))

    # persistent_workers only helps when workers > 0; MPS dislikes pin_memory.
    loader_kw = dict(
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=(device == "cuda"),
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=make_balanced_sampler(train_ds), **loader_kw
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kw)

    model, head, _ = build_model(args.arch, len(classes), args.dropout)
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    history, best = [], {"macro_f1": -1.0, "epoch": -1}
    stale = 0
    optimizer = scheduler = None
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        # --- stage switch: head-only warmup -> full fine-tune ---
        if epoch == 1:
            log.info("stage 1: head-only warmup (backbone frozen)")
            set_backbone_trainable(model, head, False)
            optimizer = torch.optim.AdamW(
                head.parameters(), lr=args.head_lr, weight_decay=args.weight_decay
            )
            scheduler = None
        elif epoch == args.warmup_epochs + 1:
            log.info("stage 2: unfreezing backbone, full fine-tune @ lr=%g", args.lr)
            set_backbone_trainable(model, head, True)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, args.epochs - args.warmup_epochs)
            )

        train_loss, *_ = run_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_preds, val_labels, val_logits = run_epoch(
            model, val_loader, device, criterion
        )
        if scheduler is not None:
            scheduler.step()

        acc = float(accuracy_score(val_labels, val_preds))
        macro_f1 = float(f1_score(val_labels, val_preds, average="macro"))
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(acc, 4),
            "val_macro_f1": round(macro_f1, 4),
            "lr": optimizer.param_groups[0]["lr"],
        })
        log.info(
            "epoch %2d/%d  train_loss=%.4f  val_loss=%.4f  acc=%.4f  macroF1=%.4f",
            epoch, args.epochs, train_loss, val_loss, acc, macro_f1,
        )

        if macro_f1 > best["macro_f1"]:
            best = {
                "macro_f1": macro_f1,
                "accuracy": acc,
                "epoch": epoch,
                "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "logits": val_logits,
                "labels": val_labels,
                "preds": val_preds,
            }
            stale = 0
            log.info("  ^ new best")
        else:
            stale += 1
            if stale >= args.patience:
                log.info("early stop: no improvement in %d epochs", args.patience)
                break

    # --- Calibration on the best epoch's validation logits --------------------
    temperature = fit_temperature(best["logits"], best["labels"])
    probs_raw = torch.softmax(best["logits"], dim=1).numpy()
    probs_cal = torch.softmax(best["logits"] / temperature, dim=1).numpy()
    ece_raw = expected_calibration_error(probs_raw, best["labels"])
    ece_cal = expected_calibration_error(probs_cal, best["labels"])
    log.info(
        "temperature=%.4f  ECE %.4f -> %.4f", temperature, ece_raw, ece_cal
    )

    report = classification_report(
        best["labels"], best["preds"], target_names=classes,
        output_dict=True, zero_division=0,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = MODELS_DIR / args.out
    torch.save(
        {
            "arch": args.arch,
            "num_classes": len(classes),
            "classes": classes,
            "idx_to_label": {i: c for i, c in enumerate(classes)},
            "model_state_dict": best["state_dict"],
            "input_size": args.size,
            "normalize_mean": IMAGENET_MEAN,
            "normalize_std": IMAGENET_STD,
            "temperature": temperature,
            "best_epoch": best["epoch"],
            "best_macro_f1": best["macro_f1"],
            "dropout": args.dropout,
        },
        ckpt_path,
    )

    metrics = {
        "arch": args.arch,
        "dataset_root": str(args.data),
        "device": device,
        "num_classes": len(classes),
        "classes": classes,
        "train_images": len(train_ds),
        "val_images": len(val_ds),
        "epochs_run": len(history),
        "best_epoch": best["epoch"],
        "val_accuracy": round(best["accuracy"], 4),
        "val_macro_f1": round(best["macro_f1"], 4),
        "calibration": {
            "temperature": round(temperature, 4),
            "ece_before": round(ece_raw, 4),
            "ece_after": round(ece_cal, 4),
        },
        "hyperparameters": vars(args),
        "history": history,
        "per_class": {
            c: {
                "precision": round(report[c]["precision"], 4),
                "recall": round(report[c]["recall"], 4),
                "f1": round(report[c]["f1-score"], 4),
                "support": int(report[c]["support"]),
            }
            for c in classes
        },
        "wall_clock_seconds": round(time.time() - started, 1),
    }
    (MODELS_DIR / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_curves(history, MODELS_DIR / "training_curves.png")

    log.info(
        "DONE — best epoch %d, val macro-F1 %.4f -> %s",
        best["epoch"], best["macro_f1"], ckpt_path,
    )


if __name__ == "__main__":
    main()
