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
  * MixUp / CutMix. Blending images and their labels stops the network betting
    everything on one dominant cue, which is exactly the PlantVillage failure
    mode (memorise the plain background, ignore the lesion). It also flattens
    overconfidence, which matters here because the serving layer routes on the
    confidence value itself.
  * An exponential moving average of the weights, evaluated alongside the raw
    weights. The EMA copy is usually a little better and noticeably more stable;
    whichever genuinely scores higher on validation is what gets saved.
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
import contextlib
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


class ModelEMA:
    """Exponential moving average of model weights.

    Keeps a slowly-updated shadow copy of every float parameter. The averaged
    weights sit nearer the centre of the loss basin than any single SGD iterate,
    which usually generalises slightly better and swings far less between epochs.
    Cheap: one extra copy of the weights and a lerp per step.
    """

    def __init__(self, model, decay=0.999):
        import copy

        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @property
    def module(self):
        return self.ema

    def update(self, model):
        import torch

        with torch.no_grad():
            for ema_v, model_v in zip(
                self.ema.state_dict().values(), model.state_dict().values()
            ):
                if ema_v.dtype.is_floating_point:
                    ema_v.mul_(self.decay).add_(model_v.detach(), alpha=1 - self.decay)
                else:
                    ema_v.copy_(model_v)  # counters/num_batches_tracked


def mix_batch(images, labels, num_classes, mixup_alpha, cutmix_alpha, smoothing):
    """Apply MixUp or CutMix, returning soft targets.

    Returns (images, soft_targets). Soft targets carry the label smoothing too,
    so the caller uses a plain soft-target cross-entropy for every batch and the
    two regularisers compose instead of fighting.
    """
    import numpy as np
    import torch

    batch = images.size(0)
    off = smoothing / num_classes
    targets = torch.full((batch, num_classes), off, device=labels.device)
    targets.scatter_(1, labels.unsqueeze(1), 1.0 - smoothing + off)

    use_mixup = mixup_alpha > 0
    use_cutmix = cutmix_alpha > 0
    if not (use_mixup or use_cutmix):
        return images, targets

    # Pick one of the two per batch so their effects stay interpretable.
    do_cutmix = use_cutmix and (not use_mixup or np.random.rand() < 0.5)
    alpha = cutmix_alpha if do_cutmix else mixup_alpha
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(batch, device=images.device)

    if do_cutmix:
        _, _, h, w = images.shape
        ratio = np.sqrt(1.0 - lam)
        cut_h, cut_w = int(h * ratio), int(w * ratio)
        cy, cx = np.random.randint(h), np.random.randint(w)
        y1, y2 = np.clip([cy - cut_h // 2, cy + cut_h // 2], 0, h)
        x1, x2 = np.clip([cx - cut_w // 2, cx + cut_w // 2], 0, w)
        images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
        # Recompute lam from the true pasted area (rounding makes it drift).
        lam = 1.0 - ((y2 - y1) * (x2 - x1) / (h * w))
    else:
        images = lam * images + (1.0 - lam) * images[perm]

    targets = lam * targets + (1.0 - lam) * targets[perm]
    return images, targets


def soft_target_cross_entropy(logits, targets):
    import torch

    return -(targets * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


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


def train_epoch(model, loader, device, optimizer, scheduler, num_classes, args, ema):
    """One training pass with MixUp/CutMix, per-step cosine schedule and EMA."""
    import torch

    model.train()
    total_loss, seen = 0.0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        images, targets = mix_batch(
            images, labels, num_classes,
            args.mixup, args.cutmix, args.label_smoothing,
        )

        logits = model(images)
        loss = soft_target_cross_entropy(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()  # stepped per batch, not per epoch
        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * images.size(0)
        seen += images.size(0)

    return total_loss / max(seen, 1)


@contextlib.contextmanager
def _eval_mode(model):
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


def evaluate(model, loader, device, num_classes, smoothing=0.0):
    """Evaluation pass. Returns (loss, preds, labels, logits)."""
    import torch

    all_logits, all_labels = [], []
    with _eval_mode(model), torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device, non_blocking=True))
            all_logits.append(logits.cpu())
            all_labels.append(labels)

    logits_cat = torch.cat(all_logits)
    labels_cat = torch.cat(all_labels)
    loss = torch.nn.functional.cross_entropy(
        logits_cat, labels_cat, label_smoothing=smoothing
    ).item()
    return loss, logits_cat.argmax(1).numpy(), labels_cat.numpy(), logits_cat


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
    ap.add_argument("--mixup", type=float, default=0.2,
                    help="MixUp alpha (0 disables)")
    ap.add_argument("--cutmix", type=float, default=1.0,
                    help="CutMix alpha (0 disables)")
    ap.add_argument("--ema-decay", type=float, default=0.999,
                    help="EMA decay (0 disables the averaged copy)")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=5, help="early-stop patience")
    ap.add_argument(
        "--init-from",
        default="",
        help=(
            "start from an existing checkpoint's weights instead of ImageNet. "
            "Used for progressive resizing: train at 224, then fine-tune the "
            "result at a larger --size for a few epochs."
        ),
    )
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

    if args.init_from:
        init_path = PROJECT_ROOT / args.init_from
        prior = torch.load(init_path, map_location="cpu", weights_only=False)
        if prior.get("classes") and list(prior["classes"]) != list(classes):
            raise SystemExit(
                f"{init_path} was trained on different classes — refusing to load "
                "weights whose output layer means something else"
            )
        model.load_state_dict(prior["model_state_dict"])
        log.info(
            "initialised from %s (macro-F1 %.4f, trained at %dpx) -> now %dpx",
            init_path.name,
            float(prior.get("best_macro_f1", 0.0)),
            int(prior.get("input_size", 0)),
            args.size,
        )

    model.to(device)

    ema = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None
    if ema is not None:
        ema.ema.to(device)
    log.info(
        "regularisation: mixup=%.2f cutmix=%.2f smoothing=%.2f ema=%s",
        args.mixup, args.cutmix, args.label_smoothing,
        args.ema_decay if ema else "off",
    )

    history, best = [], {"macro_f1": -1.0, "epoch": -1}
    stale = 0
    optimizer = scheduler = None
    started = time.time()
    steps_per_epoch = max(1, len(train_loader))

    for epoch in range(1, args.epochs + 1):
        # --- stage switch: head-only warmup -> full fine-tune ---
        # `--warmup-epochs 0` skips straight to the full fine-tune, which is what
        # you want when initialising from an already-trained checkpoint: the head
        # is not random, so freezing the backbone to protect it buys nothing.
        if epoch == 1 and args.warmup_epochs > 0:
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
            # Cosine decay stepped per batch: smoother than per-epoch jumps, and
            # the schedule no longer depends on how long early stopping lets it run.
            total_steps = max(1, (args.epochs - args.warmup_epochs) * steps_per_epoch)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps
            )

        train_loss = train_epoch(
            model, train_loader, device, optimizer, scheduler,
            len(classes), args, ema,
        )

        # Evaluate the raw weights and the EMA copy; keep whichever is genuinely
        # better rather than assuming EMA always wins.
        val_loss, val_preds, val_labels, val_logits = evaluate(
            model, val_loader, device, len(classes)
        )
        acc = float(accuracy_score(val_labels, val_preds))
        macro_f1 = float(f1_score(val_labels, val_preds, average="macro"))
        chosen, chosen_state = "raw", model.state_dict()

        if ema is not None:
            e_loss, e_preds, e_labels, e_logits = evaluate(
                ema.module, val_loader, device, len(classes)
            )
            e_f1 = float(f1_score(e_labels, e_preds, average="macro"))
            if e_f1 > macro_f1:
                chosen, chosen_state = "ema", ema.module.state_dict()
                val_loss, val_preds, val_labels, val_logits = (
                    e_loss, e_preds, e_labels, e_logits
                )
                acc = float(accuracy_score(e_labels, e_preds))
                macro_f1 = e_f1

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(acc, 4),
            "val_macro_f1": round(macro_f1, 4),
            "weights": chosen,
            "lr": optimizer.param_groups[0]["lr"],
        })
        log.info(
            "epoch %2d/%d  train_loss=%.4f  val_loss=%.4f  acc=%.4f  macroF1=%.4f  [%s]",
            epoch, args.epochs, train_loss, val_loss, acc, macro_f1, chosen,
        )

        if macro_f1 > best["macro_f1"]:
            best = {
                "macro_f1": macro_f1,
                "accuracy": acc,
                "epoch": epoch,
                "weights": chosen,
                # Must snapshot the weights that produced these metrics — which is
                # the EMA copy whenever EMA won this epoch, not the raw model.
                "state_dict": {
                    k: v.detach().cpu().clone() for k, v in chosen_state.items()
                },
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
            "weights_source": best.get("weights", "raw"),
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
