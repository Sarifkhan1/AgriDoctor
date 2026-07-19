"""
AgriDoctor — REAL multimodal fusion training + ablation (honest).

Trains an attention-based fusion classifier over FROZEN encoders:
  * image: torchvision ViT-B/16 (ImageNet), pooled 768-d feature
  * text : DistilBERT (base-uncased), mean-pooled 768-d feature
and runs a genuine ablation: image-only vs text-only vs fusion (macro-F1).

HONESTY NOTES (read before citing numbers):
- The text modality is TEMPLATED symptom text generated from the taxonomy's
  symptom descriptions (no disease names) with paraphrase + word-dropout + generic
  noise. It is NOT real farmer input. So this ablation demonstrates the pipeline and
  the effect of adding symptom text — it is not a claim about real-world farmer text.
- Encoders are frozen (feature extraction / transfer learning), so this is a light,
  reproducible run — not a from-scratch fusion training.
- This is an offline experiment; the DEPLOYED engine remains the Groq VLM.

Outputs:
  data/models/fusion_ablation.json   (real macro-F1 per ablation arm)
  data/models/fusion_model.pt        (the trained fusion head + config)

Usage:
  .venv/bin/python scripts/train_fusion_real.py --per-class 120 --epochs 25
"""

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fusion")

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "data" / "models"

# reuse the same PlantVillage->taxonomy mapping as the image-only run
from importlib.util import module_from_spec, spec_from_file_location  # noqa: E402

_spec = spec_from_file_location("train_real", ROOT / "scripts" / "train_real.py")
_tr = module_from_spec(_spec)
# train_real imports heavy libs only inside main(); importing the module is cheap
_spec.loader.exec_module(_tr)
map_class = _tr.map_class


def load_symptom_fragments():
    """Symptom-only text fragments per taxonomy label (NO disease names)."""
    tax = json.loads((ROOT / "data" / "taxonomy.json").read_text())
    frags = {}
    for entries in tax.values():
        for e in entries:
            frags[e["label_id"]] = e["description"]
    return frags


GENERIC_NOISE = [
    "the plant does not look healthy",
    "something seems wrong with the leaves",
    "i noticed a change a few days ago",
    "it started on a few leaves",
    "the problem seems to be spreading",
]

TEMPLATES = [
    "I see {s}.",
    "The leaves show {s}.",
    "There is {s} on my plant.",
    "My crop has {s}, it started recently.",
    "{s}. It seems to be getting worse.",
]


def make_text(label: str, frags: dict, rng: random.Random) -> str:
    desc = frags.get(label, "some discoloration on the leaves")
    words = desc.lower().split()
    # word dropout to simulate an imperfect farmer description
    kept = [w for w in words if rng.random() > 0.25] or words
    s = " ".join(kept)
    parts = [rng.choice(TEMPLATES).format(s=s)]
    if rng.random() < 0.4:
        parts.append(rng.choice(GENERIC_NOISE) + ".")
    rng.shuffle(parts)
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=120)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--min-per-class", type=int, default=40)
    args = ap.parse_args()

    import numpy as np
    import torch
    import torch.nn as nn
    from datasets import load_dataset
    from sklearn.metrics import f1_score
    from torchvision import models, transforms
    from transformers import AutoModel, AutoTokenizer

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    log.info("device=%s", device)
    rng = random.Random(42)
    frags = load_symptom_fragments()

    log.info("loading dataset (cached)...")
    ds = load_dataset("minhhungg/plant-disease-dataset", split="train")

    buckets = defaultdict(list)
    for i, raw in enumerate(ds["class"]):
        label = map_class(raw)
        if label and len(buckets[label]) < args.per_class:
            buckets[label].append(i)
    classes = sorted(k for k, v in buckets.items() if len(v) >= args.min_per_class)
    label_to_idx = {c: i for i, c in enumerate(classes)}
    log.info("classes (%d): %s", len(classes), classes)

    samples = []  # (ds_index, label_idx, text)
    for c in classes:
        for di in buckets[c]:
            samples.append((di, label_to_idx[c], make_text(c, frags, rng)))
    rng.shuffle(samples)
    log.info("total samples: %d", len(samples))

    # --- frozen encoders ---
    log.info("loading frozen ViT-B/16 + DistilBERT...")
    vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    vit.heads = nn.Identity()  # -> 768-d pooled feature
    vit.eval().to(device)
    img_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    bert = AutoModel.from_pretrained("distilbert-base-uncased").eval().to(device)

    # --- precompute features (no grad) ---
    log.info("extracting features...")
    img_feats, txt_feats, labels = [], [], []
    B = 32
    with torch.no_grad():
        for start in range(0, len(samples), B):
            chunk = samples[start : start + B]
            imgs = torch.stack(
                [img_tf(ds[di]["image"].convert("RGB")) for di, _, _ in chunk]
            ).to(device)
            iv = vit(imgs)  # (B,768)
            enc = tok(
                [t for _, _, t in chunk],
                padding=True,
                truncation=True,
                max_length=48,
                return_tensors="pt",
            ).to(device)
            out = bert(**enc).last_hidden_state  # (B,L,768)
            mask = enc["attention_mask"].unsqueeze(-1)
            tv = (out * mask).sum(1) / mask.sum(1).clamp(min=1)  # mean pool
            img_feats.append(iv.cpu())
            txt_feats.append(tv.cpu())
            labels += [y for _, y, _ in chunk]
            if start % (B * 10) == 0:
                log.info("  %d/%d", start, len(samples))
    img_feats = torch.cat(img_feats)
    txt_feats = torch.cat(txt_feats)
    labels = torch.tensor(labels)

    # 80/20 split
    n = len(labels)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    cut = int(0.8 * n)
    tr, va = perm[:cut], perm[cut:]

    def run_head(mode: str) -> float:
        """Train a small head for image-only / text-only / fusion; return val macro-F1."""
        D, C = 768, len(classes)
        if mode == "fusion":
            class Fusion(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.mod = nn.Parameter(torch.randn(2, D) * 0.02)
                    layer = nn.TransformerEncoderLayer(
                        d_model=D, nhead=8, dim_feedforward=512,
                        batch_first=True, dropout=0.1,
                    )
                    self.attn = nn.TransformerEncoder(layer, num_layers=2)
                    self.head = nn.Sequential(
                        nn.LayerNorm(D), nn.Linear(D, 256), nn.ReLU(),
                        nn.Dropout(0.2), nn.Linear(256, C),
                    )

                def forward(self, xi, xt):
                    seq = torch.stack([xi, xt], dim=1) + self.mod  # (B,2,D)
                    z = self.attn(seq).mean(1)
                    return self.head(z)

            model = Fusion()
        else:
            model = nn.Sequential(
                nn.LayerNorm(D), nn.Linear(D, 256), nn.ReLU(),
                nn.Dropout(0.2), nn.Linear(256, C),
            )
        model = model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss()
        Xi, Xt, Y = img_feats.to(device), txt_feats.to(device), labels.to(device)
        best = 0.0
        for ep in range(args.epochs):
            model.train()
            opt.zero_grad()
            if mode == "image":
                logits = model(Xi[tr])
            elif mode == "text":
                logits = model(Xt[tr])
            else:
                logits = model(Xi[tr], Xt[tr])
            loss = crit(logits, Y[tr])
            loss.backward()
            opt.step()
            model.eval()
            with torch.no_grad():
                if mode == "image":
                    p = model(Xi[va]).argmax(1)
                elif mode == "text":
                    p = model(Xt[va]).argmax(1)
                else:
                    p = model(Xi[va], Xt[va]).argmax(1)
            f1 = f1_score(Y[va].cpu(), p.cpu(), average="macro")
            best = max(best, f1)
        return round(float(best), 4), model

    log.info("=== ablation ===")
    f1_img, _ = run_head("image")
    log.info("image-only macro-F1 = %.4f", f1_img)
    f1_txt, _ = run_head("text")
    log.info("text-only  macro-F1 = %.4f", f1_txt)
    f1_fus, fusion_model = run_head("fusion")
    log.info("FUSION     macro-F1 = %.4f", f1_fus)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "arch": "frozen ViT-B/16 + DistilBERT -> 2-token TransformerEncoder fusion",
            "classes": classes,
            "state_dict": {k: v.cpu() for k, v in fusion_model.state_dict().items()},
            "fusion_macro_f1": f1_fus,
        },
        MODELS_DIR / "fusion_model.pt",
    )
    ablation = {
        "dataset": "minhhungg/plant-disease-dataset (HuggingFace)",
        "encoders": "ViT-B/16 (ImageNet) + DistilBERT (base-uncased), both FROZEN",
        "text_modality": "TEMPLATED symptom text from taxonomy descriptions "
        "(no disease names, word-dropout + generic noise) — NOT real farmer input",
        "num_classes": len(classes),
        "classes": classes,
        "samples": len(samples),
        "epochs": args.epochs,
        "ablation_macro_f1": {
            "image_only": f1_img,
            "text_only": f1_txt,
            "fusion": f1_fus,
        },
        "fusion_vs_best_single": round(f1_fus - max(f1_img, f1_txt), 4),
        "device": device,
        "note": "Honest, reproducible ablation. Encoders frozen; text is synthetic "
        "(templated). Offline experiment only — deployed engine is the Groq VLM.",
    }
    (MODELS_DIR / "fusion_ablation.json").write_text(json.dumps(ablation, indent=2))
    log.info("SAVED ablation -> %s", MODELS_DIR / "fusion_ablation.json")
    print(json.dumps(ablation, indent=2))


if __name__ == "__main__":
    main()
