#!/usr/bin/env bash
# AgriDoctor — 6-crop fast pipeline: wait for build -> prune -> train -> evaluate.
#
# The six project crops: tomato, potato, rice, maize, chili, cucumber.
# Pepper and eggplant are pruned from the built dataset (fast: just delete their
# class dirs) so we don't rebuild. The two OOS reject classes are kept.
#
# Speed choices for an 8GB / MPS MacBook (hardware is the ceiling here, not
# config): memory-safe batch 16 + 2 workers, and 12 epochs with early stopping —
# the model reached 0.97 macro-F1 by epoch 3 last run, so 16 was wasteful.
# Trains to a SEPARATE file; the live 4-crop model is never touched.
set -uo pipefail
cd "$(dirname "$0")/.."

DATA=data/dataset_v2
OUT=agridoctor_cnn_6crop.pt
CKPT=data/models/$OUT
TLOG=/tmp/train_6crop.log

banner() { echo; echo "=========== $* ==========="; echo; }

# --- 1. Wait for the dataset build ----------------------------------------
banner "WAITING FOR DATASET BUILD"
while [ ! -f "$DATA/manifest.json" ]; do sleep 30; done
sleep 5

# --- 2. Prune pepper + eggplant (keep the 6 crops + 2 OOS) -----------------
banner "PRUNING PEPPER + EGGPLANT"
for split in train val test; do
  for cls in "$DATA/$split"/PEP_* "$DATA/$split"/EGG_*; do
    [ -d "$cls" ] && rm -rf "$cls" && echo "  removed $cls"
  done
done
echo "remaining classes:"
ls "$DATA/train" | sed 's/^/  /'

# --- 3. Train (fast, 6 crops) ---------------------------------------------
banner "TRAINING (6 crops, 12 epochs, memory-safe)"
.venv/bin/python -u scripts/train_cnn.py \
  --data "$DATA" --arch efficientnet_b0 \
  --epochs 12 --warmup-epochs 2 --batch-size 16 --workers 2 --patience 4 \
  --mixup 0.2 --cutmix 1.0 --ema-decay 0.999 \
  --out "$OUT" 2>&1 | tee "$TLOG"

if ! grep -q "DONE —" "$TLOG"; then
  echo "TRAINING DID NOT COMPLETE — aborting before evaluation"
  exit 1
fi

# --- 4. Evaluate ----------------------------------------------------------
banner "HELD-OUT EVALUATION (6 crops)"
.venv/bin/python scripts/evaluate_cnn.py \
  --model "$CKPT" --data "$DATA" --out eval_report_6crop.json 2>&1

banner "PIPELINE COMPLETE"
echo "Live 4-crop model untouched at data/models/agridoctor_cnn.pt"
echo "Review data/models/eval_report_6crop.json — watch false_accept_rate,"
echo "ood AUROC, and per-class recall for RICE_* / CHILI_* / CUC_* before promoting."
