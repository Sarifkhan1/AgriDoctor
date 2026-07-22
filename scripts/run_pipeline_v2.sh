#!/usr/bin/env bash
# AgriDoctor — 8-crop pipeline: wait for build -> train -> evaluate.
#
# Memory-safe config for an 8GB machine: batch 16, 2 workers. The class-balanced
# sampler handles the heavy imbalance (chili ~430 images vs PlantVillage ~1800).
# Trains to data/models/agridoctor_cnn_v2.pt so the live 4-crop model is never
# touched until the safety metrics are checked and we deliberately promote.
set -uo pipefail
cd "$(dirname "$0")/.."

DATA=data/dataset_v2
OUT=agridoctor_cnn_v2.pt
CKPT=data/models/$OUT
TLOG=/tmp/train_v2.log

banner() { echo; echo "=========== $* ==========="; echo; }

# --- 1. Wait for the dataset build ----------------------------------------
banner "WAITING FOR 8-CROP DATASET BUILD"
while [ ! -f "$DATA/manifest.json" ]; do sleep 30; done
sleep 5
banner "DATASET READY"
.venv/bin/python - <<'PY'
import json
from collections import defaultdict
m=json.load(open("data/dataset_v2/manifest.json"))
d=defaultdict(int)
for k,v in m["counts"].items():
    d["OOS" if k.split("/")[1].startswith("OOS") else k.split("/")[1].split("_")[0]]+=v
print("total",m["total_images"],"classes",m["num_classes"])
for p in ["TOM","POT","PEP","MAIZE","RICE","CHILI","CUC","EGG","OOS"]:
    print(f"  {p:6} {d[p]:6d}")
PY

# --- 2. Train -------------------------------------------------------------
banner "TRAINING (8 crops, memory-safe config)"
.venv/bin/python -u scripts/train_cnn.py \
  --data "$DATA" --arch efficientnet_b0 \
  --epochs 16 --warmup-epochs 2 --batch-size 16 --workers 2 --patience 5 \
  --mixup 0.2 --cutmix 1.0 --ema-decay 0.999 \
  --out "$OUT" 2>&1 | tee "$TLOG"

if ! grep -q "DONE —" "$TLOG"; then
  echo "TRAINING DID NOT COMPLETE — aborting before evaluation"
  exit 1
fi

# --- 3. Evaluate ----------------------------------------------------------
banner "HELD-OUT EVALUATION (8 crops)"
.venv/bin/python scripts/evaluate_cnn.py \
  --model "$CKPT" --data "$DATA" --out eval_report_v2.json 2>&1

banner "PIPELINE COMPLETE — review eval_report_v2.json before promoting"
echo "The live 4-crop model is untouched at data/models/agridoctor_cnn.pt"
echo "Promote with:  cp $CKPT data/models/agridoctor_cnn.pt   (only if metrics hold)"
