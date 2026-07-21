#!/usr/bin/env bash
# AgriDoctor — unattended train -> evaluate -> refine -> evaluate pipeline.
#
# Stage 1 (already running when this is launched) trains at 224px from ImageNet.
# This script waits for it, then:
#   2. evaluates it on the held-out test split
#   3. progressively resizes: fine-tunes those weights at 288px for a few epochs
#   4. evaluates the refined model
#   5. prints a side-by-side comparison
#
# The refined model is written to a SEPARATE file and never auto-promoted: a
# larger input size usually helps but is not guaranteed to, and silently
# overwriting a good checkpoint with a worse one is not a trade worth making.
# Promotion is a deliberate step once the comparison is on screen.
set -uo pipefail
cd "$(dirname "$0")/.."

TRAIN_LOG="${1:?usage: run_pipeline.sh <path-to-stage1-training-log>}"
STAGE1=data/models/agridoctor_cnn.pt
STAGE2=data/models/agridoctor_cnn_288.pt

banner() { echo; echo "=========== $* ==========="; echo; }

# --- 1. Wait for stage-1 training -----------------------------------------
# Watch the log, not the process list: a pgrep pattern would match this
# script's own command line and return immediately.
banner "WAITING FOR STAGE-1 TRAINING"
while ! grep -qE "DONE —|Traceback" "$TRAIN_LOG" 2>/dev/null; do sleep 60; done

if grep -q "Traceback" "$TRAIN_LOG"; then
  echo "STAGE-1 TRAINING FAILED — aborting pipeline"
  tail -30 "$TRAIN_LOG"
  exit 1
fi
sleep 20   # let the checkpoint + plots finish flushing

if [ ! -f "$STAGE1" ]; then
  echo "no checkpoint at $STAGE1 — aborting"
  exit 1
fi

banner "STAGE-1 TRAINING COMPLETE"
grep -E "epoch |temperature|DONE" "$TRAIN_LOG" | tail -20

# --- 2. Evaluate stage 1 ---------------------------------------------------
banner "STAGE-1 EVALUATION (224px, held-out test split)"
.venv/bin/python scripts/evaluate_cnn.py \
  --model "$STAGE1" --out eval_report.json 2>&1

# --- 3. Progressive resize: fine-tune at 288px -----------------------------
# Lower LR and no head-only warmup: the head is already trained, so freezing the
# backbone to protect it from a random head would achieve nothing here.
banner "STAGE-2 TRAINING (progressive resize -> 288px)"
.venv/bin/python -u scripts/train_cnn.py \
  --data data/dataset --arch efficientnet_b0 \
  --init-from "$STAGE1" \
  --size 288 --epochs 5 --warmup-epochs 0 --patience 3 \
  --lr 6e-5 --batch-size 24 --workers 6 \
  --mixup 0.2 --cutmix 1.0 --ema-decay 0.999 \
  --out "$(basename "$STAGE2")" 2>&1

# --- 4. Evaluate stage 2 ---------------------------------------------------
if [ -f "$STAGE2" ]; then
  banner "STAGE-2 EVALUATION (288px, same held-out split)"
  .venv/bin/python scripts/evaluate_cnn.py \
    --model "$STAGE2" --out eval_report_288.json 2>&1
fi

# --- 5. Compare ------------------------------------------------------------
banner "COMPARISON"
.venv/bin/python - <<'PY'
import json
from pathlib import Path

M = Path("data/models")
rows = []
for label, name in (("224px (stage 1)", "eval_report.json"),
                    ("288px (stage 2)", "eval_report_288.json")):
    p = M / name
    if not p.exists():
        continue
    r = json.loads(p.read_text())
    rows.append((label, r))

if not rows:
    print("no evaluation reports found")
else:
    hdr = f"{'model':<18}{'acc':>8}{'macroF1':>9}{'ECE':>8}{'AUROC':>8}{'falseAcc':>10}{'meanCorrupt':>13}"
    print(hdr)
    print("-" * len(hdr))
    for label, r in rows:
        rob = r.get("robustness", {}).get("_summary", {})
        print(
            f"{label:<18}"
            f"{r['accuracy']:>8.4f}"
            f"{r['macro_f1']:>9.4f}"
            f"{r['calibration']['ece']:>8.4f}"
            f"{r.get('ood_detection', {}).get('auroc_oos_vs_in_scope', float('nan')):>8.4f}"
            f"{r['safety_gate'].get('false_accept_rate', float('nan')):>10.4f}"
            f"{rob.get('mean_corrupted_accuracy', float('nan')):>13.4f}"
        )
    if len(rows) == 2:
        d = rows[1][1]["macro_f1"] - rows[0][1]["macro_f1"]
        verdict = "PROMOTE 288px" if d > 0 else "KEEP 224px"
        print(f"\nmacro-F1 delta: {d:+.4f}  ->  {verdict}")
        print("(nothing has been promoted automatically — this is a recommendation)")
PY

banner "PIPELINE COMPLETE"
