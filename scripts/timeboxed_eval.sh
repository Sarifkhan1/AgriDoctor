#!/usr/bin/env bash
# Path B: time-boxed. Let training reach a usable checkpoint, then stop and
# evaluate the best-so-far. Training saves the best epoch continuously, so
# stopping early still yields the best checkpoint reached.
set -uo pipefail
cd "$(dirname "$0")/.."

TLOG=/tmp/train_6crop.log
CKPT=data/models/agridoctor_cnn_6crop.pt
DEADLINE_EPOCH="${1:-6}"     # stop once this epoch completes
MAX_WAIT_MIN="${2:-72}"      # ...or after this many minutes, whichever first

banner() { echo; echo "=========== $* ==========="; echo; }
start=$(date +%s)

banner "WAITING FOR EPOCH $DEADLINE_EPOCH OR ${MAX_WAIT_MIN}min"
while true; do
  done_ep=$(grep -oE "epoch +[0-9]+/" "$TLOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+" || echo 0)
  elapsed=$(( ($(date +%s) - start) / 60 ))
  if grep -q "DONE —" "$TLOG" 2>/dev/null; then echo "training finished on its own"; break; fi
  if [ "${done_ep:-0}" -ge "$DEADLINE_EPOCH" ]; then echo "reached epoch $done_ep"; break; fi
  if [ "$elapsed" -ge "$MAX_WAIT_MIN" ]; then echo "hit ${MAX_WAIT_MIN}min budget at epoch ${done_ep}"; break; fi
  sleep 30
done

banner "STOPPING TRAINING (best checkpoint is already saved)"
pkill -f "train_cnn.py.*dataset_v2" 2>/dev/null || true
sleep 5
grep -E "epoch +[0-9]+/|new best" "$TLOG" | tail -3

if [ ! -f "$CKPT" ]; then echo "NO CHECKPOINT — cannot evaluate"; exit 1; fi

banner "EVALUATING BEST 6-CROP CHECKPOINT (held-out test)"
.venv/bin/python scripts/evaluate_cnn.py \
  --model "$CKPT" --data data/dataset_v2 --out eval_report_6crop.json \
  --skip-robustness 2>&1

banner "SAFETY SUMMARY FOR PROMOTION DECISION"
.venv/bin/python - <<'PY'
import json
r=json.load(open("data/models/eval_report_6crop.json"))
sg=r["safety_gate"]; ood=r["ood_detection"]
print(f"accuracy         : {r['accuracy']}   macro-F1: {r['macro_f1']}")
print(f"false-accept rate: {sg['false_accept_rate']}   (4-crop was 0.0006)")
print(f"OOD AUROC        : {ood.get('auroc_oos_vs_in_scope')}   (4-crop was 1.0)")
print(f"calibration ECE  : {r['calibration']['ece']}")
print("\nper-crop recall (worst first):")
pc=r["per_class"]
for c in sorted(pc,key=lambda k:pc[k]['recall'])[:8]:
    print(f"  {c:22} recall={pc[c]['recall']:.3f} f1={pc[c]['f1']:.3f} n={pc[c]['support']}")
verdict = "PROMOTE" if (sg['false_accept_rate'] is not None and sg['false_accept_rate']<0.01
                        and (ood.get('auroc_oos_vs_in_scope') or 0)>0.97) else "REVIEW"
print(f"\nrecommendation: {verdict}")
PY

banner "DONE — review above; promote with: cp $CKPT data/models/agridoctor_cnn.pt"
