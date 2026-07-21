"""
AgriDoctor — reproduce the report's validation scenarios against the real system.

Section 5 of the submission report documents four end-to-end test cases. This
script re-runs them through the actual serving pipeline (`backend.ai.analyzer`)
using held-out test images, so the reported behaviour is measured rather than
described from memory.

The four scenarios exercise each distinct routing outcome:

  1. Diseased supported crop   -> local diagnosis, no network call
  2. Healthy supported crop    -> local "healthy" result, no network call
  3. Unsupported plant species -> escalation (or an honest low-confidence result
                                  when no hosted model is configured)
  4. Non-plant photograph      -> local rejection, no network call

Images come from the held-out test split, so none were seen during training or
model selection.

Usage:
    .venv/bin/python scripts/demo_scenarios.py
    .venv/bin/python scripts/demo_scenarios.py --json > scenarios.json
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# The analyzer reads settings at import time; make sure a local-CNN-first
# configuration is active regardless of what happens to be in .env.
os.environ.setdefault("SECRET_KEY", "demo-scenarios-not-a-real-secret")
os.environ.setdefault("USE_LOCAL_CNN", "true")

SCENARIOS = [
    {
        "n": 1,
        "title": "Target crop leaf pathology (tomato late blight)",
        "class_dir": "TOM_LATE_BLIGHT",
        "crop_hint": "tomato",
        "notes": "spots on lower tomato leaves spreading upwards after heavy rain",
        "onset_days": 4,
        "spread": "fast",
        "expect": "diagnosis, answered locally",
    },
    {
        "n": 2,
        "title": "Target crop healthy state (healthy potato leaf)",
        "class_dir": "POT_HEALTHY",
        "crop_hint": "potato",
        "notes": None,
        "onset_days": None,
        "spread": None,
        "expect": "healthy, answered locally",
    },
    {
        "n": 3,
        "title": "Out-of-scope leaf rejection (unsupported plant species)",
        "class_dir": "OOS_OTHER_PLANT",
        "crop_hint": "tomato",
        "notes": None,
        "onset_days": None,
        "spread": None,
        "expect": "escalate to hosted model (or honest low-confidence without a key)",
    },
    {
        "n": 4,
        "title": "Non-plant visual noise rejection (everyday object)",
        "class_dir": "OOS_NOT_PLANT",
        "crop_hint": "cucumber",
        "notes": None,
        "onset_days": None,
        "spread": None,
        "expect": "rejected locally, no API call",
    },
]


def pick_image(split_dir: Path, class_dir: str, rng) -> Path | None:
    d = split_dir / class_dir
    if not d.is_dir():
        return None
    files = sorted(d.glob("*.jpg"))
    return rng.choice(files) if files else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset")
    ap.add_argument("--split", default="test", help="held-out split to sample from")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json", action="store_true", help="emit JSON instead of prose")
    args = ap.parse_args()

    from backend.ai.analyzer import Analyzer
    from backend.ai.cnn_predictor import get_predictor
    from backend.config import get_settings

    settings = get_settings()
    split_dir = PROJECT_ROOT / args.data / args.split
    if not split_dir.is_dir():
        raise SystemExit(f"no {args.split} split at {split_dir}")

    predictor = get_predictor(settings.cnn_model_path)
    if not predictor.available:
        raise SystemExit(
            f"no trained checkpoint at {settings.cnn_model_path} — train first"
        )

    rng = random.Random(args.seed)
    analyzer = Analyzer()
    has_key = bool(settings.groq_api_key)

    results = []
    for spec in SCENARIOS:
        img_path = pick_image(split_dir, spec["class_dir"], rng)
        if img_path is None:
            print(f"!! no images for {spec['class_dir']}, skipping", file=sys.stderr)
            continue

        image_bytes = img_path.read_bytes()
        # Raw CNN output first: shows the routing decision behind the final result.
        cnn = predictor.predict(image_bytes)

        started = time.perf_counter()
        result = analyzer.analyze(
            image_bytes=image_bytes,
            image_mime="image/jpeg",
            crop_hint=spec["crop_hint"],
            notes=spec["notes"],
            onset_days=spec["onset_days"],
            spread=spec["spread"],
        )
        elapsed_ms = (time.perf_counter() - started) * 1000

        row = {
            "scenario": spec["n"],
            "title": spec["title"],
            "image": str(img_path.relative_to(PROJECT_ROOT)),
            "true_class": spec["class_dir"],
            "user_selected_crop": spec["crop_hint"],
            "expected": spec["expect"],
            "cnn_decision": cnn["decision"] if cnn else None,
            "cnn_label": cnn["primary_label"] if cnn else None,
            "cnn_confidence": cnn["confidence"] if cnn else None,
            "cnn_margin": cnn["margin"] if cnn else None,
            "result_kind": str(result.kind),
            "detected_crop": result.detected_crop,
            "primary_label": result.primary_label,
            "disease_name": result.disease_name,
            "confidence": round(result.confidence, 4),
            "severity": round(result.severity_score, 4),
            "urgency": str(result.urgency_level),
            "provider": result.provider,
            "matches_hint": result.matches_hint,
            "has_heatmap": result.heatmap_png_b64 is not None,
            "heatmap_focus": result.heatmap_focus,
            "message": result.message,
            "advice_actions": (result.advice.what_to_do_now if result.advice else []),
            "elapsed_ms": round(elapsed_ms, 1),
            # provider == "local_cnn" is the proof no API call was made.
            "used_network": result.provider not in ("local_cnn",),
        }
        results.append(row)

    if args.json:
        print(json.dumps(
            {"hosted_model_configured": has_key, "scenarios": results}, indent=2
        ))
        return

    print("=" * 78)
    print("AGRIDOCTOR — VALIDATION SCENARIOS (held-out test images)")
    print(f"hosted model configured: {has_key}")
    print("=" * 78)
    for r in results:
        print(f"\n--- Scenario {r['scenario']}: {r['title']} ---")
        print(f"  image (unseen)     : {r['image']}")
        print(f"  true class         : {r['true_class']}")
        print(f"  user selected crop : {r['user_selected_crop']}")
        print(f"  CNN decision       : {r['cnn_decision']}  "
              f"({r['cnn_label']}, conf={r['cnn_confidence']}, margin={r['cnn_margin']})")
        print(f"  result kind        : {r['result_kind']}")
        if r["primary_label"]:
            print(f"  diagnosis          : {r['disease_name']} ({r['primary_label']})")
            print(f"  detected crop      : {r['detected_crop']}")
            print(f"  confidence/severity: {r['confidence']:.0%} / {r['severity']:.0%}"
                  f"   urgency={r['urgency']}")
        if r["message"]:
            print(f"  message            : {r['message']}")
        print(f"  engine             : {r['provider']}"
              f"   network used: {r['used_network']}")
        print(f"  Grad-CAM           : {r['has_heatmap']}"
              + (f" (focus={r['heatmap_focus']})" if r["has_heatmap"] else ""))
        print(f"  latency            : {r['elapsed_ms']} ms")
        if r["advice_actions"]:
            print("  immediate actions  :")
            for a in r["advice_actions"][:3]:
                print(f"      - {a}")

    offline = sum(1 for r in results if not r["used_network"])
    print(f"\n{'=' * 78}")
    print(f"{offline}/{len(results)} scenarios resolved with no network call.")
    print("=" * 78)


if __name__ == "__main__":
    main()
