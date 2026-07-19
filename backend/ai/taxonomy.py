"""
Subject/condition taxonomy loader.

Single source of truth is data/taxonomy.json. Used both to constrain the model's
label output and to validate/enrich responses server-side. A "subject" is either
a supported crop (leaf disease) or a supported livestock animal (visible health
issue).
"""

import json
from functools import lru_cache
from pathlib import Path

SUPPORTED_CROPS = [
    "tomato",
    "potato",
    "rice",
    "maize",
    "chili",
    "cucumber",
    "pepper",
    "eggplant",
]
SUPPORTED_LIVESTOCK = ["cattle", "goat", "sheep", "poultry"]
# Everything the app can diagnose (crops + livestock).
SUPPORTED_SUBJECTS = SUPPORTED_CROPS + SUPPORTED_LIVESTOCK

_TAXONOMY_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "taxonomy.json"
)


def is_livestock(subject: str) -> bool:
    return (subject or "").strip().lower() in SUPPORTED_LIVESTOCK


@lru_cache
def load_taxonomy() -> dict:
    return json.loads(_TAXONOMY_PATH.read_text(encoding="utf-8"))


@lru_cache
def all_label_ids() -> frozenset[str]:
    ids = set()
    for entries in load_taxonomy().values():
        for e in entries:
            ids.add(e["label_id"])
    return frozenset(ids)


@lru_cache
def label_meta(label_id: str) -> dict | None:
    for crop, entries in load_taxonomy().items():
        for e in entries:
            if e["label_id"] == label_id:
                return {**e, "crop": crop}
    return None


def labels_for_prompt() -> str:
    """Render allowed label codes + descriptions for injection into the prompt.

    Crops and livestock are grouped under clear headings so the model knows which
    label set applies to a leaf photo versus an animal photo.
    """
    taxonomy = load_taxonomy()

    def _render(subject: str) -> list[str]:
        out = [f"\n{subject.upper()}:"]
        for e in taxonomy.get(subject, []):
            out.append(
                f"  - {e['label_id']}: {e['disease_name']} "
                f"({e['category']}) — {e['description']}"
            )
        return out

    lines: list[str] = ["=== CROPS (leaf/plant disease labels) ==="]
    for crop in SUPPORTED_CROPS:
        lines += _render(crop)
    lines.append("\n=== LIVESTOCK (animal health labels) ===")
    for animal in SUPPORTED_LIVESTOCK:
        lines += _render(animal)
    return "\n".join(lines)
