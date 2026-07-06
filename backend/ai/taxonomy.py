"""
Crop/disease taxonomy loader.

Single source of truth is data/taxonomy.json (generated from
docs/CROP_DISEASE_TAXONOMY.md). Used both to constrain the model's label output
and to validate/enrich responses server-side.
"""

import json
from functools import lru_cache
from pathlib import Path

SUPPORTED_CROPS = ["tomato", "potato", "rice", "maize", "chili", "cucumber"]

_TAXONOMY_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "taxonomy.json"
)


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
    """Render allowed label codes + descriptions for injection into the prompt."""
    lines: list[str] = []
    for crop, entries in load_taxonomy().items():
        lines.append(f"\n{crop.upper()}:")
        for e in entries:
            lines.append(
                f"  - {e['label_id']}: {e['disease_name']} "
                f"({e['category']}) — {e['description']}"
            )
    return "\n".join(lines)
