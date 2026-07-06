"""
Deterministic, offline symptom keyword extraction.

Wraps the existing rule-based NLU (src/preprocessing/text_nlu.py) to turn a
farmer's free text into a compact hint string that we inject into the vision
prompt. Fully optional: any failure degrades gracefully to no hints.
"""

import logging
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)


@lru_cache
def _nlu():
    try:
        from src.preprocessing.text_nlu import (
            TextNLU,  # namespace import from repo root
        )

        return TextNLU()
    except Exception:
        logger.info(
            "Text NLU unavailable; proceeding without symptom hints", exc_info=True
        )
        return None


def extract_hints(text: Optional[str]) -> Optional[str]:
    if not text or not text.strip():
        return None
    nlu = _nlu()
    if nlu is None:
        return None
    try:
        e = nlu.extract(text)
        bits: list[str] = []
        if e.symptoms:
            bits.append("symptoms=" + ",".join(e.symptoms))
        if e.affected_parts:
            bits.append("parts=" + ",".join(e.affected_parts))
        if e.weather_conditions:
            bits.append("weather=" + ",".join(e.weather_conditions))
        if e.spread_speed:
            bits.append(f"spread={e.spread_speed}")
        return "; ".join(bits) or None
    except Exception:
        logger.warning("NLU extraction failed", exc_info=True)
        return None
