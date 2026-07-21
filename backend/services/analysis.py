"""
Analysis orchestrator: validate -> transcribe/NLU -> vision -> persist.

The Groq calls are synchronous/network-bound, so they run in a threadpool to
avoid blocking the event loop. Persistence is best-effort and never turns a good
diagnosis into an error for the user.
"""

import json
import logging
import uuid
from pathlib import Path
from typing import Optional

from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool

from ..ai.analyzer import Analyzer
from ..ai.schemas import AnalysisResult
from ..db import UPLOADS_DIR, get_db
from .nlu_hints import extract_hints
from .validation import validate_audio, validate_image

logger = logging.getLogger(__name__)


class AnalysisService:
    def __init__(self, analyzer: Optional[Analyzer] = None) -> None:
        self._analyzer = analyzer or Analyzer()

    async def run(
        self,
        *,
        image_bytes: bytes,
        image_filename: Optional[str],
        image_content_type: Optional[str],
        audio: Optional[dict],
        crop_hint: Optional[str],
        onset_days: Optional[int],
        spread: Optional[str],
        notes: Optional[str],
        user: Optional[dict],
        language: Optional[str] = None,
    ) -> tuple[AnalysisResult, Optional[str]]:
        mime, ext = validate_image(
            image_bytes, image_filename or "", image_content_type
        )
        if audio:
            validate_audio(audio)

        nlu_hints = extract_hints(" ".join(filter(None, [notes])))

        try:
            result: AnalysisResult = await run_in_threadpool(
                self._analyzer.analyze,
                image_bytes=image_bytes,
                image_mime=mime,
                audio=audio,
                crop_hint=crop_hint,
                notes=notes,
                onset_days=onset_days,
                spread=spread,
                nlu_hints=nlu_hints,
                language=language,
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("AI analysis failed")
            msg = str(exc).lower()
            if "rate_limit" in msg or "rate limit" in msg or "429" in msg:
                raise HTTPException(
                    429,
                    "The AI service is busy right now (free-tier limit reached). "
                    "Please wait a few seconds and try again.",
                )
            raise HTTPException(
                503,
                "The analysis service is temporarily unavailable. Please try again.",
            )

        logger.info(
            "analysis kind=%s crop=%s conf=%.2f",
            result.kind,
            result.detected_crop,
            result.confidence,
        )

        case_id: Optional[str] = None
        if user:
            try:
                case_id = await run_in_threadpool(
                    self._persist,
                    user=user,
                    image_bytes=image_bytes,
                    ext=ext,
                    mime=mime,
                    crop_hint=crop_hint,
                    onset_days=onset_days,
                    spread=spread,
                    notes=notes,
                    result=result,
                )
            except Exception:
                logger.warning("Persisting analysis failed (non-fatal)", exc_info=True)

        return result, case_id

    # ------------------------------------------------------------------ #
    def _persist(
        self,
        *,
        user: dict,
        image_bytes: bytes,
        ext: str,
        mime: str,
        crop_hint: Optional[str],
        onset_days: Optional[int],
        spread: Optional[str],
        notes: Optional[str],
        result: AnalysisResult,
    ) -> str:
        case_id = str(uuid.uuid4())
        media_id = str(uuid.uuid4())
        pred_id = str(uuid.uuid4())

        image_dir = UPLOADS_DIR / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        file_path = image_dir / f"{media_id}{ext}"
        file_path.write_bytes(image_bytes)

        crop_name = result.detected_crop or crop_hint
        advice = result.advice.model_dump() if result.advice else None

        # The Grad-CAM overlay is ~200KB of base64 per diagnosis and is derived
        # data — it can be recomputed at any time from the image we store below.
        # Persisting it would put a quarter-megabyte blob in every row of
        # `predictions`, and `GET /api/cases/{id}` does `SELECT *`, so it would be
        # read back and shipped to the client on every history view too. The
        # heatmap stays in the live API response; only the stored copy drops it.
        stored_result = result.model_dump(
            exclude={"heatmap_png_b64"}, mode="json"
        )

        with get_db() as conn:
            conn.execute(
                """INSERT INTO cases (id, user_id, category, crop_name, status)
                   VALUES (?, ?, 'crop', ?, 'done')""",
                (case_id, user["id"], crop_name),
            )
            conn.execute(
                """INSERT INTO media (id, case_id, type, uri, content_type)
                   VALUES (?, ?, 'image', ?, ?)""",
                (media_id, case_id, str(file_path), mime),
            )
            conn.execute(
                """INSERT OR REPLACE INTO metadata
                   (case_id, onset_days, spread_speed, notes_text, language)
                   VALUES (?, ?, ?, ?, 'en')""",
                (case_id, onset_days, spread, notes),
            )
            conn.execute(
                """INSERT INTO predictions
                   (id, case_id, model_version, provider, model_id, kind, detected_crop,
                    crop_supported, is_leaf, primary_label, disease_name,
                    secondary_labels_json, severity_score, urgency_level, confidence,
                    visual_evidence, advice_json, raw_response)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    pred_id,
                    case_id,
                    "2.0.0",
                    result.provider,
                    result.model_id,
                    str(result.kind),
                    result.detected_crop,
                    int(result.crop_supported),
                    int(result.is_leaf),
                    result.primary_label,
                    result.disease_name,
                    json.dumps(result.secondary_labels),
                    result.severity_score,
                    str(result.urgency_level),
                    result.confidence,
                    result.visual_evidence,
                    json.dumps(advice) if advice else None,
                    json.dumps(stored_result, default=str),
                ),
            )
        return case_id
