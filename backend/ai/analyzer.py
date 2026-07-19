"""
Analysis orchestrator: transcript -> prompt -> vision call -> validated result.

Hybrid pipeline:
1. If a trained local CNN model is available, try it first (fast, no API cost).
2. If CNN confidence >= threshold, use it as the primary diagnosis and call
   Groq only for treatment advice (cheap text call, no vision).
3. Otherwise, fall back to the full Groq vision pipeline.

Applies server-side invariants so a single model hallucination can never leak a
fabricated diagnosis for an unsupported crop or a non-leaf image.
"""

import base64
import json
import logging
from typing import Optional

from pydantic import ValidationError

from ..config import get_settings
from .cnn_predictor import CNNPredictor, get_predictor
from .prompts import build_system_prompt, build_user_context
from .provider import AIProvider, GroqProvider
from .schemas import AdviceBlock, AnalysisResult, ResultKind, Urgency
from .taxonomy import (
    SUPPORTED_CROPS,
    SUPPORTED_LIVESTOCK,
    SUPPORTED_SUBJECTS,
    all_label_ids,
    label_meta,
)

logger = logging.getLogger(__name__)


class Analyzer:
    def __init__(
        self,
        provider: Optional[AIProvider] = None,
        cnn_predictor: Optional[CNNPredictor] = None,
    ) -> None:
        self._provider = provider  # if None, created lazily (keeps imports cheap)
        self._cnn: Optional[CNNPredictor] = cnn_predictor

    @property
    def provider(self) -> AIProvider:
        if self._provider is None:
            self._provider = GroqProvider()
        return self._provider

    @property
    def cnn(self) -> Optional[CNNPredictor]:
        """Return the CNN predictor (created lazily from config)."""
        if self._cnn is None:
            settings = get_settings()
            if settings.use_local_cnn:
                self._cnn = get_predictor(settings.cnn_model_path)
        return self._cnn

    # ------------------------------------------------------------------ #
    def analyze(
        self,
        *,
        image_bytes: bytes,
        image_mime: str,
        audio: Optional[dict] = None,
        crop_hint: Optional[str] = None,
        notes: Optional[str] = None,
        onset_days: Optional[int] = None,
        spread: Optional[str] = None,
        nlu_hints: Optional[str] = None,
    ) -> AnalysisResult:
        settings = get_settings()

        # --- Audio transcription (always, if present) ---
        transcript = ""
        if audio:
            try:
                transcript = self.provider.transcribe(
                    audio["bytes"], audio.get("filename", "")
                )
            except Exception:
                logger.warning(
                    "Audio transcription failed; continuing image-only", exc_info=True
                )

        # --- Try local CNN first ---
        cnn_result = self._try_cnn(image_bytes, settings)

        if cnn_result is not None:
            logger.info(
                "CNN prediction: %s (conf=%.2f)",
                cnn_result["primary_label"],
                cnn_result["confidence"],
            )

            if cnn_result["confidence"] >= settings.cnn_confidence_threshold:
                # CNN is confident — build result from CNN, optionally get advice from Groq
                result = self._cnn_to_result(cnn_result, settings)

                # Try to get treatment advice from Groq (text-only, cheap)
                try:
                    advice = self._get_advice_from_groq(
                        result, crop_hint, transcript, notes, onset_days, spread, nlu_hints
                    )
                    if advice:
                        result.advice = advice
                except Exception:
                    logger.warning("Groq advice call failed (non-fatal)", exc_info=True)

                return self._enforce_invariants(result, crop_hint, bool(transcript))
            else:
                logger.info(
                    "CNN confidence %.2f < threshold %.2f — falling back to Groq",
                    cnn_result["confidence"],
                    settings.cnn_confidence_threshold,
                )

        # --- Groq vision pipeline (full) ---
        # NOTE: we deliberately do NOT pass crop_hint into the vision prompt.
        # The crop button is an unreliable guess and, when injected as text, the
        # model tends to defer to it and mislabel other species (e.g. a grape leaf
        # called "tomato"). Species identification must be purely visual. The hint
        # is still used afterwards in _enforce_invariants for the "you picked X but
        # this looks like Y" flag.
        context = build_user_context(
            crop_hint=None,
            transcript=transcript,
            notes=notes,
            onset_days=onset_days,
            spread=spread,
            nlu_hints=nlu_hints,
        )
        system_prompt = build_system_prompt()
        data_url = self._data_url(image_bytes, image_mime)

        raw = self.provider.analyze_image(data_url, system_prompt, context)
        result = self._parse(raw, system_prompt, context, data_url)

        result.provider = "groq"
        result.model_id = settings.groq_vision_model
        return self._enforce_invariants(result, crop_hint, bool(transcript))

    # ------------------------------------------------------------------ #
    # CNN helpers
    # ------------------------------------------------------------------ #
    def _try_cnn(self, image_bytes: bytes, settings) -> Optional[dict]:
        """Attempt CNN prediction. Returns None on any failure."""
        if not settings.use_local_cnn:
            return None
        predictor = self.cnn
        if predictor is None or not predictor.available:
            return None
        try:
            return predictor.predict(image_bytes)
        except Exception:
            logger.warning("CNN inference failed", exc_info=True)
            return None

    @staticmethod
    def _cnn_to_result(cnn: dict, settings) -> AnalysisResult:
        """Convert CNN prediction dict to an AnalysisResult."""
        label = cnn["primary_label"]
        meta = label_meta(label)

        # Determine the crop from the label prefix
        crop_name = None
        if meta:
            crop_name = meta.get("crop")

        is_healthy = label.endswith("_HEALTHY")

        return AnalysisResult(
            kind=ResultKind.HEALTHY if is_healthy else ResultKind.DIAGNOSIS,
            is_plant=True,
            is_leaf=True,
            detected_crop=crop_name,
            crop_supported=crop_name in SUPPORTED_SUBJECTS if crop_name else False,
            primary_label=label,
            disease_name=meta["disease_name"] if meta else None,
            category=meta["category"] if meta else None,
            secondary_labels=[
                p["label"]
                for p in cnn.get("top_predictions", [])[1:3]
                if p["label"] in all_label_ids()
            ],
            confidence=cnn["confidence"],
            severity_score=cnn.get("severity", 0.5),
            urgency_level=(
                Urgency.low
                if is_healthy
                else (Urgency.high if cnn.get("severity", 0.5) > 0.7 else Urgency.medium)
            ),
            provider="local_cnn",
            model_id="CropDiseaseViT",
        )

    def _get_advice_from_groq(
        self,
        result: AnalysisResult,
        crop_hint: Optional[str],
        transcript: str,
        notes: Optional[str],
        onset_days: Optional[int],
        spread: Optional[str],
        nlu_hints: Optional[str],
    ) -> Optional[AdviceBlock]:
        """Ask Groq (text-only, no vision) for treatment advice."""
        if result.kind == ResultKind.HEALTHY:
            return AdviceBlock(
                summary="Your plant looks healthy! No disease symptoms detected.",
                what_to_do_now=["Continue regular care and monitoring."],
                prevention=["Maintain good watering practices.", "Ensure proper spacing."],
            )

        settings = get_settings()
        prompt = (
            f"You are an agricultural disease advisor. A CNN model identified "
            f"{result.disease_name} ({result.primary_label}) on {result.detected_crop} "
            f"with {result.confidence:.0%} confidence and severity {result.severity_score:.0%}.\n"
        )
        if transcript:
            prompt += f"Farmer's voice description: {transcript}\n"
        if notes:
            prompt += f"Notes: {notes}\n"
        if onset_days is not None:
            prompt += f"Onset: {onset_days} days ago\n"
        if spread:
            prompt += f"Spread: {spread}\n"
        prompt += (
            "\nRespond with a JSON object with these fields: "
            '"summary" (string), "what_to_do_now" (array of strings), '
            '"prevention" (array of strings), "when_to_get_help" (array of strings).'
        )

        try:
            from groq import Groq

            client = Groq(
                api_key=settings.groq_api_key,
                timeout=float(settings.ai_timeout_seconds),
            )
            resp = client.chat.completions.create(
                model=settings.groq_text_model,
                temperature=0.3,
                max_tokens=800,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.choices[0].message.content or ""
            data = json.loads(raw)
            return AdviceBlock(**data)
        except Exception:
            logger.warning("Groq text advice call failed", exc_info=True)
            return None

    # ------------------------------------------------------------------ #
    @staticmethod
    def _data_url(image_bytes: bytes, mime: str) -> str:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        return f"data:{mime};base64,{b64}"

    def _parse(
        self, raw: str, system_prompt: str, context: str, data_url: str
    ) -> AnalysisResult:
        parsed = self._try_parse(raw)
        if parsed is not None:
            return parsed

        # One repair attempt with a stricter reminder.
        try:
            fixed = self.provider.analyze_image(
                data_url,
                system_prompt + "\nREMINDER: return ONLY one valid JSON object.",
                context,
            )
            parsed = self._try_parse(fixed)
            if parsed is not None:
                return parsed
        except Exception:
            logger.warning("Repair attempt failed", exc_info=True)

        logger.error("Model returned unparseable JSON: %s", (raw or "")[:500])
        return AnalysisResult(
            kind=ResultKind.LOW_CONFIDENCE,
            message=(
                "We couldn't analyze this image reliably. Please retake a clear, "
                "close photo of a single affected leaf in good light."
            ),
        )

    @staticmethod
    def _try_parse(raw: str) -> Optional[AnalysisResult]:
        if not raw:
            return None
        text = raw.strip()
        # Strip reasoning-model <think>...</think> blocks if any leaked through.
        if "<think>" in text:
            import re as _re

            text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL).strip()
        # Strip accidental code fences.
        if text.startswith("```"):
            text = text.strip("`")
            if text.lstrip().lower().startswith("json"):
                text = text.lstrip()[4:]
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            try:
                data = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
        try:
            return AnalysisResult(**data)
        except ValidationError:
            logger.warning("JSON did not match AnalysisResult schema", exc_info=True)
            return None

    def _enforce_invariants(
        self, r: AnalysisResult, crop_hint: Optional[str], had_audio: bool
    ) -> AnalysisResult:
        # Clamp numerics.
        r.confidence = min(max(r.confidence, 0.0), 1.0)
        r.severity_score = min(max(r.severity_score, 0.0), 1.0)

        rejection = {
            ResultKind.UNSUPPORTED_CROP,
            ResultKind.NOT_A_LEAF,
            ResultKind.LOW_CONFIDENCE,
        }
        kind = ResultKind(r.kind)

        if kind in rejection:
            # Strip any disease content the model may have leaked.
            r.primary_label = None
            r.disease_name = None
            r.category = None
            r.secondary_labels = []
            r.advice = None
            r.crop_supported = False
        else:
            # Diagnosis / healthy: must be a supported subject (crop or livestock)
            # with a valid label that BELONGS to that subject.
            detected = (r.detected_crop or "").strip().lower()
            r.crop_supported = detected in SUPPORTED_SUBJECTS

            if not r.crop_supported:
                # Model claimed a diagnosis for an unsupported subject -> reject.
                r.kind = ResultKind.UNSUPPORTED_CROP
                r.primary_label = None
                r.disease_name = None
                r.category = None
                r.secondary_labels = []
                r.advice = None
                if not r.message:
                    name = r.detected_crop or "this subject"
                    r.message = (
                        f"This looks like {name}, which AgriDoctor doesn't support yet. "
                        f"Supported crops: {', '.join(SUPPORTED_CROPS)}. "
                        f"Supported livestock: {', '.join(SUPPORTED_LIVESTOCK)}."
                    )
            else:
                meta = label_meta(r.primary_label) if r.primary_label else None
                # Label must exist AND belong to the detected subject (no cross-subject
                # leakage, e.g. a TOM_* label on a cattle photo).
                label_ok = meta is not None and meta.get("crop") == detected
                if not label_ok:
                    r.kind = ResultKind.LOW_CONFIDENCE
                    r.primary_label = None
                    r.disease_name = None
                    r.category = None
                    r.secondary_labels = []
                    r.advice = None
                    if not r.message:
                        subj = "animal" if detected in SUPPORTED_LIVESTOCK else "crop"
                        r.message = (
                            f"We recognized the {subj} but couldn't confidently identify "
                            f"the condition. Try a sharper close-up of the affected area."
                        )
                else:
                    r.disease_name = r.disease_name or meta["disease_name"]
                    r.category = r.category or meta["category"]
                    # Keep only valid secondary labels that belong to this subject.
                    r.secondary_labels = [
                        s
                        for s in r.secondary_labels
                        if (m := label_meta(s)) and m.get("crop") == detected
                    ][:2]
                    # A *_HEALTHY label means healthy, not a disease.
                    if r.primary_label.endswith("_HEALTHY"):
                        r.kind = ResultKind.HEALTHY

        # Normalize urgency enum.
        try:
            r.urgency_level = Urgency(r.urgency_level)
        except ValueError:
            r.urgency_level = Urgency.medium

        # Hint match flag for the UI ("you picked tomato, this looks like potato").
        if crop_hint and r.detected_crop:
            r.matches_hint = (
                crop_hint.strip().lower() == r.detected_crop.strip().lower()
            )

        # Note when a voice note couldn't be used.
        if (
            not had_audio
            and r.message is None
            and ResultKind(r.kind) == ResultKind.DIAGNOSIS
        ):
            pass

        return r
