"""
Analysis orchestrator: image -> local CNN -> (escalate to hosted VLM) -> result.

The local CNN is the primary engine. It was trained with explicit out-of-scope
classes, so it produces a routing decision rather than a forced label:

  diagnosis  confident, in-scope crop disease
               -> answer entirely locally: curated advice, Grad-CAM, no network
  not_plant  confidently not vegetation
               -> reject locally, no network
  escalate   another plant species, or low confidence
               -> hand the image to the hosted vision model

Escalation is a designed tier, not a failure path. AgriDoctor's taxonomy covers
eight crops and four livestock species; the public PlantVillage corpus only has
images for tomato, potato, pepper and maize. Rice, chili, cucumber, eggplant and
all livestock therefore *cannot* be learned locally and are meant to escalate.

The practical effect is that the common case — a tomato or potato leaf — costs
no API call at all, which is what removes the free-tier rate limit from normal
use. When no Groq key is configured the app still works; escalated cases return
an honest low-confidence result instead.

Every path, local or hosted, exits through _enforce_invariants so a model
mistake cannot leak a fabricated diagnosis for an unsupported subject.
"""

import base64
import json
import logging
from typing import Optional

from pydantic import ValidationError

from ..config import get_settings
from . import local_advice
from .cnn_predictor import (
    DECISION_DIAGNOSIS,
    DECISION_NOT_PLANT,
    CNNPredictor,
    get_predictor,
)
from .prompts import build_system_prompt, build_user_context
from .provider import AIProvider, GroqProvider
from .schemas import AnalysisResult, ResultKind, Urgency
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
                self._cnn = get_predictor(
                    settings.cnn_model_path,
                    confidence_threshold=settings.cnn_confidence_threshold,
                    margin_threshold=settings.cnn_margin_threshold,
                    reject_threshold=settings.cnn_reject_threshold,
                    use_tta=settings.cnn_use_tta,
                )
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
        language: Optional[str] = None,
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

        # --- Tier 1: local CNN ---
        cnn_result = self._try_cnn(image_bytes, settings)

        if cnn_result is not None:
            decision = cnn_result["decision"]
            logger.info(
                "CNN: %s -> %s (conf=%.2f, margin=%.2f) — %s",
                cnn_result["primary_label"],
                decision,
                cnn_result["confidence"],
                cnn_result["margin"],
                cnn_result["reason"],
            )

            if decision == DECISION_DIAGNOSIS:
                result = self._cnn_to_result(
                    cnn_result, transcript, notes, onset_days, spread
                )
                if settings.enable_gradcam:
                    self._attach_gradcam(result, image_bytes)
                return self._enforce_invariants(result, crop_hint, bool(transcript))

            if decision == DECISION_NOT_PLANT:
                # Rejected locally — no API call, so this path is never rate-limited.
                return self._enforce_invariants(
                    AnalysisResult(
                        kind=ResultKind.NOT_A_LEAF,
                        is_plant=False,
                        is_leaf=False,
                        confidence=cnn_result["confidence"],
                        message=(
                            "No plant leaf was detected in this photo. Please upload a "
                            "clear, close-up image of a single crop leaf."
                        ),
                        provider="local_cnn",
                        model_id=cnn_result.get("model"),
                    ),
                    crop_hint,
                    bool(transcript),
                )
            # decision == escalate -> fall through to the hosted model.

        # --- Tier 2: hosted vision model ---
        # Reached when the CNN declined, is disabled, or has no checkpoint. Without
        # a configured key we say so honestly rather than inventing a diagnosis.
        if not settings.groq_api_key:
            return self._enforce_invariants(
                AnalysisResult(
                    kind=ResultKind.LOW_CONFIDENCE,
                    message=(
                        "We couldn't confidently identify this image. The offline "
                        "model covers tomato, potato, pepper and maize leaves — for "
                        "other crops or animals, a cloud AI key must be configured."
                    ),
                    provider="local_cnn",
                ),
                crop_hint,
                bool(transcript),
            )
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
        system_prompt = build_system_prompt(language=language)
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
        """Attempt CNN prediction. Returns None on any failure.

        An explicitly injected predictor is always used: the config flag governs
        whether we build one from settings, not whether a caller-supplied one is
        honoured.
        """
        if self._cnn is None and not settings.use_local_cnn:
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
    def _cnn_to_result(
        cnn: dict,
        transcript: str = "",
        notes: Optional[str] = None,
        onset_days: Optional[int] = None,
        spread: Optional[str] = None,
    ) -> AnalysisResult:
        """Build a complete AnalysisResult from a confident CNN prediction.

        Advice comes from the curated offline knowledge base rather than a
        language model: the disease is already identified, so generating the
        guidance would only add latency, a rate limit and a hallucination risk.
        """
        label = cnn["primary_label"]
        meta = label_meta(label)
        crop_name = meta.get("crop") if meta else None
        is_healthy = label.endswith("_HEALTHY")

        severity = local_advice.severity_for(label)
        advice = local_advice.advice_for(label)
        if advice is not None:
            # Fold in anything the farmer told us that changes the urgency of the
            # actions (not the diagnosis itself — that stays purely visual).
            extra = local_advice.contextual_notes(transcript, notes, onset_days, spread)
            if extra:
                advice.what_to_do_now = advice.what_to_do_now + extra

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
                for p in cnn.get("in_scope_predictions", [])[1:3]
                if p["label"] in all_label_ids()
            ],
            confidence=cnn["confidence"],
            severity_score=severity,
            urgency_level=local_advice.urgency_for(label),
            advice=advice,
            visual_evidence=(
                meta.get("description") if meta and not is_healthy else None
            ),
            provider="local_cnn",
            model_id=cnn.get("model"),
        )

    def _attach_gradcam(self, result: AnalysisResult, image_bytes: bytes) -> None:
        """Best-effort Grad-CAM overlay; never fails a diagnosis."""
        predictor = self.cnn
        if predictor is None:
            return
        try:
            from .gradcam import explain_image

            cam = explain_image(predictor, image_bytes)
            if cam:
                result.heatmap_png_b64 = cam["overlay_png_b64"]
                result.heatmap_focus = cam["peak_fraction"]
        except Exception:
            logger.warning("Grad-CAM failed (non-fatal)", exc_info=True)

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
