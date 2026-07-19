"""
AI provider abstraction + Groq implementation.

The interface keeps the vendor swappable and makes the analyzer testable with a
fake provider (no network, no cost). GroqProvider uses open-source models:
Llama 4 Scout (vision) and Whisper-large-v3 (audio).
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

from ..config import get_settings

logger = logging.getLogger(__name__)

# Known vision-capable Groq model IDs, in preference order. Groq deprecates and
# renames models periodically (e.g. Llama-4 Maverick was retired in Feb 2026),
# so we never hard-depend on one ID: we try the configured model first, then fall
# back to whichever of these the account can actually see. This keeps live image
# analysis working across Groq's model churn without a code change.
VISION_MODEL_FALLBACKS: List[str] = [
    "qwen/qwen3.6-27b",
    "qwen/qwen2.5-vl-32b-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
]


class AIProvider(ABC):
    @abstractmethod
    def transcribe(self, audio_bytes: bytes, filename: str) -> str:
        """Return the transcript text for an audio clip (best-effort)."""

    @abstractmethod
    def analyze_image(
        self, image_data_url: str, system_prompt: str, user_context: str
    ) -> str:
        """Return the model's raw JSON string for the image + context."""


class GroqProvider(AIProvider):
    def __init__(self) -> None:
        from groq import Groq  # imported lazily so the module loads without the dep

        settings = get_settings()
        self._settings = settings
        self._client = Groq(
            api_key=settings.groq_api_key,
            timeout=float(settings.ai_timeout_seconds),
        )
        self._vision_model: Optional[str] = None  # resolved lazily on first use

    # ------------------------------------------------------------------ #
    def _create_with_retry(self, **kwargs):
        """chat.completions.create with retry on rate-limit / transient errors.

        Groq's free tier has a low tokens-per-minute budget, so a burst of
        requests can hit HTTP 429. We honour the server's suggested wait and
        retry a bounded number of times so a transient limit doesn't surface as
        a hard failure to the farmer.
        """
        import time

        from groq import APIConnectionError, InternalServerError, RateLimitError

        attempts = max(1, self._settings.ai_max_retries) + 1
        for i in range(attempts):
            try:
                return self._client.chat.completions.create(**kwargs)
            except RateLimitError as exc:
                if i >= attempts - 1:
                    raise
                wait = _parse_retry_after(str(exc)) or (2 ** i)
                logger.warning("Groq rate-limited; retrying in %.1fs", wait)
                time.sleep(min(wait, 12.0))
            except (APIConnectionError, InternalServerError):
                if i >= attempts - 1:
                    raise
                time.sleep(2 ** i)

    def _available_models(self) -> set:
        try:
            return {m.id for m in self._client.models.list().data}
        except Exception:
            logger.warning("Could not list Groq models", exc_info=True)
            return set()

    def _resolve_vision_model(self) -> str:
        """Pick a vision model the account can actually use.

        Groq retires/renames vision models over time, so we don't trust the
        configured ID blindly: we intersect the configured model + our known
        fallbacks with the account's live model list and use the first match.
        """
        if self._vision_model:
            return self._vision_model

        available = self._available_models()
        preference = [self._settings.groq_vision_model, *VISION_MODEL_FALLBACKS]
        # de-dupe while preserving order
        seen = set()
        ordered = [m for m in preference if not (m in seen or seen.add(m))]

        if available:
            for model in ordered:
                if model in available:
                    if model != self._settings.groq_vision_model:
                        logger.warning(
                            "Configured vision model %r unavailable; using %r",
                            self._settings.groq_vision_model,
                            model,
                        )
                    self._vision_model = model
                    return model
            logger.error(
                "No known vision model is available on this Groq account. "
                "Available: %s",
                sorted(available),
            )
        # Fall back to the configured value; the call will surface the real error.
        self._vision_model = self._settings.groq_vision_model
        return self._vision_model

    def transcribe(self, audio_bytes: bytes, filename: str) -> str:
        resp = self._client.audio.transcriptions.create(
            file=(filename or "audio.webm", audio_bytes),
            model=self._settings.groq_audio_model,
            response_format="text",
        )
        return resp if isinstance(resp, str) else getattr(resp, "text", "")

    def analyze_image(
        self, image_data_url: str, system_prompt: str, user_context: str
    ) -> str:
        model = self._resolve_vision_model()
        try:
            return self._vision_call(model, image_data_url, system_prompt, user_context)
        except Exception as exc:  # noqa: BLE001 - want to inspect + maybe retry
            # If the chosen model was decommissioned between resolution and call,
            # re-resolve once against the live list and retry with a fresh pick.
            if _looks_like_model_error(exc):
                logger.warning("Vision model %r failed (%s); re-resolving", model, exc)
                self._vision_model = None
                new_model = self._resolve_vision_model()
                if new_model != model:
                    return self._vision_call(
                        new_model, image_data_url, system_prompt, user_context
                    )
            raise

    def _vision_call(
        self, model: str, image_data_url: str, system_prompt: str, user_context: str
    ) -> str:
        base = dict(
            model=model,
            temperature=0.2,
            max_tokens=1400,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_context},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                },
            ],
        )
        json_fmt = {"response_format": {"type": "json_object"}}

        # Request shapes tried in order. Thinking models (e.g. Qwen3) emit
        # reasoning tokens that break strict json_object validation, so we first
        # suppress reasoning; if a model rejects that param we drop it; and as a
        # last resort we drop json_object entirely and parse the raw text
        # ourselves (the analyzer strips <think> blocks and extracts the object).
        attempts = [
            {**base, **json_fmt, "reasoning_effort": "none"},
            {**base, **json_fmt},
            {**base, "reasoning_effort": "none"},
            {**base},
        ]

        last_exc: Optional[Exception] = None
        for kwargs in attempts:
            try:
                completion = self._create_with_retry(**kwargs)
                return completion.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001
                if _looks_like_model_error(exc):
                    raise  # a dead model — let the caller re-resolve
                last_exc = exc
                logger.info("Vision request shape failed (%s); trying next", exc)
        assert last_exc is not None
        raise last_exc


def _looks_like_model_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    if "rate_limit" in msg or "rate limit" in msg:
        return False  # a 429 is not a dead model
    return any(
        k in msg
        for k in ("model", "decommission", "does not exist", "not found", "deprecat")
    )


def _parse_retry_after(message: str) -> Optional[float]:
    """Extract the 'try again in 1.5s' / '960ms' hint from a Groq 429 message."""
    import re

    m = re.search(r"try again in\s+([\d.]+)\s*(ms|s)", message, re.IGNORECASE)
    if not m:
        return None
    value = float(m.group(1))
    return value / 1000.0 if m.group(2).lower() == "ms" else value
