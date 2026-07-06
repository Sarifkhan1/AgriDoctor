"""
AI provider abstraction + Groq implementation.

The interface keeps the vendor swappable and makes the analyzer testable with a
fake provider (no network, no cost). GroqProvider uses open-source models:
Llama 4 Scout (vision) and Whisper-large-v3 (audio).
"""

from abc import ABC, abstractmethod
from typing import Optional

from ..config import get_settings


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
        completion = self._client.chat.completions.create(
            model=self._settings.groq_vision_model,
            temperature=0.2,
            max_tokens=1400,
            response_format={"type": "json_object"},
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
        return completion.choices[0].message.content or ""
