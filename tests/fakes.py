"""Test doubles."""

import json

from backend.ai.provider import AIProvider


class FakeProvider(AIProvider):
    """Returns a canned JSON payload; never hits the network."""

    def __init__(self, payload: dict, transcript: str = ""):
        self.payload = payload
        self._transcript = transcript

    def transcribe(self, audio_bytes: bytes, filename: str) -> str:
        return self._transcript

    def analyze_image(self, image_data_url, system_prompt, user_context) -> str:
        return json.dumps(self.payload)
