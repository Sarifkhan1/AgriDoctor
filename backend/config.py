"""
AgriDoctor AI - Application configuration.

All settings come from the environment (loaded from `.env` in development).
There are NO insecure defaults for secrets: the app refuses to start if
`secret_key` or `groq_api_key` are missing. This is deliberate (see the audit
and docs/implementation/06-SECURITY-HARDENING.md).
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Security (required) ---
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24  # 24 hours

    # --- Groq (optional — the escalation tier, not the primary engine) ---
    # Empty is allowed: with a trained CNN the app diagnoses its four supported
    # crops entirely offline. Without a key, cases the CNN declines are returned
    # as "low confidence" instead of being escalated.
    groq_api_key: str = ""
    groq_vision_model: str = "qwen/qwen3.6-27b"
    groq_audio_model: str = "whisper-large-v3-turbo"
    groq_text_model: str = "llama-3.3-70b-versatile"
    ai_timeout_seconds: int = 30
    ai_max_retries: int = 2

    # --- Local CNN (the primary engine) ---
    # Safe to default ON because the trained model has explicit out-of-scope
    # classes (OOS_NOT_PLANT / OOS_OTHER_PLANT) and only answers when it is both
    # confident and in-scope. Everything else escalates. If the checkpoint is
    # missing the predictor reports unavailable and every request escalates, so a
    # fresh clone with no model still behaves correctly.
    cnn_model_path: str = "data/models/agridoctor_cnn.pt"
    # Confidence is post-temperature-scaling (calibrated), so this threshold means
    # what it says — unlike a raw softmax score.
    cnn_confidence_threshold: float = 0.75
    # Minimum gap between top-1 and top-2 before we trust a prediction.
    cnn_margin_threshold: float = 0.10
    # Confidence needed to reject an image locally as "not a plant".
    cnn_reject_threshold: float = 0.60
    # Average predictions over the image and its mirror (~2x inference, still <100ms).
    cnn_use_tta: bool = True
    use_local_cnn: bool = True

    # --- Explainability ---
    # Attach a Grad-CAM heatmap to locally-diagnosed results.
    enable_gradcam: bool = True

    # --- Uploads / CORS ---
    max_upload_mb: int = 10
    max_audio_mb: int = 15
    allowed_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    # --- Logging ---
    log_level: str = "INFO"

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor. Import-safe; instantiates on first use."""
    return Settings()  # type: ignore[call-arg]
