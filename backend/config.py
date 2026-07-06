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

    # --- Groq (required) ---
    groq_api_key: str
    groq_vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    groq_audio_model: str = "whisper-large-v3-turbo"
    groq_text_model: str = "llama-3.3-70b-versatile"
    ai_timeout_seconds: int = 30
    ai_max_retries: int = 2

    # --- Local CNN (optional — requires torch + a trained model) ---
    cnn_model_path: str = "data/models/best_model.pt"
    cnn_confidence_threshold: float = 0.80
    use_local_cnn: bool = True

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
