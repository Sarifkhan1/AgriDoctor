"""
Upload validation — dependency-free magic-byte sniffing + Pillow verification.

We intentionally do NOT rely on libmagic/python-magic (a system dependency that
isn't present everywhere). Image/audio types are detected from their byte
signatures, and images are additionally opened with Pillow to confirm validity
and dimensions.
"""

from io import BytesIO

from fastapi import HTTPException
from PIL import Image

from ..config import get_settings

_IMAGE_EXT = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}


def sniff_image_mime(data: bytes) -> str | None:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def sniff_audio_mime(data: bytes) -> str | None:
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "audio/wav"
    if data[:4] == b"\x1a\x45\xdf\xa3":  # EBML (webm/mkv)
        return "audio/webm"
    if data[:4] == b"OggS":
        return "audio/ogg"
    if data[:3] == b"ID3" or (
        len(data) > 1 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0
    ):
        return "audio/mpeg"
    if data[4:8] == b"ftyp":  # mp4 / m4a
        return "audio/mp4"
    return None


def validate_image(
    data: bytes, filename: str, declared_ct: str | None
) -> tuple[str, str]:
    """Validate an uploaded image. Returns (mime, extension). Raises HTTP 400."""
    settings = get_settings()
    if not data:
        raise HTTPException(400, "The image file is empty.")
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(
            400, f"Image exceeds the {settings.max_upload_mb} MB limit."
        )

    mime = sniff_image_mime(data)
    if mime is None:
        raise HTTPException(400, "Please upload a valid JPG, PNG, or WebP image.")

    try:
        with Image.open(BytesIO(data)) as im:
            im.verify()  # integrity check
        with Image.open(BytesIO(data)) as im2:
            width, height = im2.size  # re-open: verify() leaves the file unusable
    except Exception:
        raise HTTPException(400, "That file isn't a readable image.")

    if width < 64 or height < 64:
        raise HTTPException(400, "Image is too small to analyze (min 64x64).")

    return mime, _IMAGE_EXT[mime]


def validate_audio(audio: dict) -> str:
    """Validate an uploaded audio clip. Returns mime. Raises HTTP 400."""
    settings = get_settings()
    data = audio.get("bytes", b"")
    if not data:
        raise HTTPException(400, "The audio file is empty.")
    if len(data) > settings.max_audio_mb * 1024 * 1024:
        raise HTTPException(
            400, f"Voice note exceeds the {settings.max_audio_mb} MB limit."
        )
    mime = sniff_audio_mime(data)
    if mime is None:
        raise HTTPException(
            400, "Unsupported audio format. Use WebM, WAV, MP3, M4A, or OGG."
        )
    return mime
