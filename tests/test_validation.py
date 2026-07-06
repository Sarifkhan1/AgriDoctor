"""Upload validation unit tests (dependency-free sniffing + Pillow)."""

from io import BytesIO

import pytest
from fastapi import HTTPException
from PIL import Image

from backend.services import validation as v


def _png(w=128, h=128):
    buf = BytesIO()
    Image.new("RGB", (w, h), (10, 120, 80)).save(buf, "PNG")
    return buf.getvalue()


def test_sniff_png_jpeg_webp():
    assert v.sniff_image_mime(_png()) == "image/png"
    jpg = BytesIO()
    Image.new("RGB", (64, 64), "red").save(jpg, "JPEG")
    assert v.sniff_image_mime(jpg.getvalue()) == "image/jpeg"


def test_validate_image_ok():
    mime, ext = v.validate_image(_png(), "leaf.png", "image/png")
    assert mime == "image/png" and ext == ".png"


def test_rejects_non_image():
    with pytest.raises(HTTPException) as e:
        v.validate_image(b"just some text, not an image", "x.txt", "text/plain")
    assert e.value.status_code == 400


def test_rejects_tiny_image():
    with pytest.raises(HTTPException):
        v.validate_image(_png(10, 10), "tiny.png", "image/png")


def test_rejects_oversized(monkeypatch):
    from backend.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("MAX_UPLOAD_MB", "0")  # forces the size check to trip
    with pytest.raises(HTTPException):
        v.validate_image(_png(), "leaf.png", "image/png")
    get_settings.cache_clear()


def test_audio_sniffing():
    wav = b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 32
    assert v.sniff_audio_mime(wav) == "audio/wav"
    ogg = b"OggS" + b"\x00" * 32
    assert v.sniff_audio_mime(ogg) == "audio/ogg"
    assert v.sniff_audio_mime(b"nope") is None
