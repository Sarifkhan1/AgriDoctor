"""The instant analysis endpoint: POST /api/analyze."""

from typing import Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile

from ..security import get_current_user_optional
from ..services.analysis import AnalysisService

router = APIRouter(prefix="/api", tags=["Analyze"])
_service = AnalysisService()


@router.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    audio: Optional[UploadFile] = File(None),
    crop_hint: Optional[str] = Form(None),
    onset_days: Optional[int] = Form(None),
    spread: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
    user: Optional[dict] = Depends(get_current_user_optional),
):
    image_bytes = await image.read()
    audio_payload = None
    if audio is not None:
        audio_bytes = await audio.read()
        if audio_bytes:
            audio_payload = {
                "bytes": audio_bytes,
                "filename": audio.filename or "audio.webm",
            }

    result, case_id = await _service.run(
        image_bytes=image_bytes,
        image_filename=image.filename,
        image_content_type=image.content_type,
        audio=audio_payload,
        crop_hint=crop_hint,
        onset_days=onset_days,
        spread=spread,
        notes=notes,
        user=user,
    )
    return {**result.model_dump(), "case_id": case_id}
