"""Serve stored media files (owner-only)."""

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from ..db import get_db
from ..security import get_current_user

router = APIRouter(prefix="/api/media", tags=["Media"])


@router.get("/{media_id}")
async def get_media(media_id: str, current_user: dict = Depends(get_current_user)):
    with get_db() as conn:
        row = conn.execute(
            """SELECT m.uri, m.content_type
               FROM media m JOIN cases c ON c.id = m.case_id
               WHERE m.id = ? AND c.user_id = ?""",
            (media_id, current_user["id"]),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Media not found")
    path = Path(row["uri"])
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File missing")
    return FileResponse(
        str(path), media_type=row["content_type"] or "application/octet-stream"
    )
