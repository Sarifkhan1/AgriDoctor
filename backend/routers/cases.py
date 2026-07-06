"""Case history endpoints (read-oriented). Diagnosis happens via /api/analyze."""

import json

from fastapi import APIRouter, Depends, HTTPException

from ..db import get_db
from ..security import get_current_user

router = APIRouter(prefix="/api/cases", tags=["Cases"])


@router.get("")
async def list_cases(
    limit: int = 20,
    offset: int = 0,
    current_user: dict = Depends(get_current_user),
):
    limit = max(1, min(limit, 100))
    offset = max(0, offset)
    with get_db() as conn:
        rows = conn.execute(
            """SELECT c.*, p.kind, p.disease_name, p.primary_label, p.confidence,
                      p.severity_score, p.urgency_level, p.detected_crop
               FROM cases c
               LEFT JOIN predictions p ON p.case_id = c.id
               WHERE c.user_id = ?
               ORDER BY c.created_at DESC
               LIMIT ? OFFSET ?""",
            (current_user["id"], limit, offset),
        ).fetchall()
    return [dict(r) for r in rows]


@router.get("/{case_id}")
async def get_case(case_id: str, current_user: dict = Depends(get_current_user)):
    with get_db() as conn:
        case = conn.execute(
            "SELECT * FROM cases WHERE id = ? AND user_id = ?",
            (case_id, current_user["id"]),
        ).fetchone()
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")

        media = conn.execute(
            "SELECT * FROM media WHERE case_id = ?", (case_id,)
        ).fetchall()
        metadata = conn.execute(
            "SELECT * FROM metadata WHERE case_id = ?", (case_id,)
        ).fetchone()
        prediction = conn.execute(
            "SELECT * FROM predictions WHERE case_id = ? ORDER BY created_at DESC LIMIT 1",
            (case_id,),
        ).fetchone()

    result = dict(case)
    result["media"] = [dict(m) for m in media]
    result["metadata"] = dict(metadata) if metadata else None
    if prediction:
        pred = dict(prediction)
        pred["secondary_labels"] = json.loads(pred.get("secondary_labels_json") or "[]")
        pred["advice"] = json.loads(pred.get("advice_json") or "null")
        result["prediction"] = pred
    else:
        result["prediction"] = None
    return result
