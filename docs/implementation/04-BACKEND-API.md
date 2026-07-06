# 04 — Backend & API

## 4.1 The new instant endpoint

Replaces the create → upload → run-job → poll dance with **one call**.

```
POST /api/analyze
Content-Type: multipart/form-data
Auth: optional (Bearer). Anonymous allowed; if authed, the case is saved to history.

Fields:
  image      file      required   the leaf photo (jpg/png/webp)
  audio      file      optional   voice note (webm/wav/m4a/mp3/ogg)
  crop_hint  string    optional   "tomato" (Step-1 pick — treated as a hint only)
  onset_days integer   optional
  spread     string    optional   slow|moderate|fast
  notes      string    optional   free text

200 → AnalysisResult (see 03 §3.3), plus "case_id" if persisted.
400 → invalid/oversized/non-image upload.
429 → rate limited.
503 → AI provider unavailable/timeout.
```

### Handler sketch (`backend/routers/analyze.py`)

```python
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from typing import Optional
from ..services.analysis import AnalysisService
from ..security import get_current_user_optional

router = APIRouter(prefix="/api", tags=["Analyze"])
service = AnalysisService()

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
        audio_payload = {"bytes": await audio.read(), "filename": audio.filename}

    result, case_id = await service.run(
        image_bytes=image_bytes, image_filename=image.filename,
        image_content_type=image.content_type, audio=audio_payload,
        crop_hint=crop_hint, onset_days=onset_days, spread=spread, notes=notes,
        user=user,
    )
    return {**result.model_dump(), "case_id": case_id}
```

## 4.2 The service orchestrator (`backend/services/analysis.py`)

Ties validation → NLU → analyzer → persistence. Runs the (sync, network-bound) Groq calls in a threadpool so the event loop isn't blocked.

```python
from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool
from ..ai.analyzer import Analyzer
from .validation import validate_image, validate_audio   # 06 §6.5
from .nlu_hints import extract_hints                      # wraps src/preprocessing/text_nlu

class AnalysisService:
    def __init__(self):
        self.analyzer = Analyzer()

    async def run(self, *, image_bytes, image_filename, image_content_type,
                  audio, crop_hint, onset_days, spread, notes, user):
        mime = validate_image(image_bytes, image_filename, image_content_type)  # raises 400
        if audio:
            validate_audio(audio)                                               # raises 400

        nlu_hints = extract_hints(notes)   # deterministic, offline, cheap

        try:
            result = await run_in_threadpool(
                self.analyzer.analyze,
                image_bytes=image_bytes, image_mime=mime, audio=audio,
                crop_hint=crop_hint, notes=notes, onset_days=onset_days,
                spread=spread, nlu_hints=nlu_hints,
            )
        except Exception as e:
            # Log full error server-side; return a clean 503
            logger.exception("AI analysis failed")
            raise HTTPException(503, "Analysis service is temporarily unavailable. Please try again.")

        case_id = None
        if user:
            case_id = await run_in_threadpool(self._persist, user, image_bytes,
                                              image_filename, result)
        return result, case_id
```

Persistence is **best-effort** and never blocks the user's result — wrap it so a DB hiccup can't turn a good diagnosis into an error.

## 4.3 Backward compatibility

Keep the existing endpoints working during migration:
- `POST /api/cases`, `GET /api/cases`, `GET /api/cases/{id}` — history CRUD, refactored into `routers/cases.py`.
- The old `POST /api/cases/{id}/run` + polling can remain for now but is deprecated; the frontend stops using it once `/api/analyze` lands.
- **Delete `generate_mock_prediction`** once `/api/analyze` is verified — no dual sources of truth.

## 4.4 Auth changes

- `get_current_user_optional`: like `get_current_user` but returns `None` instead of raising when no/invalid token — enables anonymous instant analysis (D8).
- Registration/login keep PBKDF2 hashing (already correct) but:
  - Raise password minimum to **8** and add a basic strength check.
  - Add rate limiting on `/api/auth/login` (see 4.5).
- `SECRET_KEY` becomes mandatory via `config.py` (no insecure fallback) — see 06.

## 4.5 Rate limiting

Two reasons: protect the Groq free-tier quota, and stop auth brute force.

- Use `slowapi` (Starlette-friendly) or a small in-process token bucket keyed on client IP.
- Suggested limits: `POST /api/analyze` → **10/min/IP**, `POST /api/auth/login` → **5/min/IP**.
- Nginx already has a `limit_req` zone ([nginx.conf](../../nginx.conf)); keep it as a coarse outer layer and add the app-level limiter for per-route control.

```python
from slowapi import Limiter
from slowapi.util import get_remote_address
limiter = Limiter(key_func=get_remote_address)
# @limiter.limit("10/minute") on the analyze route
```

## 4.6 Observability (fixes audit §13 — backend has zero logging today)

- Configure `logging` in `main.py` (structured, level from `LOG_LEVEL`).
- Log per request: method, path, status, duration, and for `/api/analyze`: `kind`, `detected_crop`, `confidence`, `model_id`, latency — **never** the API key or full user PII.
- Store `raw_response` on the prediction row for offline calibration/debugging.
- Add a `/metrics` counter later if needed (Prometheus) — optional.

## 4.7 App wiring (`backend/main.py` after refactor)

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .db import init_db
from .routers import auth, cases, analyze

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(title="AgriDoctor AI", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,   # NOT "*"
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(auth.router)
app.include_router(cases.router)
app.include_router(analyze.router)
```

Note: `@app.on_event("startup")` and `datetime.utcnow()` are deprecated — replaced by `lifespan` and timezone-aware `datetime.now(timezone.utc)` (audit §12).
