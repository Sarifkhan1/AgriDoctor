"""
AgriDoctor AI - FastAPI application entrypoint.

Diagnosis runs local-CNN-first with the hosted Groq vision model as the
escalation tier for crops the CNN wasn't trained on (backend/ai/*). Auth + case
history use SQLite. See docs/implementation/.
"""

import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import get_settings
from .db import init_db
from .routers import analyze, auth, cases, media

APP_NAME = "AgriDoctor AI"
VERSION = "2.0.0"

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("agridoctor")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info(
        "%s v%s started (vision=%s)", APP_NAME, VERSION, settings.groq_vision_model
    )
    yield


app = FastAPI(
    title=APP_NAME,
    version=VERSION,
    description="Multimodal agricultural disease diagnosis API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


# --------------------------------------------------------------------------- #
# Lightweight in-process rate limiting (per IP, per sensitive route).
# For multi-worker deployments, put a shared limiter (nginx/Redis) in front.
# --------------------------------------------------------------------------- #
_RATE_LIMITS = {"/api/analyze": (10, 60), "/api/auth/login": (5, 60)}
_buckets: dict[tuple[str, str], deque] = defaultdict(deque)


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    limit_cfg = _RATE_LIMITS.get(request.url.path)
    if limit_cfg and request.method == "POST":
        max_calls, window = limit_cfg
        key = (request.url.path, _client_ip(request))
        now = time.monotonic()
        bucket = _buckets[key]
        while bucket and now - bucket[0] > window:
            bucket.popleft()
        if len(bucket) >= max_calls:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "You're going too fast. Please wait a moment and retry."
                },
            )
        bucket.append(now)
    return await call_next(request)


app.include_router(auth.router)
app.include_router(cases.router)
app.include_router(analyze.router)
app.include_router(media.router)


@app.get("/health", tags=["System"])
async def health_check():
    """Liveness plus which inference engines are actually usable right now.

    Reports the CNN without loading it — `available` is a file check, so hitting
    /health stays cheap and doesn't pull PyTorch into memory on a cold process.
    """
    from .config import get_settings

    settings = get_settings()
    engines = {
        "local_cnn": {
            "enabled": settings.use_local_cnn,
            "checkpoint": settings.cnn_model_path,
            "available": Path(settings.cnn_model_path).exists(),
        },
        "hosted_vlm": {
            "configured": bool(settings.groq_api_key),
            "model": settings.groq_vision_model,
        },
    }
    return {"status": "healthy", "version": VERSION, "engines": engines}


@app.get("/", tags=["System"])
async def root():
    return {"name": APP_NAME, "version": VERSION, "docs": "/docs", "health": "/health"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
