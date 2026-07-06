# =============================================================================
# AgriDoctor AI - Backend Dockerfile (multi-stage, slim serving image)
# The serving API needs no heavy ML libs (inference runs on Groq). Build the
# annotator/training image with:  --build-arg REQUIREMENTS=requirements-ml.txt
# =============================================================================

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# -------------------------------------------------------------------------
# Builder stage
# -------------------------------------------------------------------------
FROM base AS builder

ARG REQUIREMENTS=requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt requirements-ml.txt ./
RUN pip install --no-cache-dir -r ${REQUIREMENTS}

# -------------------------------------------------------------------------
# Production stage
# -------------------------------------------------------------------------
FROM base AS production

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN groupadd -r agridoctor && useradd -r -g agridoctor agridoctor

COPY backend/ ./backend/
COPY src/ ./src/
COPY config/ ./config/
COPY data/schemas/ ./data/schemas/
COPY data/taxonomy.json ./data/taxonomy.json

RUN mkdir -p ./data/uploads/images ./data/uploads/speech ./models \
    && chown -R agridoctor:agridoctor /app

USER agridoctor

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
