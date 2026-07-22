# =============================================================================
# AgriDoctor AI - Backend Dockerfile (multi-stage, slim serving image)
#
# The serving image includes PyTorch because the local CNN is the primary
# diagnosis engine — the hosted model only handles escalations. Torch is
# installed from the CPU-only wheel index: the default PyPI wheels bundle CUDA
# and weigh ~2.5GB, the CPU wheels ~200MB, and a VPS has no GPU to use either way.
#
# Builds:
#   default                                        -> API + local CNN (production)
#   --build-arg REQUIREMENTS=requirements.txt      -> API only, CNN disabled,
#                                                     every request escalates
#   --build-arg REQUIREMENTS=requirements-ml.txt   -> training/annotation image
#
# The trained checkpoint is COPYed in below. Without it the API still starts and
# /health reports local_cnn.available=false, so a missing model is visible rather
# than silently degrading every request to the rate-limited hosted tier.
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

ARG REQUIREMENTS=requirements-serve.txt

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt requirements-ml.txt requirements-serve.txt ./
# The CPU wheel index is an *extra* index, so non-torch packages still resolve
# from PyPI normally while torch/torchvision come from the CPU-only build.
RUN pip install --no-cache-dir -r ${REQUIREMENTS} \
        --extra-index-url https://download.pytorch.org/whl/cpu

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
