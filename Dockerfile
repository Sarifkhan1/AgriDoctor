# =============================================================================
# AgriDoctor AI - Backend Dockerfile
# Multi-stage build for optimized image size
# =============================================================================

FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create app directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------------------------
# Builder stage - install Python dependencies
# -------------------------------------------------------------------------
FROM base as builder

# Copy requirements
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch CPU-only first (smaller image)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------------------------------------------------------
# Production stage
# -------------------------------------------------------------------------
FROM base as production

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN groupadd -r agridoctor && useradd -r -g agridoctor agridoctor

# Copy application code
COPY backend/ ./backend/
COPY src/ ./src/
COPY config/ ./config/
COPY data/schemas/ ./data/schemas/

# Create data directories
RUN mkdir -p ./data/uploads/images ./data/uploads/speech ./models \
    && chown -R agridoctor:agridoctor /app

# Switch to non-root user
USER agridoctor

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health')" || exit 1

# Run the application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
