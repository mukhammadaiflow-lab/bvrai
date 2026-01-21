# =============================================================================
# BVRAI Backend API - Production Dockerfile for Railway
# =============================================================================
# Multi-stage build for optimal performance
# =============================================================================

# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt ./

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Production Runner
FROM python:3.11-slim AS runner

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libpq5 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy source code
COPY bvrai_core/ ./bvrai_core/
COPY alembic/ ./alembic/
COPY alembic.ini ./

# Create non-root user for security
RUN groupadd -r bvrai && useradd -r -g bvrai bvrai \
    && mkdir -p /app/logs /app/data \
    && chown -R bvrai:bvrai /app

USER bvrai

# Railway uses PORT env variable (default to 8000)
EXPOSE 8000
ENV PORT=8000

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Start command - Railway will set PORT
CMD ["sh", "-c", "uvicorn bvrai_core.api.app:create_app --host 0.0.0.0 --port ${PORT} --factory"]
