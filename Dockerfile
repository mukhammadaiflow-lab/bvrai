# BVRAI API Dockerfile for Railway
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libpq5 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy source code
COPY bvrai_core/ ./bvrai_core/
COPY alembic/ ./alembic/
COPY alembic.ini ./

# Create non-root user
RUN groupadd -r bvrai && useradd -r -g bvrai bvrai \
    && mkdir -p /app/logs \
    && chown -R bvrai:bvrai /app

USER bvrai

# Railway uses PORT env variable
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Start command - Railway will set PORT
CMD uvicorn bvrai_core.api.app:create_app --host 0.0.0.0 --port ${PORT:-8000} --factory
