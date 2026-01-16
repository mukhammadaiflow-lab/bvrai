# BVRAI Voice AI Platform - Backend API Dockerfile
# Simplified for MVP development

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY bvrai_core/ ./bvrai_core/

# Create non-root user
RUN groupadd -r bvrai && useradd -r -g bvrai bvrai \
    && chown -R bvrai:bvrai /app

USER bvrai

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the API
CMD ["uvicorn", "bvrai_core.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
