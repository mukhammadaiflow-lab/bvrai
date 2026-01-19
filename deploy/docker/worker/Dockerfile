# Builder Engine Worker Dockerfile
# Celery workers for background task processing

# =============================================================================
# Stage 1: Base Python image
# =============================================================================
FROM python:3.11-slim as base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=1.7.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 - \
    && ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry

# =============================================================================
# Stage 2: Builder
# =============================================================================
FROM base as builder

WORKDIR /app

COPY pyproject.toml poetry.lock* ./
RUN poetry install --only main --no-root

# =============================================================================
# Stage 3: Development
# =============================================================================
FROM base as development

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-root

COPY src/ ./src/

RUN groupadd -r builderengine && useradd -r -g builderengine builderengine \
    && mkdir -p /app/logs \
    && chown -R builderengine:builderengine /app

USER builderengine

# Run celery worker
CMD ["celery", "-A", "src.workers.celery_app", "worker", \
     "--loglevel=info", \
     "--concurrency=4", \
     "-Q", "default,calls,transcription,tts,campaigns,webhooks,analytics"]

# =============================================================================
# Stage 4: Production
# =============================================================================
FROM python:3.11-slim as production

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PATH="/app/.venv/bin:$PATH" \
    C_FORCE_ROOT=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libpq5 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY src/ ./src/

RUN groupadd -r builderengine && useradd -r -g builderengine builderengine \
    && mkdir -p /app/logs \
    && chown -R builderengine:builderengine /app

USER builderengine

CMD ["celery", "-A", "src.workers.celery_app", "worker", \
     "--loglevel=info", \
     "--concurrency=8", \
     "-Q", "default,calls,transcription,tts,campaigns,webhooks,analytics", \
     "--max-tasks-per-child=1000"]
