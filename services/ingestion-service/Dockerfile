# Ingestion Service Dockerfile
FROM python:3.11-slim

RUN groupadd -r appgroup && useradd -r -g appgroup appuser
WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir .

COPY --chown=appuser:appgroup app/ ./app/
RUN mkdir -p /app/data && chown appuser:appgroup /app/data

ENV PYTHONUNBUFFERED=1
USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:3004/health').raise_for_status()" || exit 1

EXPOSE 3004
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3004"]
