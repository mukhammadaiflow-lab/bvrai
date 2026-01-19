# Builder Engine - Local Development Environment

This directory contains the Docker Compose configuration for running Builder Engine locally.

## Prerequisites

- Docker 24.0+
- Docker Compose 2.20+
- 8GB+ RAM allocated to Docker
- Available ports: 3000, 8000, 5432, 6379, 5672, 15672, 9090, 3001

## Quick Start

```bash
# 1. Copy environment file
cp .env.example .env

# 2. Edit .env with your API keys
nano .env  # or vim, code, etc.

# 3. Start all services
make setup

# OR manually:
docker-compose up -d
```

## Services

| Service | Description | Port | URL |
|---------|-------------|------|-----|
| API | FastAPI backend | 8000 | http://localhost:8000 |
| Frontend | Next.js dashboard | 3000 | http://localhost:3000 |
| Voice Gateway | WebSocket voice | 8001 | ws://localhost:8001 |
| PostgreSQL | Primary database | 5432 | - |
| Redis | Cache & sessions | 6379 | - |
| RabbitMQ | Message queue | 5672 | http://localhost:15672 |
| Qdrant | Vector database | 6333 | - |
| Prometheus | Metrics | 9090 | http://localhost:9090 |
| Grafana | Dashboards | 3001 | http://localhost:3001 |
| Jaeger | Tracing | 16686 | http://localhost:16686 |

### Development Tools (Optional)

Enable with `make dev` or `docker-compose --profile dev up -d`:

| Service | Description | Port | URL |
|---------|-------------|------|-----|
| MailHog | Email testing | 8025 | http://localhost:8025 |
| pgAdmin | Database GUI | 5050 | http://localhost:5050 |

## Common Commands

```bash
# Start services
make up

# Stop services
make down

# View logs
make logs
make logs-api
make logs-worker

# Open shells
make shell-api
make db-shell
make redis-shell

# Database operations
make db-migrate
make db-seed
make db-reset

# Testing
make test
make test-cov
make lint

# Cleanup
make clean
make prune
```

## Configuration

### Required Environment Variables

```bash
# AI/LLM Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Voice/TTS
ELEVENLABS_API_KEY=...
DEEPGRAM_API_KEY=...

# Telephony
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...

# Payments (optional for dev)
STRIPE_SECRET_KEY=sk_test_...
STRIPE_PUBLISHABLE_KEY=pk_test_...
```

### Optional Configuration

```bash
# Service ports (if defaults conflict)
API_PORT=8000
FRONTEND_PORT=3000
POSTGRES_PORT=5432

# Monitoring passwords
GRAFANA_PASSWORD=admin
PGADMIN_PASSWORD=admin
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          nginx (optional)                        │
│                    (reverse proxy, load balancer)                │
└─────────────────────────────────────────────────────────────────┘
         │                        │                      │
         ▼                        ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    Frontend     │    │       API       │    │  Voice Gateway  │
│   (Next.js)     │    │   (FastAPI)     │    │   (WebSocket)   │
│    :3000        │    │     :8000       │    │     :8001       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │                        │
                              ▼                        │
                    ┌─────────────────┐                │
                    │     Workers     │                │
                    │    (Celery)     │◄───────────────┘
                    │   transcription │
                    │   tts, campaigns │
                    └─────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   PostgreSQL    │  │      Redis      │  │    RabbitMQ     │
│     :5432       │  │     :6379       │  │     :5672       │
└─────────────────┘  └─────────────────┘  └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │     Qdrant      │
                    │   (vectors)     │
                    │     :6333       │
                    └─────────────────┘
```

## Troubleshooting

### Services won't start

```bash
# Check logs
docker-compose logs

# Check system resources
docker stats

# Verify ports are available
lsof -i :8000 -i :3000 -i :5432
```

### Database connection issues

```bash
# Check PostgreSQL is healthy
docker-compose exec postgres pg_isready

# Reset database
make db-reset
```

### Out of disk space

```bash
# Clean up Docker resources
make prune

# Remove old volumes
docker volume prune -f
```

### API not responding

```bash
# Check API health
curl http://localhost:8000/health

# View API logs
make logs-api

# Restart API
docker-compose restart api
```

## Development Workflow

### Making Code Changes

The source code is mounted as volumes, so changes are reflected immediately:

- **API**: Changes to `src/` trigger hot-reload via uvicorn
- **Frontend**: Changes to `frontend/` trigger Next.js hot-reload
- **Workers**: Restart required: `docker-compose restart worker`

### Running Migrations

```bash
# Create a new migration
docker-compose exec api alembic revision --autogenerate -m "description"

# Apply migrations
make db-migrate
```

### Adding Dependencies

```bash
# Python dependencies
docker-compose exec api poetry add <package>
docker-compose build api

# Node dependencies
docker-compose exec frontend npm install <package>
docker-compose build frontend
```

## Production Considerations

This setup is for **development only**. For production:

1. Use proper secrets management (Vault, AWS Secrets Manager)
2. Enable SSL/TLS termination
3. Configure proper backups
4. Set up monitoring alerts
5. Use managed database services
6. Configure horizontal scaling
7. Enable rate limiting
8. Set up WAF and DDoS protection

See `/deploy/kubernetes` for production Kubernetes deployment.

## Support

- Documentation: https://docs.builderengine.io
- Discord: https://discord.gg/builderengine
- Issues: https://github.com/builderengine/builderengine/issues
