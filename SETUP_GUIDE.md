# Builder Voice AI Platform - Setup Guide

Complete step-by-step guide to get the platform running.

---

## Table of Contents

1. [Quick Start (5 minutes)](#quick-start)
2. [Get Your API Keys (15-30 minutes)](#get-your-api-keys)
3. [Configure Environment](#configure-environment)
4. [Run the Platform](#run-the-platform)
5. [Verify Installation](#verify-installation)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites

You need these installed on your machine:

| Tool | Version | Check Command | Install |
|------|---------|---------------|---------|
| Docker | 20.10+ | `docker --version` | [docker.com](https://docs.docker.com/get-docker/) |
| Docker Compose | 2.0+ | `docker compose version` | Included with Docker Desktop |
| Python | 3.10+ | `python --version` | [python.org](https://www.python.org/downloads/) |
| Git | 2.0+ | `git --version` | [git-scm.com](https://git-scm.com/) |

### Step 1: Clone and Setup

```bash
# Navigate to project directory
cd /path/to/bvrai

# Copy environment template
cp .env.example .env

# Open .env in your editor and add your API keys (see next section)
nano .env  # or: code .env
```

### Step 2: Start Services

```bash
# Start all services with Docker
docker compose up -d

# Check all services are running
docker compose ps
```

### Step 3: Run Database Migrations

```bash
# Install Python dependencies (for migrations)
pip install -r requirements.txt

# Run migrations
alembic upgrade head
```

**That's it!** The platform is now running. But you need API keys for STT/TTS/LLM to actually process voice calls.

---

## Get Your API Keys

### MINIMUM REQUIRED (Pick one from each category)

#### 1. Speech-to-Text (STT) - Choose ONE

| Provider | Free Tier | Sign Up | Time |
|----------|-----------|---------|------|
| **Deepgram** (Recommended) | $200 credit | [console.deepgram.com/signup](https://console.deepgram.com/signup) | 2 min |
| OpenAI Whisper | Pay-as-you-go | [platform.openai.com/signup](https://platform.openai.com/signup) | 2 min |
| AssemblyAI | Free tier | [assemblyai.com/dashboard/signup](https://www.assemblyai.com/dashboard/signup) | 2 min |

**Recommended: Deepgram** - Best real-time performance, $200 free credit

#### 2. Text-to-Speech (TTS) - Choose ONE

| Provider | Free Tier | Sign Up | Time |
|----------|-----------|---------|------|
| **ElevenLabs** (Recommended) | 10k chars/month | [elevenlabs.io/sign-up](https://elevenlabs.io/sign-up) | 2 min |
| OpenAI TTS | Pay-as-you-go | [platform.openai.com/signup](https://platform.openai.com/signup) | 2 min |
| PlayHT | Limited free | [play.ht/signup](https://play.ht/signup) | 2 min |

**Recommended: ElevenLabs** - Best voice quality, 10k characters free

#### 3. LLM (Language Model) - Choose ONE

| Provider | Free Tier | Sign Up | Time |
|----------|-----------|---------|------|
| **OpenAI** (Recommended) | $5 credit (new users) | [platform.openai.com/signup](https://platform.openai.com/signup) | 2 min |
| Anthropic Claude | Pay-as-you-go | [console.anthropic.com](https://console.anthropic.com/) | 2 min |
| Groq (Fast) | Generous free tier | [console.groq.com](https://console.groq.com/) | 2 min |

**Recommended: OpenAI** - GPT-4o is excellent for voice agents

#### 4. Telephony (For Phone Calls) - OPTIONAL

| Provider | Free Tier | Sign Up | Time |
|----------|-----------|---------|------|
| Twilio | $15 credit | [twilio.com/try-twilio](https://www.twilio.com/try-twilio) | 5 min |

**Note:** Telephony is optional. You can test via WebRTC (browser) without it.

---

## Configure Environment

### Edit your .env file

Open `.env` and fill in the API keys you obtained:

```bash
# Open with your preferred editor
nano .env
# or
code .env
# or
vim .env
```

### Minimum Configuration

```env
# ===========================================
# MINIMUM REQUIRED - Fill these in
# ===========================================

# Database (change password!)
POSTGRES_PASSWORD=your_secure_password_here

# STT - Deepgram (or your chosen provider)
DEEPGRAM_API_KEY=paste_your_deepgram_key_here

# TTS - ElevenLabs (or your chosen provider)
ELEVENLABS_API_KEY=paste_your_elevenlabs_key_here

# LLM - OpenAI (or your chosen provider)
OPENAI_API_KEY=paste_your_openai_key_here

# Security (generate random string)
JWT_SECRET=generate_a_random_32_character_string
APP_SECRET_KEY=generate_another_random_32_char_string
```

### Generate Random Secrets

```bash
# Generate random secret keys
python -c "import secrets; print(secrets.token_hex(32))"
```

Run this twice - once for `JWT_SECRET` and once for `APP_SECRET_KEY`.

### Full Configuration (Optional)

For production or additional providers, see the full list in `.env.example`.

---

## Run the Platform

### Option A: Docker Compose (Recommended)

```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down
```

### Option B: Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start PostgreSQL and Redis (via Docker)
docker compose up -d postgres redis qdrant

# Run migrations
alembic upgrade head

# Start the platform API
cd services/platform-api
uvicorn main:app --reload --port 8086
```

---

## Verify Installation

### 1. Check Docker Services

```bash
docker compose ps
```

All services should show "Up" status:

```
NAME                     STATUS
bvrai-postgres-1         Up (healthy)
bvrai-redis-1            Up (healthy)
bvrai-platform-api-1     Up (healthy)
bvrai-asr-service-1      Up (healthy)
bvrai-tts-service-1      Up (healthy)
...
```

### 2. Check API Health

```bash
# Platform API
curl http://localhost:8086/health

# ASR Service
curl http://localhost:8082/health

# TTS Service
curl http://localhost:8083/health
```

All should return `{"status": "healthy"}` or similar.

### 3. Check Database

```bash
# Connect to PostgreSQL
docker compose exec postgres psql -U bvrai -d bvrai

# List tables
\dt

# Exit
\q
```

### 4. Test Voice Pipeline (Optional)

```bash
# Run the test suite
pytest tests/ -v
```

---

## Service URLs

Once running, access these URLs:

| Service | URL | Description |
|---------|-----|-------------|
| Platform API | http://localhost:8086 | Main REST API |
| API Docs | http://localhost:8086/docs | Swagger UI |
| WebRTC Gateway | http://localhost:8087 | Browser-based calls |
| Telephony | http://localhost:8080 | Twilio webhooks |
| PostgreSQL | localhost:5432 | Database |
| Redis | localhost:6379 | Cache |
| Qdrant | http://localhost:6333 | Vector DB |

---

## Troubleshooting

### Docker Issues

```bash
# Restart all services
docker compose down && docker compose up -d

# Rebuild images
docker compose build --no-cache

# View specific service logs
docker compose logs -f platform-api
```

### Database Issues

```bash
# Reset database (WARNING: deletes all data)
docker compose down -v
docker compose up -d postgres
alembic upgrade head
```

### API Key Issues

```bash
# Test Deepgram API key
curl -X POST 'https://api.deepgram.com/v1/listen' \
  -H "Authorization: Token YOUR_API_KEY" \
  -H "Content-Type: audio/wav" \
  --data-binary @test.wav

# Test OpenAI API key
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### Port Conflicts

If ports are in use:

```bash
# Find what's using a port
lsof -i :8086

# Kill the process
kill -9 <PID>

# Or change ports in docker-compose.yml
```

---

## Next Steps

1. **Create your first agent** via the API:
   ```bash
   curl -X POST http://localhost:8086/api/v1/agents \
     -H "Content-Type: application/json" \
     -d '{
       "name": "My First Agent",
       "system_prompt": "You are a helpful assistant..."
     }'
   ```

2. **Test via WebRTC** - Open http://localhost:8087 in your browser

3. **Connect Twilio** for phone calls (see Twilio setup in docs)

4. **Build a frontend** - Use the REST API at http://localhost:8086/docs

---

## Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Documentation**: See `/docs` folder
- **API Reference**: http://localhost:8086/docs

---

## Cost Estimates

| Provider | Usage | Approx. Cost |
|----------|-------|--------------|
| Deepgram | 1 hour of audio | ~$0.25 |
| ElevenLabs | 10k characters | Free tier |
| OpenAI GPT-4o | 1M tokens | ~$5 |
| Twilio | 1 hour calls | ~$0.85 |

**Typical cost per call**: $0.02 - $0.10 depending on duration and providers used.
