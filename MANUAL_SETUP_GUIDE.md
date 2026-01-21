# BVRAI Platform - Manual Setup Guide

**IMPORTANT:** You must complete these steps before deploying to production.

---

## Table of Contents
1. [Prerequisites](#1-prerequisites)
2. [Environment Variables Setup](#2-environment-variables-setup)
3. [Database Setup](#3-database-setup)
4. [Redis Setup](#4-redis-setup)
5. [API Keys Registration](#5-api-keys-registration)
6. [Running Database Migrations](#6-running-database-migrations)
7. [Encryption Key Setup](#7-encryption-key-setup)
8. [Telephony Setup (Twilio)](#8-telephony-setup-twilio)
9. [Production Deployment](#9-production-deployment)
10. [Verification Checklist](#10-verification-checklist)

---

## 1. Prerequisites

### Required Software
```bash
# Check you have these installed:
python --version      # Requires Python 3.10+
node --version        # Requires Node.js 18+
docker --version      # Requires Docker 20+
docker-compose --version  # Requires docker-compose 2.0+
psql --version        # PostgreSQL client
redis-cli --version   # Redis client
```

### Install if missing (Ubuntu/Debian):
```bash
# Python 3.11
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip

# Node.js 18
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# PostgreSQL client
sudo apt install postgresql-client

# Redis CLI
sudo apt install redis-tools
```

---

## 2. Environment Variables Setup

### Step 1: Copy the example file
```bash
cd /home/user/bvrai
cp .env.example .env
```

### Step 2: Generate secure secrets
```bash
# Generate JWT secret (32+ characters)
openssl rand -hex 32
# Example output: a1b2c3d4e5f6789012345678901234567890abcdef12345678

# Generate APP secret key
openssl rand -hex 32

# Generate encryption key for PII (32 bytes, base64)
openssl rand -base64 32
# Example output: K7gNU3sdo+OL0wNhqoVWhr3g6s1xYv72ol/pe/Unols=
```

### Step 3: Edit .env file with your values
```bash
nano .env
```

**CRITICAL VARIABLES TO SET:**

```env
# ===== REQUIRED FOR PRODUCTION =====

# Application
APP_ENV=production
APP_DEBUG=false
APP_SECRET_KEY=<your-generated-32-char-secret>

# Database (change password!)
POSTGRES_PASSWORD=<strong-password-here>
DATABASE_URL=postgresql+asyncpg://bvrai:<password>@localhost:5432/bvrai

# Security
JWT_SECRET=<your-generated-32-char-secret>
ENCRYPTION_KEY=<your-base64-encryption-key>

# Redis
REDIS_URL=redis://localhost:6379/0

# CORS (your production domains)
CORS_ALLOWED_ORIGINS=https://app.yourdomain.com,https://api.yourdomain.com
ENVIRONMENT=production

# ===== AT LEAST ONE PROVIDER EACH =====

# STT Provider (pick one)
DEEPGRAM_API_KEY=<your-key>          # Recommended
# OR
OPENAI_API_KEY=<your-key>

# TTS Provider (pick one)
ELEVENLABS_API_KEY=<your-key>        # Recommended
# OR
CARTESIA_API_KEY=<your-key>
# OR
PLAYHT_API_KEY=<your-key>
PLAYHT_USER_ID=<your-user-id>

# LLM Provider (pick one)
OPENAI_API_KEY=<your-key>            # Recommended
# OR
ANTHROPIC_API_KEY=<your-key>
```

---

## 3. Database Setup

### Option A: Docker (Recommended for Development)
```bash
# Start PostgreSQL container
docker run -d \
  --name bvrai-postgres \
  -e POSTGRES_USER=bvrai \
  -e POSTGRES_PASSWORD=<your-password> \
  -e POSTGRES_DB=bvrai \
  -p 5432:5432 \
  -v bvrai_postgres_data:/var/lib/postgresql/data \
  postgres:15-alpine

# Verify it's running
docker ps | grep bvrai-postgres
```

### Option B: Native PostgreSQL
```bash
# Install PostgreSQL
sudo apt install postgresql postgresql-contrib

# Start service
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create user and database
sudo -u postgres psql << 'EOF'
CREATE USER bvrai WITH PASSWORD 'your-secure-password';
CREATE DATABASE bvrai OWNER bvrai;
GRANT ALL PRIVILEGES ON DATABASE bvrai TO bvrai;
\q
EOF

# Test connection
psql -U bvrai -h localhost -d bvrai -c "SELECT 1;"
```

### Option C: Cloud PostgreSQL (Production)
- **AWS RDS**: Create PostgreSQL 15 instance
- **Google Cloud SQL**: Create PostgreSQL instance
- **Supabase**: Free tier available

Update `.env` with connection string:
```env
DATABASE_URL=postgresql+asyncpg://user:password@your-host:5432/bvrai?sslmode=require
```

---

## 4. Redis Setup

### Option A: Docker (Recommended)
```bash
docker run -d \
  --name bvrai-redis \
  -p 6379:6379 \
  redis:7-alpine \
  redis-server --appendonly yes

# Verify
docker ps | grep bvrai-redis
redis-cli ping  # Should return PONG
```

### Option B: Native Redis
```bash
sudo apt install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Verify
redis-cli ping  # Should return PONG
```

### Option C: Cloud Redis
- **AWS ElastiCache**: Create Redis cluster
- **Redis Cloud**: Free tier available
- **Upstash**: Serverless Redis

Update `.env`:
```env
REDIS_URL=redis://user:password@your-redis-host:6379/0
```

---

## 5. API Keys Registration

### STT (Speech-to-Text) - Pick One

#### Deepgram (Recommended)
1. Go to https://console.deepgram.com/signup
2. Create account and verify email
3. Navigate to API Keys
4. Create new API key with all permissions
5. Copy key to `.env`:
   ```env
   DEEPGRAM_API_KEY=your_deepgram_key_here
   DEFAULT_STT_PROVIDER=deepgram
   ```

#### OpenAI Whisper
1. Go to https://platform.openai.com/signup
2. Add payment method (pay-as-you-go)
3. Create API key at https://platform.openai.com/api-keys
4. Copy to `.env`:
   ```env
   OPENAI_API_KEY=sk-...
   ```

---

### TTS (Text-to-Speech) - Pick One

#### ElevenLabs (Recommended)
1. Go to https://elevenlabs.io/sign-up
2. Choose plan (free tier available)
3. Go to Profile → API Keys
4. Generate and copy key:
   ```env
   ELEVENLABS_API_KEY=your_key_here
   DEFAULT_TTS_PROVIDER=elevenlabs
   ```

#### Cartesia (Ultra-low latency)
1. Go to https://cartesia.ai/
2. Request API access
3. Copy key:
   ```env
   CARTESIA_API_KEY=your_key_here
   DEFAULT_TTS_PROVIDER=cartesia
   ```

#### PlayHT
1. Go to https://play.ht/signup
2. Get API key from dashboard
3. Copy both key and user ID:
   ```env
   PLAYHT_API_KEY=your_key_here
   PLAYHT_USER_ID=your_user_id_here
   DEFAULT_TTS_PROVIDER=playht
   ```

---

### LLM (Language Model) - Pick One

#### OpenAI (Recommended)
1. Use same key from Whisper setup
2. Ensure you have GPT-4 access
   ```env
   DEFAULT_LLM_PROVIDER=openai
   DEFAULT_LLM_MODEL=gpt-4o
   ```

#### Anthropic (Claude)
1. Go to https://console.anthropic.com/
2. Create account and add payment
3. Generate API key
   ```env
   ANTHROPIC_API_KEY=sk-ant-...
   DEFAULT_LLM_PROVIDER=anthropic
   DEFAULT_LLM_MODEL=claude-3-opus-20240229
   ```

---

## 6. Running Database Migrations

### Step 1: Install Python dependencies
```bash
cd /home/user/bvrai

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Run migrations
```bash
# Set environment
export $(cat .env | grep -v '^#' | xargs)

# Run Alembic migrations
alembic upgrade head

# Verify tables created
psql $DATABASE_URL -c "\dt"
```

**Expected output:**
```
                   List of relations
 Schema |         Name          | Type  | Owner
--------+-----------------------+-------+-------
 public | agents                | table | bvrai
 public | alembic_version       | table | bvrai
 public | analytics_events      | table | bvrai
 public | api_keys              | table | bvrai
 public | calls                 | table | bvrai
 public | conversations         | table | bvrai
 public | messages              | table | bvrai
 public | organizations         | table | bvrai
 public | users                 | table | bvrai
 ...
```

---

## 7. Encryption Key Setup

The platform uses AES-256-GCM encryption for PII (phone numbers, etc.).

### Generate encryption key
```bash
# Generate 32-byte key in base64
python3 -c "import secrets, base64; print(base64.b64encode(secrets.token_bytes(32)).decode())"
```

### Add to .env
```env
ENCRYPTION_KEY=<your-base64-key>
```

### Backfill existing data (if upgrading)
If you have existing phone numbers that need encryption:
```bash
# This script will encrypt existing plaintext phone numbers
python3 scripts/backfill_encrypted_phones.py
```

---

## 8. Telephony Setup (Twilio)

### Step 1: Create Twilio account
1. Go to https://www.twilio.com/try-twilio
2. Verify phone number
3. Complete setup wizard

### Step 2: Get credentials
1. Go to Console Dashboard
2. Copy Account SID and Auth Token
3. Buy a phone number (Voice capable)

### Step 3: Add to .env
```env
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_auth_token_here
TWILIO_PHONE_NUMBER=+1234567890
```

### Step 4: Configure webhook URLs
For development (using ngrok):
```bash
# Install ngrok
curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc
echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list
sudo apt update && sudo apt install ngrok

# Start tunnel
ngrok http 8080
```

For production:
```env
TWILIO_WEBHOOK_URL=https://api.yourdomain.com
```

Configure in Twilio Console:
1. Go to Phone Numbers → Manage → Active numbers
2. Click your number
3. Under Voice Configuration:
   - "A call comes in" → Webhook
   - URL: `https://your-domain/api/twilio/voice`
   - HTTP POST

---

## 9. Production Deployment

### Using Docker Compose
```bash
# Build all services
docker-compose build

# Start services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### Health Check
```bash
# Check all services
curl http://localhost:8080/health  # Telephony
curl http://localhost:8082/health  # ASR
curl http://localhost:8083/health  # TTS
curl http://localhost:8084/health  # Conversation
curl http://localhost:8085/health  # AI Orchestrator
curl http://localhost:3000/health  # Platform API
```

---

## 10. Verification Checklist

Run these commands to verify your setup:

```bash
# 1. Environment variables loaded
echo $JWT_SECRET | head -c 10  # Should show first 10 chars

# 2. Database connection
psql $DATABASE_URL -c "SELECT COUNT(*) FROM organizations;"

# 3. Redis connection
redis-cli -u $REDIS_URL ping  # Should return PONG

# 4. API provider connectivity
curl -s https://api.deepgram.com/v1/projects \
  -H "Authorization: Token $DEEPGRAM_API_KEY" | head -c 100

# 5. Services running
docker-compose ps | grep -c "Up"  # Should match service count

# 6. Frontend build
cd frontend && npm run build  # Should complete without errors
```

---

## Troubleshooting

### Database connection refused
```bash
# Check PostgreSQL is running
docker ps | grep postgres
# OR
sudo systemctl status postgresql

# Check connection manually
psql -U bvrai -h localhost -d bvrai
```

### Redis connection refused
```bash
# Check Redis is running
docker ps | grep redis
# OR
sudo systemctl status redis-server

# Test connection
redis-cli -h localhost -p 6379 ping
```

### API key not working
```bash
# Test Deepgram
curl https://api.deepgram.com/v1/projects \
  -H "Authorization: Token $DEEPGRAM_API_KEY"

# Test ElevenLabs
curl https://api.elevenlabs.io/v1/user \
  -H "xi-api-key: $ELEVENLABS_API_KEY"

# Test OpenAI
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

### Migration failed
```bash
# Check current migration state
alembic current

# Show migration history
alembic history

# Rollback if needed
alembic downgrade -1

# Re-run
alembic upgrade head
```

---

## Next Steps

After completing setup:

1. **Start the platform**:
   ```bash
   docker-compose up -d
   ```

2. **Access the dashboard**:
   Open http://localhost:3000 in your browser

3. **Create first organization**:
   Use the API or dashboard to create your organization

4. **Create first agent**:
   Navigate to Agents → Create Agent

5. **Test a call**:
   Call your Twilio number to test the voice agent

---

## Support

- GitHub Issues: https://github.com/your-org/bvrai/issues
- Documentation: https://docs.bvrai.com
