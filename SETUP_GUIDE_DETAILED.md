# BVRAI Voice AI Platform - Complete Setup Guide

This guide provides **detailed step-by-step instructions** for setting up the BVRAI Voice AI Platform for production use. Each section includes explanations of **why** each step is necessary and **what** each configuration does.

---

## Table of Contents

1. [Understanding the Architecture](#1-understanding-the-architecture)
2. [Prerequisites Installation](#2-prerequisites-installation)
3. [Database Setup (PostgreSQL)](#3-database-setup-postgresql)
4. [Cache Setup (Redis)](#4-cache-setup-redis)
5. [Environment Variables Configuration](#5-environment-variables-configuration)
6. [API Provider Registration](#6-api-provider-registration)
7. [Encryption and Security Setup](#7-encryption-and-security-setup)
8. [Telephony Setup (Twilio)](#8-telephony-setup-twilio)
9. [Running Database Migrations](#9-running-database-migrations)
10. [Starting the Application](#10-starting-the-application)
11. [Production Deployment](#11-production-deployment)
12. [Verification Checklist](#12-verification-checklist)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Understanding the Architecture

Before you begin, understand what components make up the BVRAI platform:

### Backend Services
| Service | Purpose | Port |
|---------|---------|------|
| `platform-api` | Main REST API for frontend, user management, billing | 8000 |
| `voice-gateway` | Handles real-time voice calls via WebSocket | 8001 |
| `voice-pipeline` | Processes audio (STT, TTS, NLU) | 8002 |
| `voice-lab` | Voice cloning and custom voice management | 8003 |
| `telephony-bridge` | Connects to Twilio for phone calls | 8004 |
| `knowledge-api` | RAG (Retrieval Augmented Generation) service | 8005 |

### External Dependencies
- **PostgreSQL**: Primary database for all data storage
- **Redis**: Caching, session management, rate limiting, real-time pub/sub
- **AI Providers**: OpenAI, Anthropic, Deepgram, ElevenLabs, etc.
- **Twilio**: Phone number provisioning and call routing

### Frontend
- **Next.js 14** application running on port 3000
- Connects to `platform-api` for all operations

---

## 2. Prerequisites Installation

### 2.1 Install Python 3.11+

**Why:** The backend requires Python 3.11 for better async performance and typing features.

```bash
# On Ubuntu/Debian
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip

# On macOS (with Homebrew)
brew install python@3.11

# Verify installation
python3.11 --version  # Should show Python 3.11.x
```

### 2.2 Install Node.js 18+

**Why:** The frontend uses Next.js 14 which requires Node.js 18+.

```bash
# Using nvm (recommended)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.5/install.sh | bash
source ~/.bashrc  # or ~/.zshrc
nvm install 18
nvm use 18

# Verify
node --version  # Should show v18.x.x
npm --version   # Should show 9.x.x or 10.x.x
```

### 2.3 Install Docker (Optional but Recommended)

**Why:** Docker simplifies running PostgreSQL and Redis without manual installation.

```bash
# On Ubuntu
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
# Log out and back in for group changes to take effect

# On macOS
# Download Docker Desktop from https://www.docker.com/products/docker-desktop/

# Verify
docker --version
docker-compose --version
```

### 2.4 Install Poetry (Python Dependency Manager)

**Why:** Poetry manages Python dependencies with locked versions for reproducibility.

```bash
curl -sSL https://install.python-poetry.org | python3 -

# Add to PATH (add this to ~/.bashrc or ~/.zshrc)
export PATH="$HOME/.local/bin:$PATH"

# Verify
poetry --version
```

---

## 3. Database Setup (PostgreSQL)

### Option A: Using Docker (Recommended for Development)

**Why:** Isolated database that can be easily reset without affecting your system.

```bash
# Create a persistent volume for data
docker volume create bvrai_postgres_data

# Run PostgreSQL container
docker run -d \
  --name bvrai-postgres \
  -e POSTGRES_USER=bvrai \
  -e POSTGRES_PASSWORD=your_secure_password_here \
  -e POSTGRES_DB=bvrai \
  -p 5432:5432 \
  -v bvrai_postgres_data:/var/lib/postgresql/data \
  --restart unless-stopped \
  postgres:15-alpine

# Verify it's running
docker ps | grep bvrai-postgres

# Test connection
docker exec -it bvrai-postgres psql -U bvrai -d bvrai -c "SELECT 1;"
```

**Understanding the flags:**
- `-d`: Run in background (detached)
- `--name`: Container name for easy reference
- `-e POSTGRES_*`: Environment variables for initial setup
- `-p 5432:5432`: Map container port to host port
- `-v`: Persist data outside container
- `--restart unless-stopped`: Auto-restart on system reboot

### Option B: Native Installation

**Why:** Better performance in production, direct access to PostgreSQL tools.

```bash
# Ubuntu/Debian
sudo apt install -y postgresql-15 postgresql-contrib-15

# Start service
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create user and database
sudo -u postgres psql << EOF
CREATE USER bvrai WITH PASSWORD 'your_secure_password_here';
CREATE DATABASE bvrai OWNER bvrai;
GRANT ALL PRIVILEGES ON DATABASE bvrai TO bvrai;

-- Enable required extensions
\c bvrai
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";  -- For UUID generation
CREATE EXTENSION IF NOT EXISTS "pgcrypto";    -- For encryption functions
CREATE EXTENSION IF NOT EXISTS "pg_trgm";     -- For fuzzy text search
EOF
```

### Option C: Cloud PostgreSQL (Production)

**Why:** Managed service with automatic backups, scaling, and high availability.

**AWS RDS Setup:**
1. Go to AWS Console → RDS → Create database
2. Choose PostgreSQL 15
3. Select "Production" template
4. Settings:
   - DB instance identifier: `bvrai-prod`
   - Master username: `bvrai_admin`
   - Master password: Generate a secure one
5. Instance configuration: Start with `db.t3.medium` ($50/month)
6. Storage: 100GB GP3 with autoscaling
7. Connectivity: Create new VPC security group
8. Additional configuration:
   - Initial database name: `bvrai`
   - Enable automated backups (7 days retention)
   - Enable encryption

**Connection String:**
```
postgresql://bvrai_admin:PASSWORD@your-instance.xxxxx.us-east-1.rds.amazonaws.com:5432/bvrai
```

---

## 4. Cache Setup (Redis)

### Option A: Using Docker (Recommended for Development)

```bash
# Create persistent volume
docker volume create bvrai_redis_data

# Run Redis with persistence
docker run -d \
  --name bvrai-redis \
  -p 6379:6379 \
  -v bvrai_redis_data:/data \
  --restart unless-stopped \
  redis:7-alpine \
  redis-server --appendonly yes --requirepass your_redis_password

# Verify
docker exec bvrai-redis redis-cli -a your_redis_password PING
# Should return: PONG
```

**Understanding the options:**
- `--appendonly yes`: Persist data to disk (AOF persistence)
- `--requirepass`: Require password for connections (security)

### Option B: Native Installation

```bash
# Ubuntu/Debian
sudo apt install -y redis-server

# Configure Redis
sudo nano /etc/redis/redis.conf
```

Find and modify these lines:
```conf
# Enable password authentication
requirepass your_redis_password

# Bind to localhost only (for security)
bind 127.0.0.1

# Enable persistence
appendonly yes

# Set memory limit (adjust based on your RAM)
maxmemory 512mb
maxmemory-policy allkeys-lru
```

```bash
# Restart Redis
sudo systemctl restart redis-server
sudo systemctl enable redis-server

# Test
redis-cli -a your_redis_password PING
```

### Option C: Cloud Redis (Production)

**AWS ElastiCache:**
1. Go to ElastiCache → Redis → Create cluster
2. Cluster mode: Disabled (simpler for start)
3. Node type: `cache.t3.medium` (~$25/month)
4. Number of replicas: 1 (for high availability)
5. Enable encryption at rest and in transit
6. Security group: Allow port 6379 from your VPC

---

## 5. Environment Variables Configuration

### 5.1 Create the .env file

**Why:** Environment variables keep secrets out of code and allow different configs per environment.

```bash
# Navigate to project root
cd /path/to/bvrai

# Create .env file
cp .env.example .env
nano .env  # or use your preferred editor
```

### 5.2 Complete Environment Variables with Explanations

```bash
# ============================================
# ENVIRONMENT SETTINGS
# ============================================
# Options: development, staging, production
# This affects logging levels, debug features, and CORS settings
ENVIRONMENT=production

# Secret key for JWT tokens and session encryption
# CRITICAL: Generate a unique 64-character hex string
# Generate with: openssl rand -hex 32
SECRET_KEY=your_64_character_hex_secret_key_here

# ============================================
# DATABASE CONFIGURATION
# ============================================
# PostgreSQL connection string
# Format: postgresql://USER:PASSWORD@HOST:PORT/DATABASE
DATABASE_URL=postgresql://bvrai:your_password@localhost:5432/bvrai

# Connection pool settings (for production)
DATABASE_POOL_SIZE=20          # Max connections in pool
DATABASE_MAX_OVERFLOW=10       # Extra connections if pool exhausted
DATABASE_POOL_RECYCLE=300      # Recycle connections after 5 minutes

# ============================================
# REDIS CONFIGURATION
# ============================================
# Redis connection string
# Format: redis://:PASSWORD@HOST:PORT/DB_NUMBER
REDIS_URL=redis://:your_redis_password@localhost:6379/0

# Rate limiting settings (requests per time window)
RATE_LIMIT_PER_MINUTE=100      # API requests per minute
RATE_LIMIT_PER_HOUR=2000       # API requests per hour

# ============================================
# AI/LLM PROVIDERS
# ============================================
# OpenAI - For GPT-4, Whisper (STT), TTS
# Get key: https://platform.openai.com/api-keys
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Anthropic - For Claude models
# Get key: https://console.anthropic.com/
ANTHROPIC_API_KEY=sk-ant-api03-xxxxxxxxxxxxxxxxxxxxxxxx

# Default LLM settings
DEFAULT_LLM_PROVIDER=openai    # Options: openai, anthropic
DEFAULT_LLM_MODEL=gpt-4-turbo  # Model to use for conversations

# ============================================
# SPEECH-TO-TEXT (STT) PROVIDERS
# ============================================
# Deepgram - Real-time transcription (recommended)
# Get key: https://console.deepgram.com/
DEEPGRAM_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# AssemblyAI - Alternative STT
# Get key: https://www.assemblyai.com/dashboard
ASSEMBLYAI_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Default STT settings
DEFAULT_STT_PROVIDER=deepgram  # Options: deepgram, assemblyai, openai
DEFAULT_STT_MODEL=nova-2       # Deepgram model

# ============================================
# TEXT-TO-SPEECH (TTS) PROVIDERS
# ============================================
# ElevenLabs - High-quality voices with cloning
# Get key: https://elevenlabs.io/
ELEVENLABS_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# PlayHT - Alternative TTS with voice cloning
# Get key: https://play.ht/studio/api-access
PLAYHT_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
PLAYHT_USER_ID=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Cartesia - Ultra-low latency TTS
# Get key: https://cartesia.ai/
CARTESIA_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Default TTS settings
DEFAULT_TTS_PROVIDER=elevenlabs  # Options: elevenlabs, playht, cartesia, openai
DEFAULT_VOICE_ID=rachel          # Default voice ID

# ============================================
# TELEPHONY (TWILIO)
# ============================================
# Get credentials: https://console.twilio.com/
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_PHONE_NUMBER=+15551234567  # Your Twilio phone number

# Twilio TwiML App SID (for voice webhooks)
TWILIO_TWIML_APP_SID=APxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# ============================================
# ENCRYPTION (PII PROTECTION)
# ============================================
# AES-256 encryption key for PII data (phone numbers, etc.)
# Generate with: openssl rand -hex 32
PII_ENCRYPTION_KEY=your_32_byte_hex_encryption_key_here

# Fields to encrypt (comma-separated)
PII_ENCRYPTED_FIELDS=phone_number,caller_id,customer_name

# ============================================
# APPLICATION URLS
# ============================================
# Frontend URL (for CORS and redirects)
FRONTEND_URL=https://app.yourdomain.com

# API URL (for webhooks and callbacks)
API_URL=https://api.yourdomain.com

# WebSocket URL (for real-time connections)
WEBSOCKET_URL=wss://api.yourdomain.com

# ============================================
# EMAIL CONFIGURATION (Optional)
# ============================================
# For sending notifications and invitations
SMTP_HOST=smtp.sendgrid.net
SMTP_PORT=587
SMTP_USER=apikey
SMTP_PASSWORD=SG.xxxxxxxxxxxxxxxxxxxxxxxx
EMAIL_FROM=noreply@yourdomain.com

# ============================================
# MONITORING (Optional but Recommended)
# ============================================
# Sentry for error tracking
# Get DSN: https://sentry.io/
SENTRY_DSN=https://xxxxxxxx@sentry.io/xxxxx

# ============================================
# STORAGE (for recordings, transcripts)
# ============================================
# AWS S3 or compatible storage
AWS_ACCESS_KEY_ID=AKIAxxxxxxxxxxxxxxxx
AWS_SECRET_ACCESS_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
AWS_S3_BUCKET=bvrai-storage
AWS_S3_REGION=us-east-1

# ============================================
# FEATURE FLAGS
# ============================================
ENABLE_VOICE_CLONING=true
ENABLE_KNOWLEDGE_BASE=true
ENABLE_CALL_RECORDING=true
ENABLE_ANALYTICS=true
```

### 5.3 Security Best Practices for .env

```bash
# Set proper permissions (owner read/write only)
chmod 600 .env

# Never commit .env to git
echo ".env" >> .gitignore

# For production, use secret management:
# - AWS Secrets Manager
# - HashiCorp Vault
# - Kubernetes Secrets
```

---

## 6. API Provider Registration

### 6.1 OpenAI Setup

**What it provides:** GPT-4 for conversations, Whisper for STT, TTS voices

1. Go to https://platform.openai.com/
2. Click "Sign Up" or "Log In"
3. Navigate to "API Keys" section
4. Click "Create new secret key"
5. Give it a name: `bvrai-production`
6. Copy the key (starts with `sk-proj-`)

**Billing Setup:**
1. Go to "Billing" → "Payment methods"
2. Add a credit card
3. Set usage limits:
   - Hard limit: $500/month (prevents unexpected charges)
   - Soft limit: $400/month (sends warning email)

**Understanding Costs:**
- GPT-4-turbo: ~$0.01 per 1K input tokens, ~$0.03 per 1K output tokens
- Whisper (STT): ~$0.006 per minute
- TTS: ~$0.015 per 1K characters

### 6.2 Deepgram Setup

**What it provides:** Real-time speech-to-text with low latency

1. Go to https://console.deepgram.com/
2. Sign up or log in
3. Go to "API Keys"
4. Create new API key with permissions:
   - Transcription: Read
   - Usage: Read (for monitoring)
5. Copy the key

**Understanding Models:**
- `nova-2`: Best accuracy, recommended for production
- `nova`: Good balance of speed and accuracy
- `base`: Fastest, lower accuracy

**Pricing:** ~$0.0043 per minute of audio

### 6.3 ElevenLabs Setup

**What it provides:** High-quality text-to-speech with voice cloning

1. Go to https://elevenlabs.io/
2. Sign up or log in
3. Navigate to "Profile" → "API"
4. Copy your API key

**Voice Selection:**
1. Go to "Voice Library"
2. Browse available voices
3. Click on a voice to get its ID
4. Add voices to your "Voice Lab" for customization

**Voice Cloning (if enabled):**
1. Go to "Voice Lab"
2. Click "Add Generative or Cloned Voice"
3. Upload 1-5 minutes of clear audio
4. The cloned voice will get a unique ID

**Pricing:**
- Starter: $5/month for 30K characters
- Creator: $22/month for 100K characters
- Pro: $99/month for 500K characters

### 6.4 Twilio Setup (See Section 8 for detailed instructions)

---

## 7. Encryption and Security Setup

### 7.1 Generate Encryption Keys

**Why:** PII (Personally Identifiable Information) like phone numbers must be encrypted at rest.

```bash
# Generate SECRET_KEY (for JWT and sessions)
openssl rand -hex 32
# Example output: 8f7d3e2a1b9c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e

# Generate PII_ENCRYPTION_KEY (for AES-256)
openssl rand -hex 32
# Example output: a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2
```

**IMPORTANT:**
- Never share these keys
- Store them securely (password manager, secrets manager)
- If compromised, rotate immediately and re-encrypt data

### 7.2 Understanding What Gets Encrypted

The system encrypts these fields using AES-256-GCM:
- `phone_number`: Caller and callee phone numbers
- `caller_id`: Caller identification
- `customer_name`: Customer names in call records
- `transcript_text`: Call transcripts (optional)

### 7.3 SSL/TLS Certificates (Production)

**Why:** All traffic must be encrypted in transit.

**Using Let's Encrypt (Free):**
```bash
# Install Certbot
sudo apt install certbot

# Get certificate for your domain
sudo certbot certonly --standalone -d api.yourdomain.com -d app.yourdomain.com

# Certificates are saved at:
# /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem
# /etc/letsencrypt/live/api.yourdomain.com/privkey.pem

# Auto-renewal is set up automatically
# Test renewal: sudo certbot renew --dry-run
```

---

## 8. Telephony Setup (Twilio)

### 8.1 Create Twilio Account

1. Go to https://www.twilio.com/try-twilio
2. Sign up with email
3. Verify your phone number
4. Complete account setup

### 8.2 Get Account Credentials

1. In Twilio Console, find your credentials in the dashboard:
   - **Account SID**: Starts with `AC...` (this is your account identifier)
   - **Auth Token**: Click "Show" to reveal (this is your secret key)

### 8.3 Buy a Phone Number

1. Go to Console → Phone Numbers → Buy a number
2. Search criteria:
   - Country: Select your target country
   - Capabilities: Voice (required), SMS (optional)
   - Type: Local (cheaper) or Toll-Free (professional)
3. Click "Buy" on your chosen number
4. Cost: ~$1-2/month for local US numbers

### 8.4 Create TwiML Application

**Why:** TwiML apps route incoming calls to your webhook.

1. Go to Console → Voice → TwiML Apps
2. Click "Create new TwiML App"
3. Fill in:
   - **Friendly Name**: `BVRAI Voice App`
   - **Voice Request URL**: `https://api.yourdomain.com/api/v1/telephony/twilio/voice`
   - **Voice Request Method**: POST
   - **Voice Status Callback URL**: `https://api.yourdomain.com/api/v1/telephony/twilio/status`
4. Click "Create"
5. Copy the **SID** (starts with `AP...`)

### 8.5 Configure Phone Number to Use TwiML App

1. Go to Console → Phone Numbers → Active Numbers
2. Click on your phone number
3. Under "Voice & Fax":
   - Configure With: TwiML App
   - TwiML App: Select "BVRAI Voice App"
4. Click "Save"

### 8.6 Understanding Twilio Webhooks

When a call comes in:
1. Twilio sends POST to your `/voice` webhook
2. Your server returns TwiML XML instructions
3. Twilio executes those instructions
4. Status updates are sent to `/status` webhook

**Example incoming call flow:**
```
Caller → Twilio → Your Voice Webhook → TwiML Response → Twilio → Call Connected
```

---

## 9. Running Database Migrations

### 9.1 Install Python Dependencies

```bash
# Navigate to project
cd /path/to/bvrai

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install poetry
poetry install
```

### 9.2 Run Alembic Migrations

**What migrations do:** Create all database tables, indexes, and constraints.

```bash
# Ensure .env is configured with DATABASE_URL
source venv/bin/activate

# Check current migration status
alembic current

# Run all pending migrations
alembic upgrade head

# Verify tables were created
psql $DATABASE_URL -c "\dt"
```

**Expected tables:**
- `users` - User accounts
- `organizations` - Organization/tenant data
- `agents` - Voice agent configurations
- `calls` - Call records and metadata
- `transcripts` - Call transcripts
- `webhooks` - Webhook configurations
- `api_keys` - API key management
- And more...

### 9.3 Seed Initial Data (Optional)

```bash
# Create initial admin user
python -m scripts.seed_admin \
  --email admin@yourdomain.com \
  --password YourSecurePassword123! \
  --organization "Your Company"
```

---

## 10. Starting the Application

### 10.1 Development Mode

```bash
# Terminal 1: Start backend
cd /path/to/bvrai
source venv/bin/activate
uvicorn services.platform-api.app.main:app --reload --port 8000

# Terminal 2: Start frontend
cd /path/to/bvrai/frontend
npm install
npm run dev
```

### 10.2 Production Mode with Docker Compose

```bash
# Build and start all services
docker-compose up -d --build

# View logs
docker-compose logs -f

# Check service health
docker-compose ps
```

### 10.3 Verify Services Are Running

```bash
# Check API health
curl http://localhost:8000/health
# Expected: {"status": "healthy", "version": "1.0.0"}

# Check frontend
curl http://localhost:3000
# Expected: HTML content
```

---

## 11. Production Deployment

### 11.1 Server Requirements

**Minimum (for testing):**
- 2 vCPU
- 4 GB RAM
- 50 GB SSD

**Recommended (for production):**
- 4 vCPU
- 8 GB RAM
- 100 GB SSD
- Load balancer for high availability

### 11.2 Deploy with Docker

```bash
# On your production server
git clone https://github.com/your-org/bvrai.git
cd bvrai

# Configure production environment
cp .env.example .env
nano .env  # Configure with production values

# Build and run
docker-compose -f docker-compose.prod.yml up -d --build

# Set up Nginx reverse proxy (see below)
```

### 11.3 Nginx Configuration

```nginx
# /etc/nginx/sites-available/bvrai
server {
    listen 80;
    server_name api.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.yourdomain.com/privkey.pem;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

```bash
# Enable the site
sudo ln -s /etc/nginx/sites-available/bvrai /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## 12. Verification Checklist

Run through this checklist to ensure everything is working:

### Database
- [ ] PostgreSQL is running: `docker ps | grep postgres`
- [ ] Can connect: `psql $DATABASE_URL -c "SELECT 1"`
- [ ] Tables exist: `psql $DATABASE_URL -c "\dt"`

### Redis
- [ ] Redis is running: `docker ps | grep redis`
- [ ] Can connect: `redis-cli -a password PING`

### API
- [ ] Health endpoint: `curl https://api.yourdomain.com/health`
- [ ] Can authenticate: Test login endpoint
- [ ] CORS working: Test from frontend

### Frontend
- [ ] Loads correctly: Visit `https://app.yourdomain.com`
- [ ] Can log in: Use test credentials
- [ ] API calls work: Check browser network tab

### Telephony
- [ ] Twilio webhook configured
- [ ] Test call connects: Call your Twilio number
- [ ] Call audio works: Verify two-way audio

### Voice Pipeline
- [ ] STT working: Test transcription
- [ ] TTS working: Test voice synthesis
- [ ] LLM responding: Test conversation

---

## 13. Troubleshooting

### Common Issues

**Database connection refused:**
```bash
# Check if PostgreSQL is running
docker ps | grep postgres

# Check if port is accessible
nc -zv localhost 5432

# Check DATABASE_URL format
echo $DATABASE_URL
```

**Redis connection error:**
```bash
# Verify Redis is running
docker ps | grep redis

# Test with password
redis-cli -a your_password PING
```

**API returns 500 errors:**
```bash
# Check API logs
docker-compose logs platform-api

# Common causes:
# - Missing environment variables
# - Database not accessible
# - Invalid API keys
```

**Twilio calls not connecting:**
```bash
# Verify webhook URL is accessible
curl -X POST https://api.yourdomain.com/api/v1/telephony/twilio/voice

# Check Twilio debugger
# https://console.twilio.com/debugger
```

**Voice quality issues:**
- Check network latency to provider endpoints
- Verify audio codec settings
- Monitor CPU usage during calls

---

## Support

If you encounter issues not covered here:

1. Check the logs: `docker-compose logs -f [service]`
2. Review the error in Sentry (if configured)
3. Check provider dashboards (Twilio, OpenAI, etc.) for rate limits or errors
4. Contact support at support@bvrai.com
