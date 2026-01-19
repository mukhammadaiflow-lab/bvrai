# Builder Voice AI Platform - Complete Deployment Guide

## Table of Contents
1. [Architecture Overview](#1-architecture-overview)
2. [Prerequisites](#2-prerequisites)
3. [Environment Setup](#3-environment-setup)
4. [Database Setup](#4-database-setup)
5. [Backend Deployment](#5-backend-deployment)
6. [Frontend Deployment](#6-frontend-deployment)
7. [Docker Compose (All-in-One)](#7-docker-compose-all-in-one)
8. [Production Deployment](#8-production-deployment)
9. [Post-Deployment Verification](#9-post-deployment-verification)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Architecture Overview

### What Gets Deployed

```
┌─────────────────────────────────────────────────────────────────┐
│                        INTERNET                                  │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     LOAD BALANCER / CDN                          │
│                    (CloudFlare / AWS ALB)                        │
└──────────────┬────────────────────────────────┬─────────────────┘
               │                                │
               ▼                                ▼
┌──────────────────────────┐      ┌──────────────────────────────┐
│      FRONTEND            │      │         BACKEND API           │
│   (Next.js 16.1.3)       │      │     (FastAPI + Python)        │
│                          │      │                                │
│   Port: 3000             │      │   Port: 8086                   │
│   Static + SSR           │◄────►│   REST API + WebSocket         │
└──────────────────────────┘      └──────────────┬────────────────┘
                                                 │
                    ┌────────────────────────────┼────────────────────────────┐
                    │                            │                            │
                    ▼                            ▼                            ▼
        ┌───────────────────┐      ┌───────────────────┐      ┌───────────────────┐
        │    PostgreSQL     │      │       Redis       │      │     RabbitMQ      │
        │                   │      │                   │      │                   │
        │   Port: 5432      │      │   Port: 6379      │      │   Port: 5672      │
        │   Main Database   │      │   Cache + Queue   │      │   Message Queue   │
        └───────────────────┘      └───────────────────┘      └───────────────────┘
```

### Service Dependencies

| Service | Depends On | Purpose |
|---------|------------|---------|
| Frontend | Backend API | Serves UI, calls API |
| Backend API | PostgreSQL, Redis | Handles all business logic |
| Workers | PostgreSQL, Redis, RabbitMQ | Background job processing |
| Voice Gateway | Backend API, Redis | Real-time call handling |

---

## 2. Prerequisites

### 2.1 Required Software

```bash
# Check versions (minimum requirements)
node --version    # v18.0.0 or higher
npm --version     # v9.0.0 or higher
python --version  # 3.11 or higher
docker --version  # 24.0.0 or higher
docker-compose --version  # 2.20.0 or higher
```

### 2.2 Required Accounts & API Keys

| Service | Required For | Get From |
|---------|--------------|----------|
| **Twilio** | Phone calls | https://www.twilio.com/console |
| **OpenAI** | GPT models | https://platform.openai.com/api-keys |
| **Deepgram** | Speech-to-Text | https://console.deepgram.com |
| **ElevenLabs** | Text-to-Speech | https://elevenlabs.io/api |

### 2.3 Server Requirements

**Minimum (Development/Testing):**
- CPU: 2 cores
- RAM: 4GB
- Storage: 20GB SSD
- OS: Ubuntu 22.04 / macOS / Windows WSL2

**Recommended (Production):**
- CPU: 4+ cores
- RAM: 8GB+
- Storage: 50GB+ SSD
- OS: Ubuntu 22.04 LTS

---

## 3. Environment Setup

### 3.1 Clone Repository

```bash
git clone https://github.com/mukhammadaiflow-lab/bvrai.git
cd bvrai
```

### 3.2 Create Environment File

```bash
# Copy example environment file
cp .env.example .env

# Edit with your values
nano .env
```

### 3.3 Environment Variables Explained

```bash
# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================
# PostgreSQL connection string
# Format: postgresql://username:password@host:port/database
DATABASE_URL=postgresql://bvrai:your_secure_password@localhost:5432/bvrai

# Connection pool settings (for production)
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=10

# =============================================================================
# REDIS CONFIGURATION
# =============================================================================
# Redis connection for caching and session storage
REDIS_URL=redis://localhost:6379/0

# =============================================================================
# RABBITMQ CONFIGURATION
# =============================================================================
# Message queue for background jobs
RABBITMQ_URL=amqp://bvrai:your_secure_password@localhost:5672/

# =============================================================================
# API CONFIGURATION
# =============================================================================
# Secret key for JWT tokens (generate with: openssl rand -hex 32)
SECRET_KEY=your_64_character_secret_key_here_generate_with_openssl

# API server settings
API_HOST=0.0.0.0
API_PORT=8086
API_WORKERS=4

# CORS allowed origins (comma-separated for production)
CORS_ORIGINS=http://localhost:3000,https://yourdomain.com

# =============================================================================
# FRONTEND CONFIGURATION
# =============================================================================
# URL where backend API is accessible
NEXT_PUBLIC_API_URL=http://localhost:8086

# =============================================================================
# TELEPHONY - TWILIO
# =============================================================================
# Get from: https://www.twilio.com/console
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_auth_token_here
TWILIO_PHONE_NUMBER=+1234567890

# =============================================================================
# AI PROVIDERS
# =============================================================================
# OpenAI (GPT-4, GPT-4-turbo)
# Get from: https://platform.openai.com/api-keys
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Anthropic (Claude) - Optional
# Get from: https://console.anthropic.com
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxx

# =============================================================================
# SPEECH PROVIDERS
# =============================================================================
# Deepgram (Speech-to-Text)
# Get from: https://console.deepgram.com
DEEPGRAM_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# ElevenLabs (Text-to-Speech)
# Get from: https://elevenlabs.io/api
ELEVENLABS_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# =============================================================================
# SECURITY
# =============================================================================
# Encryption key for sensitive data (generate with: openssl rand -hex 32)
ENCRYPTION_KEY=your_64_character_encryption_key_here

# =============================================================================
# OPTIONAL SERVICES
# =============================================================================
# Sentry for error tracking
SENTRY_DSN=https://xxx@xxx.ingest.sentry.io/xxx

# AWS S3 for file storage (recordings, etc.)
AWS_ACCESS_KEY_ID=AKIAXXXXXXXXXXXXXXXX
AWS_SECRET_ACCESS_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
AWS_S3_BUCKET=bvrai-recordings
AWS_REGION=us-east-1
```

### 3.4 Generate Secure Keys

```bash
# Generate SECRET_KEY
openssl rand -hex 32

# Generate ENCRYPTION_KEY
openssl rand -hex 32
```

---

## 4. Database Setup

### 4.1 Option A: Using Docker (Recommended)

```bash
# Start PostgreSQL container
docker run -d \
  --name bvrai-postgres \
  -e POSTGRES_USER=bvrai \
  -e POSTGRES_PASSWORD=your_secure_password \
  -e POSTGRES_DB=bvrai \
  -p 5432:5432 \
  -v bvrai_postgres_data:/var/lib/postgresql/data \
  postgres:16-alpine

# Start Redis container
docker run -d \
  --name bvrai-redis \
  -p 6379:6379 \
  -v bvrai_redis_data:/data \
  redis:7-alpine redis-server --appendonly yes

# Start RabbitMQ container
docker run -d \
  --name bvrai-rabbitmq \
  -e RABBITMQ_DEFAULT_USER=bvrai \
  -e RABBITMQ_DEFAULT_PASS=your_secure_password \
  -p 5672:5672 \
  -p 15672:15672 \
  rabbitmq:3.12-management-alpine
```

### 4.2 Option B: Local Installation

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install postgresql postgresql-contrib redis-server rabbitmq-server

# Create database and user
sudo -u postgres psql << EOF
CREATE USER bvrai WITH PASSWORD 'your_secure_password';
CREATE DATABASE bvrai OWNER bvrai;
GRANT ALL PRIVILEGES ON DATABASE bvrai TO bvrai;
EOF

# Start services
sudo systemctl start postgresql redis-server rabbitmq-server
sudo systemctl enable postgresql redis-server rabbitmq-server
```

### 4.3 Run Database Migrations

```bash
# Install Python dependencies first
cd /path/to/bvrai
pip install -r requirements.txt

# Run Alembic migrations
alembic upgrade head
```

### 4.4 Verify Database Connection

```bash
# Test PostgreSQL
psql postgresql://bvrai:your_secure_password@localhost:5432/bvrai -c "SELECT 1;"

# Test Redis
redis-cli ping  # Should return PONG

# Test RabbitMQ (management UI)
# Open http://localhost:15672 (user: bvrai, pass: your_secure_password)
```

---

## 5. Backend Deployment

### 5.1 Install Python Dependencies

```bash
cd /path/to/bvrai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 5.2 Start Backend API

**Development Mode:**
```bash
# Start with auto-reload
uvicorn platform.api.app:create_app --factory --reload --host 0.0.0.0 --port 8086
```

**Production Mode:**
```bash
# Start with Gunicorn + Uvicorn workers
gunicorn platform.api.app:create_app \
  --factory \
  --bind 0.0.0.0:8086 \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --timeout 120 \
  --keep-alive 5 \
  --access-logfile - \
  --error-logfile -
```

### 5.3 Start Background Workers

```bash
# In a separate terminal
source venv/bin/activate

# Start Celery worker
celery -A platform.worker worker --loglevel=info --concurrency=4

# In another terminal - Start Celery beat (scheduler)
celery -A platform.worker beat --loglevel=info
```

### 5.4 Verify Backend is Running

```bash
# Health check
curl http://localhost:8086/health

# API docs
open http://localhost:8086/docs
```

---

## 6. Frontend Deployment

### 6.1 Install Node.js Dependencies

```bash
cd /path/to/bvrai/frontend

# Install dependencies
npm install
```

### 6.2 Configure Frontend Environment

```bash
# Create .env.local file
cat > .env.local << EOF
NEXT_PUBLIC_API_URL=http://localhost:8086
EOF
```

### 6.3 Start Frontend

**Development Mode:**
```bash
npm run dev
# Open http://localhost:3000
```

**Production Mode:**
```bash
# Build production bundle
npm run build

# Start production server
npm run start
# Or with PM2:
pm2 start npm --name "bvrai-frontend" -- start
```

### 6.4 Verify Frontend is Running

```bash
# Open in browser
open http://localhost:3000

# Check it can reach API
curl http://localhost:3000/api/v1/health
```

---

## 7. Docker Compose (All-in-One)

### 7.1 Start Everything with One Command

```bash
cd /path/to/bvrai

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop everything
docker-compose down
```

### 7.2 What Docker Compose Starts

| Service | Port | URL |
|---------|------|-----|
| Frontend | 3000 | http://localhost:3000 |
| Backend API | 8086 | http://localhost:8086 |
| PostgreSQL | 5432 | localhost:5432 |
| Redis | 6379 | localhost:6379 |
| RabbitMQ | 5672, 15672 | http://localhost:15672 |
| Prometheus | 9090 | http://localhost:9090 |
| Grafana | 3001 | http://localhost:3001 |

### 7.3 Docker Compose Commands

```bash
# Start in background
docker-compose up -d

# View running containers
docker-compose ps

# View logs for specific service
docker-compose logs -f api
docker-compose logs -f frontend

# Restart a specific service
docker-compose restart api

# Stop everything
docker-compose down

# Stop and remove volumes (DELETES DATA!)
docker-compose down -v

# Rebuild after code changes
docker-compose build --no-cache
docker-compose up -d
```

---

## 8. Production Deployment

### 8.1 Cloud Platform Options

| Platform | Best For | Estimated Cost |
|----------|----------|----------------|
| **DigitalOcean** | Simple, affordable | $24-96/month |
| **AWS** | Enterprise, scalable | $50-500/month |
| **Google Cloud** | AI/ML focused | $50-500/month |
| **Vercel + Railway** | Serverless, easy | $20-100/month |

### 8.2 DigitalOcean Deployment (Recommended for Start)

**Step 1: Create Droplet**
```bash
# Create Ubuntu 22.04 droplet
# Recommended: 4GB RAM, 2 vCPU ($24/month)
```

**Step 2: Initial Server Setup**
```bash
# SSH into server
ssh root@your_server_ip

# Update system
apt update && apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com | sh
usermod -aG docker $USER

# Install Docker Compose
apt install docker-compose-plugin

# Install Nginx (reverse proxy)
apt install nginx certbot python3-certbot-nginx
```

**Step 3: Clone and Configure**
```bash
# Clone repository
git clone https://github.com/mukhammadaiflow-lab/bvrai.git /opt/bvrai
cd /opt/bvrai

# Create production environment file
cp .env.example .env
nano .env  # Edit with production values
```

**Step 4: Configure Nginx**
```nginx
# /etc/nginx/sites-available/bvrai
server {
    listen 80;
    server_name yourdomain.com;

    # Frontend
    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # Backend API
    location /api/ {
        proxy_pass http://localhost:8086;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket for real-time
    location /ws/ {
        proxy_pass http://localhost:8086;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

**Step 5: Enable HTTPS**
```bash
# Enable site
ln -s /etc/nginx/sites-available/bvrai /etc/nginx/sites-enabled/
nginx -t
systemctl reload nginx

# Get SSL certificate
certbot --nginx -d yourdomain.com
```

**Step 6: Start Services**
```bash
cd /opt/bvrai
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### 8.3 Environment Variables for Production

```bash
# Update these for production
CORS_ORIGINS=https://yourdomain.com
NEXT_PUBLIC_API_URL=https://yourdomain.com
SECRET_KEY=<generate_new_64_char_key>
ENCRYPTION_KEY=<generate_new_64_char_key>
DATABASE_URL=postgresql://bvrai:strong_password@db:5432/bvrai
```

### 8.4 Setup Systemd Service (Alternative to Docker)

```bash
# /etc/systemd/system/bvrai-api.service
[Unit]
Description=Builder Voice AI API
After=network.target postgresql.service redis.service

[Service]
Type=simple
User=bvrai
WorkingDirectory=/opt/bvrai
Environment="PATH=/opt/bvrai/venv/bin"
EnvironmentFile=/opt/bvrai/.env
ExecStart=/opt/bvrai/venv/bin/gunicorn platform.api.app:create_app --factory --bind 0.0.0.0:8086 --workers 4 --worker-class uvicorn.workers.UvicornWorker
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start
systemctl daemon-reload
systemctl enable bvrai-api
systemctl start bvrai-api
```

---

## 9. Post-Deployment Verification

### 9.1 Health Check Endpoints

```bash
# Backend API health
curl https://yourdomain.com/api/v1/health
# Expected: {"status": "healthy", "timestamp": "..."}

# Database connection
curl https://yourdomain.com/api/v1/health/db
# Expected: {"status": "connected"}

# Redis connection
curl https://yourdomain.com/api/v1/health/redis
# Expected: {"status": "connected"}
```

### 9.2 Functional Tests

```bash
# 1. Test user registration
curl -X POST https://yourdomain.com/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "password": "SecurePass123!", "name": "Test User"}'

# 2. Test login
curl -X POST https://yourdomain.com/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "password": "SecurePass123!"}'

# 3. Test agent creation (with token from login)
curl -X POST https://yourdomain.com/api/v1/agents \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"name": "Test Agent", "system_prompt": "You are a helpful assistant."}'
```

### 9.3 Monitoring Checklist

- [ ] Frontend loads at https://yourdomain.com
- [ ] Can log in / register
- [ ] Dashboard shows (with mock data initially)
- [ ] API docs accessible at https://yourdomain.com/api/docs
- [ ] No errors in browser console
- [ ] No errors in `docker-compose logs`

---

## 10. Troubleshooting

### 10.1 Common Issues

**Issue: Frontend can't reach backend**
```bash
# Check CORS configuration
echo $CORS_ORIGINS

# Check API is running
curl http://localhost:8086/health

# Check Nginx proxy
nginx -t
tail -f /var/log/nginx/error.log
```

**Issue: Database connection refused**
```bash
# Check PostgreSQL is running
docker-compose ps db
# or
systemctl status postgresql

# Check connection string
psql $DATABASE_URL -c "SELECT 1;"
```

**Issue: Redis connection error**
```bash
# Check Redis is running
docker-compose ps redis
# or
redis-cli ping

# Check Redis URL
redis-cli -u $REDIS_URL ping
```

**Issue: Build fails**
```bash
# Clear caches and rebuild
cd frontend
rm -rf .next node_modules
npm install
npm run build

# Check TypeScript errors
npx tsc --noEmit
```

### 10.2 Log Locations

| Service | Log Command |
|---------|-------------|
| Frontend | `docker-compose logs frontend` |
| Backend | `docker-compose logs api` |
| PostgreSQL | `docker-compose logs db` |
| Nginx | `tail -f /var/log/nginx/error.log` |
| System | `journalctl -u bvrai-api -f` |

### 10.3 Restart Services

```bash
# Docker Compose
docker-compose restart api
docker-compose restart frontend

# Systemd
systemctl restart bvrai-api
systemctl restart nginx
```

---

## Quick Start Commands Summary

```bash
# 1. Clone
git clone https://github.com/mukhammadaiflow-lab/bvrai.git
cd bvrai

# 2. Configure
cp .env.example .env
nano .env  # Edit values

# 3. Start (Docker)
docker-compose up -d

# 4. Verify
curl http://localhost:8086/health
open http://localhost:3000

# Done!
```

---

## Support

- Issues: https://github.com/mukhammadaiflow-lab/bvrai/issues
- Documentation: /docs folder in repository

---

**Last Updated:** January 2026
**Version:** 1.0.0
