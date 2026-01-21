# BVRAI - Railway Deployment Guide

Complete guide for deploying the BVRAI Voice AI Platform to Railway.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Service Architecture](#service-architecture)
5. [Step-by-Step Deployment](#step-by-step-deployment)
6. [Environment Variables](#environment-variables)
7. [Database Setup](#database-setup)
8. [Connecting Services](#connecting-services)
9. [Custom Domains](#custom-domains)
10. [Monitoring & Logs](#monitoring--logs)
11. [Troubleshooting](#troubleshooting)

---

## Overview

BVRAI can be deployed as a multi-service application on Railway. The deployment consists of:

- **Backend API** - Python FastAPI application
- **Frontend** - Next.js React application
- **PostgreSQL** - Primary database
- **Redis** - Caching and session management

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Railway Project                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Frontend   │───▶│  Backend API │───▶│  PostgreSQL  │  │
│  │   (Next.js)  │    │  (FastAPI)   │    │   Database   │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                               │
│         │                   ▼                               │
│         │            ┌──────────────┐                       │
│         └───────────▶│    Redis     │                       │
│                      │    Cache     │                       │
│                      └──────────────┘                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

1. **Railway Account**: Sign up at [railway.app](https://railway.app)
2. **Railway CLI** (optional but recommended):
   ```bash
   npm install -g @railway/cli
   railway login
   ```
3. **GitHub Account**: For connecting your repository

---

## Quick Start

### Option 1: Deploy via Railway CLI

```bash
# Clone the repository
git clone https://github.com/yourusername/bvrai.git
cd bvrai

# Login to Railway
railway login

# Initialize new project
railway init

# Add PostgreSQL
railway add postgresql

# Add Redis
railway add redis

# Deploy the backend
railway up

# Deploy the frontend (from frontend directory)
cd frontend
railway init
railway link # Link to same project
railway up
```

### Option 2: Deploy via Dashboard

1. Go to [Railway Dashboard](https://railway.app/dashboard)
2. Click **"New Project"**
3. Select **"Deploy from GitHub repo"**
4. Choose your BVRAI repository
5. Railway will auto-detect the Dockerfile

---

## Service Architecture

### Recommended Railway Project Structure

```
BVRAI Project/
├── bvrai-api        # Backend API service
├── bvrai-frontend   # Frontend service
├── PostgreSQL       # Database (Railway plugin)
└── Redis            # Cache (Railway plugin)
```

### Service Details

| Service | Port | Health Check | Description |
|---------|------|--------------|-------------|
| Backend API | 8000 | `/health` | FastAPI REST API |
| Frontend | 3000 | `/` | Next.js application |
| PostgreSQL | 5432 | Auto | Database |
| Redis | 6379 | Auto | Cache & sessions |

---

## Step-by-Step Deployment

### Step 1: Create Railway Project

1. Go to [Railway Dashboard](https://railway.app/dashboard)
2. Click **"New Project"** → **"Empty Project"**
3. Name your project (e.g., "bvrai-production")

### Step 2: Add PostgreSQL Database

1. In your project, click **"+ New"** → **"Database"** → **"Add PostgreSQL"**
2. Wait for provisioning (usually < 1 minute)
3. Click on the PostgreSQL service to see connection details
4. Copy the `DATABASE_URL` variable

### Step 3: Add Redis Cache

1. Click **"+ New"** → **"Database"** → **"Add Redis"**
2. Wait for provisioning
3. Copy the `REDIS_URL` variable

### Step 4: Deploy Backend API

1. Click **"+ New"** → **"GitHub Repo"**
2. Select your BVRAI repository
3. Railway will detect the root `Dockerfile`
4. Configure the service:
   - **Name**: `bvrai-api`
   - **Root Directory**: `/` (root of repo)
   - **Build Command**: Auto-detected from Dockerfile

5. Add environment variables (see [Environment Variables](#environment-variables))

### Step 5: Deploy Frontend

1. Click **"+ New"** → **"GitHub Repo"**
2. Select your BVRAI repository again
3. Configure the service:
   - **Name**: `bvrai-frontend`
   - **Root Directory**: `frontend`
   - **Build Command**: Auto-detected from Dockerfile

4. Add environment variables:
   ```
   NEXT_PUBLIC_API_URL=https://bvrai-api-production.up.railway.app
   ```

### Step 6: Connect Services

1. In the backend service settings, reference the database:
   ```
   DATABASE_URL=${{PostgreSQL.DATABASE_URL}}
   REDIS_URL=${{Redis.REDIS_URL}}
   ```

2. In the frontend service settings:
   ```
   NEXT_PUBLIC_API_URL=${{bvrai-api.RAILWAY_PUBLIC_DOMAIN}}
   ```

---

## Environment Variables

### Backend API (Required)

| Variable | Description | Example |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `${{PostgreSQL.DATABASE_URL}}` |
| `REDIS_URL` | Redis connection string | `${{Redis.REDIS_URL}}` |
| `JWT_SECRET` | Secret for JWT tokens | Generate: `openssl rand -hex 32` |
| `PII_ENCRYPTION_KEY` | 32-byte key for PII encryption | Generate: `openssl rand -hex 32` |

### Backend API (Optional - API Keys)

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key for LLM |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `DEEPGRAM_API_KEY` | Deepgram for speech-to-text |
| `ELEVENLABS_API_KEY` | ElevenLabs for text-to-speech |
| `TWILIO_ACCOUNT_SID` | Twilio Account SID |
| `TWILIO_AUTH_TOKEN` | Twilio Auth Token |
| `TWILIO_PHONE_NUMBER` | Twilio phone number |

### Frontend (Required)

| Variable | Description | Example |
|----------|-------------|---------|
| `NEXT_PUBLIC_API_URL` | Backend API URL | `https://bvrai-api-production.up.railway.app` |

### Setting Variables in Railway

1. Click on your service
2. Go to **"Variables"** tab
3. Click **"+ New Variable"**
4. Enter key and value
5. For references, use `${{ServiceName.VARIABLE_NAME}}`

**Example: Referencing PostgreSQL**
```
DATABASE_URL = ${{PostgreSQL.DATABASE_URL}}
```

---

## Database Setup

### Run Migrations

After deploying the backend, you need to run database migrations:

**Option 1: Via Railway CLI**
```bash
railway run --service bvrai-api alembic upgrade head
```

**Option 2: Via Railway Dashboard**
1. Go to your backend service
2. Click **"Settings"** → **"Deploy"**
3. Add a deploy command:
   ```
   alembic upgrade head && uvicorn bvrai_core.api.app:create_app --host 0.0.0.0 --port $PORT --factory
   ```

**Option 3: Via One-off Command**
1. Click on your backend service
2. Go to **"Shell"** or **"Terminal"**
3. Run:
   ```bash
   alembic upgrade head
   ```

---

## Connecting Services

### Internal Service Communication

Railway provides private networking between services:

```
# In backend environment variables
REDIS_URL=${{Redis.REDIS_PRIVATE_URL}}
DATABASE_URL=${{PostgreSQL.DATABASE_PRIVATE_URL}}
```

### Variable Reference Syntax

```
${{ServiceName.VariableName}}
```

Examples:
- `${{PostgreSQL.DATABASE_URL}}` - PostgreSQL connection string
- `${{Redis.REDIS_URL}}` - Redis connection string
- `${{bvrai-api.RAILWAY_PUBLIC_DOMAIN}}` - Public domain of API

---

## Custom Domains

### Adding a Custom Domain

1. Click on your service
2. Go to **"Settings"** → **"Networking"**
3. Click **"+ Custom Domain"**
4. Enter your domain (e.g., `api.bvrai.com`)
5. Add the provided DNS records to your domain registrar

### Recommended Domain Setup

| Service | Domain |
|---------|--------|
| Frontend | `app.bvrai.com` |
| Backend API | `api.bvrai.com` |

### DNS Configuration

Add these records to your domain:

```
# For frontend
CNAME app.bvrai.com → your-frontend-service.up.railway.app

# For backend
CNAME api.bvrai.com → your-backend-service.up.railway.app
```

---

## Monitoring & Logs

### Viewing Logs

1. Click on your service
2. Go to **"Deployments"** tab
3. Click on a deployment to see logs

### Real-time Log Streaming

```bash
railway logs --service bvrai-api
```

### Health Checks

Railway automatically monitors health endpoints:

- Backend: `GET /health`
- Frontend: `GET /`

Configure in `railway.toml`:
```toml
[deploy]
healthcheckPath = "/health"
healthcheckTimeout = 30
```

---

## Troubleshooting

### Common Issues

#### 1. Build Fails

**Symptom**: Deployment fails during build phase

**Solutions**:
- Check Docker build logs in Railway dashboard
- Ensure all dependencies are in `requirements.txt` or `package.json`
- Verify Dockerfile syntax

```bash
# Test build locally
docker build -t bvrai-test .
```

#### 2. Service Won't Start

**Symptom**: Service builds but doesn't become healthy

**Solutions**:
- Check start command in Dockerfile
- Verify PORT environment variable usage
- Check logs for application errors

```bash
# Check logs
railway logs --service bvrai-api
```

#### 3. Database Connection Failed

**Symptom**: `Connection refused` or timeout errors

**Solutions**:
- Verify `DATABASE_URL` is set correctly
- Use private URL for internal communication: `${{PostgreSQL.DATABASE_PRIVATE_URL}}`
- Check PostgreSQL service is running

#### 4. CORS Errors

**Symptom**: Frontend can't connect to API

**Solutions**:
- Add frontend URL to CORS whitelist in backend
- Check `NEXT_PUBLIC_API_URL` is correct
- Verify API is accessible

#### 5. Memory/CPU Issues

**Symptom**: Service restarts frequently or is slow

**Solutions**:
- Upgrade to Railway Pro for more resources
- Optimize application code
- Add resource limits in Railway settings

### Getting Help

1. **Railway Documentation**: [docs.railway.app](https://docs.railway.app)
2. **Railway Discord**: [discord.gg/railway](https://discord.gg/railway)
3. **BVRAI Issues**: [GitHub Issues](https://github.com/yourusername/bvrai/issues)

---

## Production Checklist

Before going live, ensure:

- [ ] All environment variables are set
- [ ] Database migrations are run
- [ ] Custom domains are configured
- [ ] SSL certificates are active (automatic with Railway)
- [ ] Health checks are passing
- [ ] API keys are added (OpenAI, Twilio, etc.)
- [ ] Monitoring is set up
- [ ] Backups are configured for PostgreSQL

---

## Cost Estimation

Railway pricing (as of 2025):

| Plan | Price | Resources |
|------|-------|-----------|
| Hobby | $5/month | 512MB RAM, 1 vCPU |
| Pro | $20/month | 8GB RAM, 8 vCPU |
| Team | Custom | Unlimited |

**Typical BVRAI deployment cost**: ~$15-30/month on Hobby plan

---

## Quick Reference

### Railway CLI Commands

```bash
# Login
railway login

# Initialize project
railway init

# Deploy
railway up

# View logs
railway logs

# Open dashboard
railway open

# Run command
railway run <command>

# Add database
railway add postgresql
railway add redis
```

### Important URLs

- Railway Dashboard: https://railway.app/dashboard
- Railway CLI: https://docs.railway.app/develop/cli
- Railway Status: https://status.railway.app

---

*Last updated: January 2026*
