#!/bin/bash
#
# Builder Voice AI Platform - Quick Start Script
# This is the simplest way to get the platform running
#
# Usage: ./scripts/quick-start.sh
#

set -e

echo "=================================="
echo "  Builder Voice AI - Quick Start  "
echo "=================================="
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed!"
    echo "Install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "ERROR: Docker Compose is not installed!"
    exit 1
fi

# Go to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"

# Create .env if not exists
if [ ! -f ".env" ]; then
    echo "[1/4] Creating environment file..."
    cp .env.example .env

    # Generate keys
    SECRET_KEY=$(openssl rand -hex 32 2>/dev/null || head -c 64 /dev/urandom | xxd -p | tr -d '\n' | head -c 64)
    ENCRYPTION_KEY=$(openssl rand -hex 32 2>/dev/null || head -c 64 /dev/urandom | xxd -p | tr -d '\n' | head -c 64)

    # Update .env with generated keys
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s/SECRET_KEY=.*/SECRET_KEY=$SECRET_KEY/" .env
        sed -i '' "s/ENCRYPTION_KEY=.*/ENCRYPTION_KEY=$ENCRYPTION_KEY/" .env
    else
        sed -i "s/SECRET_KEY=.*/SECRET_KEY=$SECRET_KEY/" .env
        sed -i "s/ENCRYPTION_KEY=.*/ENCRYPTION_KEY=$ENCRYPTION_KEY/" .env
    fi

    echo "    Created .env with generated keys"
else
    echo "[1/4] Environment file exists"
fi

# Build and start
echo "[2/4] Building Docker images..."
docker-compose build --quiet

echo "[3/4] Starting services..."
docker-compose up -d

echo "[4/4] Waiting for services to start..."
sleep 10

# Health checks
echo ""
echo "Checking services..."

check_service() {
    if curl -s "$1" > /dev/null 2>&1; then
        echo "  ✓ $2 is running"
        return 0
    else
        echo "  ✗ $2 is starting..."
        return 1
    fi
}

check_service "http://localhost:8086/health" "Backend API"
check_service "http://localhost:3000" "Frontend"

echo ""
echo "=================================="
echo "        Deployment Complete!       "
echo "=================================="
echo ""
echo "Access the platform:"
echo "  Frontend:    http://localhost:3000"
echo "  Backend API: http://localhost:8086"
echo "  API Docs:    http://localhost:8086/docs"
echo ""
echo "Useful commands:"
echo "  View logs:   docker-compose logs -f"
echo "  Stop:        docker-compose down"
echo "  Restart:     docker-compose restart"
echo ""
echo "Next steps:"
echo "  1. Open http://localhost:3000 in your browser"
echo "  2. Register a new account"
echo "  3. Create your first AI agent"
echo ""
