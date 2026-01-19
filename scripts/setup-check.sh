#!/bin/bash

# =============================================================================
# BVRAI Setup Verification Script
# =============================================================================
# This script verifies your environment is correctly configured for BVRAI.
# Run this before deploying to identify any missing requirements.
# =============================================================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo ""
echo "=================================================="
echo "  BVRAI Setup Verification"
echo "=================================================="
echo ""

ERRORS=0
WARNINGS=0

check_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

check_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ERRORS=$((ERRORS + 1))
}

check_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    WARNINGS=$((WARNINGS + 1))
}

check_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# =============================================================================
# 1. System Requirements
# =============================================================================

echo "1. Checking System Requirements"
echo "-------------------------------"

# Docker
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version | grep -oE '[0-9]+\.[0-9]+' | head -1)
    check_pass "Docker installed (v$DOCKER_VERSION)"

    if docker info &> /dev/null; then
        check_pass "Docker daemon running"
    else
        check_fail "Docker daemon not running (start Docker Desktop or run: sudo systemctl start docker)"
    fi
else
    check_fail "Docker not installed (https://docs.docker.com/get-docker/)"
fi

# Docker Compose
if command -v docker compose &> /dev/null || command -v docker-compose &> /dev/null; then
    check_pass "Docker Compose installed"
else
    check_fail "Docker Compose not installed"
fi

# Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | grep -oE '[0-9]+\.[0-9]+')
    check_pass "Python installed (v$PYTHON_VERSION)"

    if [[ "$(echo "$PYTHON_VERSION >= 3.10" | bc)" == "1" ]] 2>/dev/null || \
       [[ "${PYTHON_VERSION%%.*}" -ge "3" && "${PYTHON_VERSION#*.}" -ge "10" ]]; then
        check_pass "Python version >= 3.10"
    else
        check_warn "Python 3.10+ recommended (you have $PYTHON_VERSION)"
    fi
else
    check_fail "Python 3 not installed"
fi

# Git
if command -v git &> /dev/null; then
    check_pass "Git installed"
else
    check_fail "Git not installed"
fi

echo ""

# =============================================================================
# 2. Environment Configuration
# =============================================================================

echo "2. Checking Environment Configuration"
echo "--------------------------------------"

ENV_FILE="$PROJECT_ROOT/.env"

if [ -f "$ENV_FILE" ]; then
    check_pass ".env file exists"

    # Source the env file
    set +u
    source "$ENV_FILE" 2>/dev/null || true
    set -u

    # Check required variables
    if [ -n "${POSTGRES_PASSWORD:-}" ]; then
        check_pass "POSTGRES_PASSWORD is set"
    else
        check_fail "POSTGRES_PASSWORD not set in .env"
    fi

    if [ -n "${JWT_SECRET:-}" ]; then
        if [ "${#JWT_SECRET}" -ge 32 ]; then
            check_pass "JWT_SECRET is set (${#JWT_SECRET} chars)"
        else
            check_warn "JWT_SECRET should be at least 32 characters"
        fi
    else
        check_fail "JWT_SECRET not set in .env"
    fi

    # Check API keys (optional but recommended)
    if [ -n "${DEEPGRAM_API_KEY:-}" ]; then
        check_pass "DEEPGRAM_API_KEY is set (STT)"
    else
        check_warn "DEEPGRAM_API_KEY not set - STT will not work"
    fi

    if [ -n "${ELEVENLABS_API_KEY:-}" ]; then
        check_pass "ELEVENLABS_API_KEY is set (TTS)"
    else
        check_warn "ELEVENLABS_API_KEY not set - TTS will not work"
    fi

    if [ -n "${OPENAI_API_KEY:-}" ]; then
        check_pass "OPENAI_API_KEY is set (LLM)"
    else
        check_warn "OPENAI_API_KEY not set - LLM will not work"
    fi

else
    check_fail ".env file not found"
    check_info "Run: cp .env.example .env && nano .env"
fi

echo ""

# =============================================================================
# 3. API Key Validation
# =============================================================================

echo "3. Validating API Keys (optional)"
echo "----------------------------------"

# Validate Deepgram
if [ -n "${DEEPGRAM_API_KEY:-}" ]; then
    RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" \
        -H "Authorization: Token $DEEPGRAM_API_KEY" \
        "https://api.deepgram.com/v1/projects" 2>/dev/null || echo "000")

    if [ "$RESPONSE" == "200" ]; then
        check_pass "Deepgram API key is valid"
    elif [ "$RESPONSE" == "401" ]; then
        check_fail "Deepgram API key is invalid"
    else
        check_warn "Could not validate Deepgram key (network error?)"
    fi
else
    check_info "Skipping Deepgram validation (key not set)"
fi

# Validate ElevenLabs
if [ -n "${ELEVENLABS_API_KEY:-}" ]; then
    RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" \
        -H "xi-api-key: $ELEVENLABS_API_KEY" \
        "https://api.elevenlabs.io/v1/user" 2>/dev/null || echo "000")

    if [ "$RESPONSE" == "200" ]; then
        check_pass "ElevenLabs API key is valid"
    elif [ "$RESPONSE" == "401" ]; then
        check_fail "ElevenLabs API key is invalid"
    else
        check_warn "Could not validate ElevenLabs key (network error?)"
    fi
else
    check_info "Skipping ElevenLabs validation (key not set)"
fi

# Validate OpenAI
if [ -n "${OPENAI_API_KEY:-}" ]; then
    RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" \
        -H "Authorization: Bearer $OPENAI_API_KEY" \
        "https://api.openai.com/v1/models" 2>/dev/null || echo "000")

    if [ "$RESPONSE" == "200" ]; then
        check_pass "OpenAI API key is valid"
    elif [ "$RESPONSE" == "401" ]; then
        check_fail "OpenAI API key is invalid"
    else
        check_warn "Could not validate OpenAI key (network error?)"
    fi
else
    check_info "Skipping OpenAI validation (key not set)"
fi

echo ""

# =============================================================================
# 4. Port Availability
# =============================================================================

echo "4. Checking Port Availability"
echo "------------------------------"

check_port() {
    local port=$1
    local service=$2

    if ! lsof -i ":$port" &> /dev/null; then
        check_pass "Port $port available ($service)"
    else
        check_warn "Port $port in use ($service) - may cause conflicts"
    fi
}

check_port 5432 "PostgreSQL"
check_port 6379 "Redis"
check_port 6333 "Qdrant"
check_port 8086 "Platform API"
check_port 8082 "ASR Service"
check_port 8083 "TTS Service"
check_port 8087 "WebRTC Gateway"

echo ""

# =============================================================================
# 5. Disk Space
# =============================================================================

echo "5. Checking Disk Space"
echo "----------------------"

AVAILABLE_GB=$(df -BG "$PROJECT_ROOT" | awk 'NR==2 {print $4}' | tr -d 'G')

if [ "$AVAILABLE_GB" -ge 20 ]; then
    check_pass "Disk space: ${AVAILABLE_GB}GB available (>20GB required)"
elif [ "$AVAILABLE_GB" -ge 10 ]; then
    check_warn "Disk space: ${AVAILABLE_GB}GB available (20GB recommended)"
else
    check_fail "Disk space: ${AVAILABLE_GB}GB available (need at least 10GB)"
fi

echo ""

# =============================================================================
# 6. Memory
# =============================================================================

echo "6. Checking Memory"
echo "------------------"

if command -v free &> /dev/null; then
    TOTAL_MEM_GB=$(free -g | awk '/^Mem:/{print $2}')
    AVAILABLE_MEM_GB=$(free -g | awk '/^Mem:/{print $7}')

    if [ "$TOTAL_MEM_GB" -ge 8 ]; then
        check_pass "Total RAM: ${TOTAL_MEM_GB}GB (>8GB required)"
    elif [ "$TOTAL_MEM_GB" -ge 4 ]; then
        check_warn "Total RAM: ${TOTAL_MEM_GB}GB (8GB recommended)"
    else
        check_fail "Total RAM: ${TOTAL_MEM_GB}GB (need at least 4GB)"
    fi
else
    check_info "Could not check memory (free command not available)"
fi

echo ""

# =============================================================================
# Summary
# =============================================================================

echo "=================================================="
echo "  Summary"
echo "=================================================="
echo ""

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}All checks passed! You're ready to deploy.${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Run: ./scripts/deploy.sh local"
    echo "  2. Open: http://localhost:8086/docs"
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}$WARNINGS warnings found. Deployment may work but with limited functionality.${NC}"
    echo ""
    echo "You can proceed, but consider fixing the warnings."
else
    echo -e "${RED}$ERRORS errors found. Please fix them before deploying.${NC}"
    echo ""
    echo "Common fixes:"
    echo "  - Install Docker: https://docs.docker.com/get-docker/"
    echo "  - Create .env: cp .env.example .env && nano .env"
    echo "  - Get API keys: See README.md for provider links"
fi

echo ""
exit $ERRORS
