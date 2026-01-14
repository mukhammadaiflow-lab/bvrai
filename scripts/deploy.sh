#!/bin/bash

# =============================================================================
# BVRAI Deployment Script
# =============================================================================
# This script automates the deployment process for the BVRAI platform.
# It handles environment setup, database migrations, and service deployment.
#
# Usage:
#   ./scripts/deploy.sh [environment] [options]
#
# Examples:
#   ./scripts/deploy.sh local          # Deploy locally with Docker Compose
#   ./scripts/deploy.sh staging        # Deploy to staging Kubernetes cluster
#   ./scripts/deploy.sh production     # Deploy to production Kubernetes cluster
#   ./scripts/deploy.sh --help         # Show help
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="${1:-local}"
NAMESPACE="bvrai"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-ghcr.io/bvrai}"
VERSION="${VERSION:-$(git rev-parse --short HEAD 2>/dev/null || echo 'latest')}"

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    cat << EOF
BVRAI Deployment Script

Usage: $(basename "$0") [environment] [options]

Environments:
  local         Deploy locally using Docker Compose (default)
  staging       Deploy to staging Kubernetes cluster
  production    Deploy to production Kubernetes cluster

Options:
  --build       Build Docker images before deploying
  --migrate     Run database migrations only
  --rollback    Rollback to previous deployment
  --status      Show deployment status
  --help, -h    Show this help message

Environment Variables:
  DOCKER_REGISTRY   Docker registry URL (default: ghcr.io/bvrai)
  VERSION           Version tag for images (default: git commit SHA)
  KUBECONFIG        Path to Kubernetes config file

Examples:
  ./deploy.sh local --build
  ./deploy.sh staging
  ./deploy.sh production --migrate
  VERSION=v1.0.0 ./deploy.sh production

EOF
}

check_prerequisites() {
    log_info "Checking prerequisites..."

    local missing=()

    # Check for required tools
    if ! command -v docker &> /dev/null; then
        missing+=("docker")
    fi

    if [[ "$ENVIRONMENT" != "local" ]]; then
        if ! command -v kubectl &> /dev/null; then
            missing+=("kubectl")
        fi
        if ! command -v helm &> /dev/null; then
            missing+=("helm")
        fi
    fi

    if [ ${#missing[@]} -gt 0 ]; then
        log_error "Missing required tools: ${missing[*]}"
        log_error "Please install them and try again."
        exit 1
    fi

    log_success "All prerequisites met"
}

check_environment_file() {
    local env_file="$PROJECT_ROOT/.env"

    if [ ! -f "$env_file" ]; then
        log_warning ".env file not found"
        log_info "Creating from .env.example..."

        if [ -f "$PROJECT_ROOT/.env.example" ]; then
            cp "$PROJECT_ROOT/.env.example" "$env_file"
            log_warning "Please edit .env file with your API keys before continuing"
            log_info "Required keys:"
            echo "  - POSTGRES_PASSWORD"
            echo "  - JWT_SECRET"
            echo "  - DEEPGRAM_API_KEY (for STT)"
            echo "  - ELEVENLABS_API_KEY (for TTS)"
            echo "  - OPENAI_API_KEY (for LLM)"
            exit 1
        else
            log_error ".env.example not found"
            exit 1
        fi
    fi

    # Check required variables
    source "$env_file" 2>/dev/null || true

    local missing_vars=()
    [ -z "${POSTGRES_PASSWORD:-}" ] && missing_vars+=("POSTGRES_PASSWORD")
    [ -z "${JWT_SECRET:-}" ] && missing_vars+=("JWT_SECRET")

    if [ ${#missing_vars[@]} -gt 0 ]; then
        log_error "Missing required environment variables: ${missing_vars[*]}"
        log_info "Please edit .env file and add these values"
        exit 1
    fi

    log_success "Environment file validated"
}

# =============================================================================
# Local Deployment (Docker Compose)
# =============================================================================

deploy_local() {
    log_info "Deploying locally with Docker Compose..."

    cd "$PROJECT_ROOT"

    # Check environment file
    check_environment_file

    # Build if requested
    if [[ "${BUILD:-false}" == "true" ]]; then
        log_info "Building Docker images..."
        docker compose build --no-cache
    fi

    # Start infrastructure services first
    log_info "Starting infrastructure services..."
    docker compose up -d postgres redis qdrant

    # Wait for PostgreSQL to be ready
    log_info "Waiting for PostgreSQL to be ready..."
    local retries=30
    while [ $retries -gt 0 ]; do
        if docker compose exec -T postgres pg_isready -U bvrai &> /dev/null; then
            break
        fi
        retries=$((retries - 1))
        sleep 2
    done

    if [ $retries -eq 0 ]; then
        log_error "PostgreSQL failed to start"
        exit 1
    fi

    log_success "PostgreSQL is ready"

    # Run migrations
    log_info "Running database migrations..."
    if command -v alembic &> /dev/null; then
        alembic upgrade head
    else
        log_warning "alembic not found, skipping migrations"
        log_info "Run: pip install alembic && alembic upgrade head"
    fi

    # Start all services
    log_info "Starting all services..."
    docker compose up -d

    # Wait for services to be healthy
    log_info "Waiting for services to be healthy..."
    sleep 10

    # Check service health
    local services=("platform-api:8086" "asr-service:8082" "tts-service:8083")
    for service in "${services[@]}"; do
        local name="${service%%:*}"
        local port="${service##*:}"

        if curl -sf "http://localhost:$port/health" > /dev/null 2>&1; then
            log_success "$name is healthy"
        else
            log_warning "$name may not be healthy (check logs: docker compose logs $name)"
        fi
    done

    echo ""
    log_success "Local deployment complete!"
    echo ""
    echo "Service URLs:"
    echo "  Platform API:     http://localhost:8086"
    echo "  API Docs:         http://localhost:8086/docs"
    echo "  WebRTC Gateway:   http://localhost:8087"
    echo "  PostgreSQL:       localhost:5432"
    echo "  Redis:            localhost:6379"
    echo ""
    echo "Commands:"
    echo "  View logs:        docker compose logs -f"
    echo "  Stop services:    docker compose down"
    echo "  Restart:          docker compose restart"
}

# =============================================================================
# Kubernetes Deployment
# =============================================================================

deploy_kubernetes() {
    local env="$1"

    log_info "Deploying to Kubernetes ($env)..."

    # Verify kubectl context
    local context=$(kubectl config current-context 2>/dev/null || echo "")
    if [ -z "$context" ]; then
        log_error "No kubectl context configured"
        log_info "Run: kubectl config use-context <context-name>"
        exit 1
    fi

    log_info "Using kubectl context: $context"

    # Confirm production deployment
    if [ "$env" == "production" ]; then
        echo ""
        log_warning "You are about to deploy to PRODUCTION"
        read -p "Are you sure? (yes/no): " confirm
        if [ "$confirm" != "yes" ]; then
            log_info "Deployment cancelled"
            exit 0
        fi
    fi

    cd "$PROJECT_ROOT"

    # Create namespace if it doesn't exist
    if ! kubectl get namespace "$NAMESPACE-$env" &> /dev/null; then
        log_info "Creating namespace: $NAMESPACE-$env"
        kubectl create namespace "$NAMESPACE-$env"
    fi

    # Build and push images if requested
    if [[ "${BUILD:-false}" == "true" ]]; then
        build_and_push_images
    fi

    # Apply Kubernetes manifests
    log_info "Applying Kubernetes manifests..."

    # Check if overlay exists
    local overlay_path="deploy/kubernetes/overlays/$env"
    if [ -d "$overlay_path" ]; then
        kubectl apply -k "$overlay_path"
    else
        log_warning "Overlay not found: $overlay_path"
        log_info "Applying base manifests..."
        kubectl apply -k deploy/kubernetes
    fi

    # Wait for deployments
    log_info "Waiting for deployments to be ready..."

    local deployments=("platform-api" "voice-engine" "conversation-engine")
    for deploy in "${deployments[@]}"; do
        log_info "Waiting for $deploy..."
        kubectl -n "$NAMESPACE-$env" rollout status deployment/"$deploy" --timeout=300s || true
    done

    # Run migrations
    if [[ "${MIGRATE:-false}" == "true" ]]; then
        run_migrations_kubernetes "$env"
    fi

    # Show deployment status
    show_status "$env"

    log_success "Kubernetes deployment complete!"
}

build_and_push_images() {
    log_info "Building and pushing Docker images..."

    local services=("platform-api" "voice-engine" "conversation-engine" "dashboard" "workers")

    for service in "${services[@]}"; do
        local image="$DOCKER_REGISTRY/$service:$VERSION"
        log_info "Building $image..."

        # Find Dockerfile
        local dockerfile=""
        if [ -f "services/$service/Dockerfile" ]; then
            dockerfile="services/$service/Dockerfile"
        elif [ -f "deploy/docker/$service/Dockerfile" ]; then
            dockerfile="deploy/docker/$service/Dockerfile"
        else
            log_warning "Dockerfile not found for $service, skipping"
            continue
        fi

        docker build -t "$image" -f "$dockerfile" .
        docker push "$image"

        log_success "Pushed $image"
    done
}

run_migrations_kubernetes() {
    local env="$1"

    log_info "Running database migrations..."

    # Create a migration job
    kubectl -n "$NAMESPACE-$env" run migration-job \
        --image="$DOCKER_REGISTRY/platform-api:$VERSION" \
        --restart=Never \
        --rm \
        -it \
        --command -- alembic upgrade head
}

# =============================================================================
# Utility Functions
# =============================================================================

show_status() {
    local env="${1:-local}"

    echo ""
    log_info "Deployment Status"
    echo "================="
    echo ""

    if [ "$env" == "local" ]; then
        docker compose ps
    else
        echo "Namespace: $NAMESPACE-$env"
        echo ""
        kubectl -n "$NAMESPACE-$env" get pods
        echo ""
        kubectl -n "$NAMESPACE-$env" get svc
    fi
}

rollback() {
    local env="${1:-staging}"

    log_info "Rolling back deployment in $env..."

    if [ "$env" == "local" ]; then
        log_error "Rollback not supported for local deployments"
        exit 1
    fi

    local deployments=("platform-api" "voice-engine" "conversation-engine")
    for deploy in "${deployments[@]}"; do
        log_info "Rolling back $deploy..."
        kubectl -n "$NAMESPACE-$env" rollout undo deployment/"$deploy"
    done

    log_success "Rollback complete"
}

# =============================================================================
# Main
# =============================================================================

main() {
    # Parse arguments
    local should_build=false
    local should_migrate=false
    local should_rollback=false
    local should_show_status=false

    for arg in "$@"; do
        case "$arg" in
            --build)
                should_build=true
                ;;
            --migrate)
                should_migrate=true
                ;;
            --rollback)
                should_rollback=true
                ;;
            --status)
                should_show_status=true
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            local|staging|production)
                ENVIRONMENT="$arg"
                ;;
        esac
    done

    export BUILD="$should_build"
    export MIGRATE="$should_migrate"

    echo ""
    echo "=================================================="
    echo "  BVRAI Deployment Script"
    echo "  Environment: $ENVIRONMENT"
    echo "  Version: $VERSION"
    echo "=================================================="
    echo ""

    # Check prerequisites
    check_prerequisites

    # Handle special commands
    if [ "$should_show_status" == "true" ]; then
        show_status "$ENVIRONMENT"
        exit 0
    fi

    if [ "$should_rollback" == "true" ]; then
        rollback "$ENVIRONMENT"
        exit 0
    fi

    # Deploy based on environment
    case "$ENVIRONMENT" in
        local)
            deploy_local
            ;;
        staging|production)
            deploy_kubernetes "$ENVIRONMENT"
            ;;
        *)
            log_error "Unknown environment: $ENVIRONMENT"
            show_help
            exit 1
            ;;
    esac
}

# Run main
main "$@"
