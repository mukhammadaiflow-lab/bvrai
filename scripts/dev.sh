#!/bin/bash
# =============================================================================
# BVRAI Voice AI Platform - Local Development Script
# =============================================================================
# Usage:
#   ./scripts/dev.sh         - Start all services
#   ./scripts/dev.sh up      - Start all services
#   ./scripts/dev.sh down    - Stop all services
#   ./scripts/dev.sh logs    - View logs
#   ./scripts/dev.sh restart - Restart services
#   ./scripts/dev.sh api     - Start only API (without Docker)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${GREEN}==============================================================================${NC}"
    echo -e "${GREEN} BVRAI Voice AI Platform - Local Development${NC}"
    echo -e "${GREEN}==============================================================================${NC}"
}

print_status() {
    echo -e "${YELLOW}>>> $1${NC}"
}

print_error() {
    echo -e "${RED}ERROR: $1${NC}"
}

# Check if .env exists
check_env() {
    if [ ! -f .env ]; then
        print_status "Creating .env file from .env.example..."
        cp .env.example .env
        echo -e "${YELLOW}Please edit .env with your API keys before running again.${NC}"
    fi
}

# Start all services with Docker Compose
start_services() {
    print_header
    check_env
    print_status "Starting services with Docker Compose..."
    docker-compose -f docker-compose.local.yml up -d

    echo ""
    print_status "Services started successfully!"
    echo ""
    echo "  API:      http://localhost:8000"
    echo "  API Docs: http://localhost:8000/docs"
    echo "  Frontend: http://localhost:3000"
    echo "  Postgres: localhost:5432"
    echo "  Redis:    localhost:6379"
    echo ""
    print_status "View logs with: ./scripts/dev.sh logs"
}

# Stop all services
stop_services() {
    print_status "Stopping services..."
    docker-compose -f docker-compose.local.yml down
    print_status "Services stopped."
}

# View logs
view_logs() {
    docker-compose -f docker-compose.local.yml logs -f
}

# Restart services
restart_services() {
    stop_services
    start_services
}

# Start API only (without Docker, for development)
start_api_local() {
    print_header
    check_env

    # Load environment variables
    set -a
    source .env
    set +a

    # Use SQLite for local development without Docker
    export DATABASE_URL="sqlite+aiosqlite:///./bvrai.db"

    print_status "Starting API locally (SQLite mode)..."
    print_status "API will be available at http://localhost:8000"
    print_status "API docs at http://localhost:8000/docs"
    echo ""

    # Install dependencies if needed
    if [ ! -d "venv" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv venv
        source venv/bin/activate
        pip install -r requirements.txt
    else
        source venv/bin/activate
    fi

    # Run the API
    PYTHONPATH="$PROJECT_ROOT" uvicorn bvrai_core.api.app:create_app --factory --reload --host 0.0.0.0 --port 8000
}

# Run database migrations
run_migrations() {
    print_status "Running database migrations..."
    docker-compose -f docker-compose.local.yml exec api python -m alembic upgrade head
}

# Show help
show_help() {
    echo "Usage: ./scripts/dev.sh [command]"
    echo ""
    echo "Commands:"
    echo "  up, start     Start all services"
    echo "  down, stop    Stop all services"
    echo "  restart       Restart all services"
    echo "  logs          View logs (follow mode)"
    echo "  api           Start API locally without Docker (uses SQLite)"
    echo "  migrate       Run database migrations"
    echo "  help          Show this help message"
}

# Main command handling
case "${1:-up}" in
    up|start)
        start_services
        ;;
    down|stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    logs)
        view_logs
        ;;
    api)
        start_api_local
        ;;
    migrate)
        run_migrations
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
