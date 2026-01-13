# Builder Engine - Docker Development Commands
# Usage: make <command>

.PHONY: help build up down restart logs shell db-shell redis-shell rabbitmq-shell \
        migrate seed test lint format clean prune dev prod

# Default target
.DEFAULT_GOAL := help

# Colors
CYAN := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m

# Help command
help: ## Show this help message
	@echo "$(CYAN)Builder Engine - Docker Development Commands$(RESET)"
	@echo ""
	@echo "$(GREEN)Usage:$(RESET) make $(YELLOW)<command>$(RESET)"
	@echo ""
	@echo "$(GREEN)Available commands:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(RESET) %s\n", $$1, $$2}'

# =============================================================================
# Core Commands
# =============================================================================

build: ## Build all Docker images
	@echo "$(CYAN)Building Docker images...$(RESET)"
	docker-compose build

build-no-cache: ## Build all Docker images without cache
	@echo "$(CYAN)Building Docker images (no cache)...$(RESET)"
	docker-compose build --no-cache

up: ## Start all services
	@echo "$(CYAN)Starting all services...$(RESET)"
	docker-compose up -d
	@echo "$(GREEN)Services started!$(RESET)"
	@echo "  API:      http://localhost:8000"
	@echo "  Frontend: http://localhost:3000"
	@echo "  Grafana:  http://localhost:3001"
	@echo "  RabbitMQ: http://localhost:15672"

down: ## Stop all services
	@echo "$(CYAN)Stopping all services...$(RESET)"
	docker-compose down

restart: ## Restart all services
	@echo "$(CYAN)Restarting all services...$(RESET)"
	docker-compose restart

status: ## Show status of all services
	docker-compose ps

logs: ## Show logs from all services
	docker-compose logs -f

logs-api: ## Show API service logs
	docker-compose logs -f api

logs-worker: ## Show worker service logs
	docker-compose logs -f worker

logs-frontend: ## Show frontend service logs
	docker-compose logs -f frontend

# =============================================================================
# Development Commands
# =============================================================================

dev: ## Start development environment with dev tools
	@echo "$(CYAN)Starting development environment...$(RESET)"
	docker-compose --profile dev up -d
	@echo "$(GREEN)Development environment started!$(RESET)"
	@echo "  API:      http://localhost:8000"
	@echo "  Frontend: http://localhost:3000"
	@echo "  MailHog:  http://localhost:8025"
	@echo "  PgAdmin:  http://localhost:5050"
	@echo "  Grafana:  http://localhost:3001"

prod: ## Start production-like environment
	@echo "$(CYAN)Starting production environment...$(RESET)"
	docker-compose --profile production up -d

shell-api: ## Open shell in API container
	docker-compose exec api /bin/bash

shell-worker: ## Open shell in worker container
	docker-compose exec worker /bin/bash

shell-frontend: ## Open shell in frontend container
	docker-compose exec frontend /bin/sh

# =============================================================================
# Database Commands
# =============================================================================

db-shell: ## Open PostgreSQL shell
	docker-compose exec postgres psql -U builderengine -d builderengine

db-migrate: ## Run database migrations
	docker-compose exec api alembic upgrade head

db-rollback: ## Rollback last migration
	docker-compose exec api alembic downgrade -1

db-seed: ## Seed database with sample data
	docker-compose exec api python -m src.scripts.seed_database

db-reset: ## Reset database (WARNING: deletes all data)
	@echo "$(RED)WARNING: This will delete all data!$(RESET)"
	@read -p "Are you sure? [y/N] " confirm && [ $${confirm:-N} = y ]
	docker-compose exec postgres psql -U builderengine -c "DROP DATABASE builderengine;"
	docker-compose exec postgres psql -U builderengine -c "CREATE DATABASE builderengine;"
	$(MAKE) db-migrate
	$(MAKE) db-seed

# =============================================================================
# Cache & Queue Commands
# =============================================================================

redis-shell: ## Open Redis CLI
	docker-compose exec redis redis-cli

redis-flush: ## Flush Redis cache
	docker-compose exec redis redis-cli FLUSHALL

rabbitmq-shell: ## Open RabbitMQ management shell
	@echo "$(CYAN)RabbitMQ Management: http://localhost:15672$(RESET)"
	@echo "Username: builderengine"
	@echo "Password: builderengine"

# =============================================================================
# Testing Commands
# =============================================================================

test: ## Run all tests
	docker-compose exec api pytest

test-cov: ## Run tests with coverage
	docker-compose exec api pytest --cov=src --cov-report=html

test-unit: ## Run unit tests only
	docker-compose exec api pytest tests/unit

test-integration: ## Run integration tests only
	docker-compose exec api pytest tests/integration

lint: ## Run linters
	docker-compose exec api ruff check src
	docker-compose exec api mypy src

format: ## Format code
	docker-compose exec api ruff format src
	docker-compose exec api ruff check --fix src

# =============================================================================
# Cleanup Commands
# =============================================================================

clean: ## Stop services and remove volumes
	@echo "$(CYAN)Cleaning up...$(RESET)"
	docker-compose down -v
	@echo "$(GREEN)Cleanup complete!$(RESET)"

prune: ## Remove all unused Docker resources
	@echo "$(RED)WARNING: This will remove all unused Docker resources!$(RESET)"
	@read -p "Are you sure? [y/N] " confirm && [ $${confirm:-N} = y ]
	docker system prune -af
	docker volume prune -f

# =============================================================================
# Utility Commands
# =============================================================================

setup: ## Initial setup (copy env, build, start)
	@echo "$(CYAN)Setting up Builder Engine...$(RESET)"
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "$(YELLOW)Created .env file - please update with your values$(RESET)"; \
	fi
	$(MAKE) build
	$(MAKE) up
	@echo "$(GREEN)Setup complete!$(RESET)"

health: ## Check health of all services
	@echo "$(CYAN)Checking service health...$(RESET)"
	@curl -s http://localhost:8000/health | jq . || echo "API: $(RED)DOWN$(RESET)"
	@curl -s http://localhost:3000/api/health | jq . || echo "Frontend: $(RED)DOWN$(RESET)"

update: ## Pull latest images and restart
	@echo "$(CYAN)Updating services...$(RESET)"
	docker-compose pull
	docker-compose up -d --build
	@echo "$(GREEN)Update complete!$(RESET)"

backup-db: ## Backup PostgreSQL database
	@echo "$(CYAN)Backing up database...$(RESET)"
	@mkdir -p ./backups
	docker-compose exec -T postgres pg_dump -U builderengine builderengine > ./backups/db_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "$(GREEN)Backup saved to ./backups/$(RESET)"

restore-db: ## Restore PostgreSQL database from backup
	@echo "$(CYAN)Available backups:$(RESET)"
	@ls -la ./backups/*.sql 2>/dev/null || echo "No backups found"
	@read -p "Enter backup filename: " backup && \
	docker-compose exec -T postgres psql -U builderengine -d builderengine < ./backups/$$backup
