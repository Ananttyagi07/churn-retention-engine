# ============================================
# Customer Churn & Retention Engine
# Comprehensive Makefile for Development & Deployment
# ============================================

.PHONY: help
.DEFAULT_GOAL := help

# ============================================
# Variables
# ============================================
PYTHON := python3
PIP := pip3
VENV := venv
VENV_BIN := $(VENV)/bin
PYTHON_VENV := $(VENV_BIN)/python
PIP_VENV := $(VENV_BIN)/pip

# Project Settings
PROJECT_NAME := churn-retention-engine
SRC_DIR := src
APP_DIR := app
TESTS_DIR := tests
DATA_DIR := data
CONFIG_DIR := config

# Docker Settings
DOCKER_COMPOSE := docker-compose
DOCKER_IMAGE_API := $(PROJECT_NAME)-api
DOCKER_IMAGE_WORKER := $(PROJECT_NAME)-worker
DOCKER_IMAGE_UI := $(PROJECT_NAME)-ui
DOCKER_REGISTRY := ghcr.io/yourusername
VERSION := $(shell git describe --always --dirty --long 2>/dev/null || echo "dev")

# Code Quality Tools
BLACK := black
ISORT := isort
FLAKE8 := flake8
MYPY := mypy
PYTEST := pytest
BANDIT := bandit

# Colors for output
COLOR_RESET := \033[0m
COLOR_BOLD := \033[1m
COLOR_GREEN := \033[32m
COLOR_YELLOW := \033[33m
COLOR_BLUE := \033[36m

# ============================================
# Help Command
# ============================================
help: ## Show this help message
	@echo "$(COLOR_BOLD)$(PROJECT_NAME) - Available Commands:$(COLOR_RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(COLOR_BLUE)%-20s$(COLOR_RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(COLOR_YELLOW)Quick Start:$(COLOR_RESET)"
	@echo "  make setup          - Initial project setup"
	@echo "  make up             - Start all services with Docker"
	@echo "  make dev            - Start development server"
	@echo ""

# ============================================
# Environment Setup
# ============================================
.PHONY: setup
setup: ## Complete initial project setup
	@echo "$(COLOR_GREEN)Setting up $(PROJECT_NAME)...$(COLOR_RESET)"
	@$(MAKE) create-dirs
	@$(MAKE) venv
	@$(MAKE) install
	@$(MAKE) create-env
	@$(MAKE) pre-commit-install
	@echo "$(COLOR_GREEN)✓ Setup complete! Activate venv: source $(VENV)/bin/activate$(COLOR_RESET)"

.PHONY: create-dirs
create-dirs: ## Create necessary project directories
	@echo "$(COLOR_BLUE)Creating project directories...$(COLOR_RESET)"
	@mkdir -p $(DATA_DIR)/raw $(DATA_DIR)/interim $(DATA_DIR)/processed
	@mkdir -p $(DATA_DIR)/external $(DATA_DIR)/models $(DATA_DIR)/reports $(DATA_DIR)/logs
	@mkdir -p $(TESTS_DIR) $(CONFIG_DIR)
	@touch $(DATA_DIR)/raw/.gitkeep $(DATA_DIR)/interim/.gitkeep
	@touch $(DATA_DIR)/processed/.gitkeep $(DATA_DIR)/models/.gitkeep
	@touch $(DATA_DIR)/reports/.gitkeep $(DATA_DIR)/logs/.gitkeep
	@echo "$(COLOR_GREEN)✓ Directories created$(COLOR_RESET)"

.PHONY: venv
venv: ## Create Python virtual environment
	@echo "$(COLOR_BLUE)Creating virtual environment...$(COLOR_RESET)"
	@test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	@$(PIP_VENV) install --upgrade pip setuptools wheel
	@echo "$(COLOR_GREEN)✓ Virtual environment created$(COLOR_RESET)"

.PHONY: install
install: ## Install production dependencies
	@echo "$(COLOR_BLUE)Installing dependencies...$(COLOR_RESET)"
	@$(PIP_VENV) install -r requirements.txt
	@echo "$(COLOR_GREEN)✓ Dependencies installed$(COLOR_RESET)"

.PHONY: install-dev
install-dev: install ## Install development dependencies
	@echo "$(COLOR_BLUE)Installing development dependencies...$(COLOR_RESET)"
	@$(PIP_VENV) install -r requirements-dev.txt
	@echo "$(COLOR_GREEN)✓ Development dependencies installed$(COLOR_RESET)"

.PHONY: create-env
create-env: ## Create .env file from template
	@if [ ! -f .env ]; then \
		echo "$(COLOR_YELLOW)Creating .env from template...$(COLOR_RESET)"; \
		cp .env.example .env; \
		echo "$(COLOR_GREEN)✓ .env created - Please update with your credentials$(COLOR_RESET)"; \
	else \
		echo "$(COLOR_YELLOW).env already exists$(COLOR_RESET)"; \
	fi

.PHONY: update
update: ## Update all dependencies
	@echo "$(COLOR_BLUE)Updating dependencies...$(COLOR_RESET)"
	@$(PIP_VENV) install --upgrade -r requirements.txt
	@$(PIP_VENV) install --upgrade -r requirements-dev.txt
	@echo "$(COLOR_GREEN)✓ Dependencies updated$(COLOR_RESET)"

# ============================================
# Development
# ============================================
.PHONY: dev
dev: ## Start Flask development server
	@echo "$(COLOR_GREEN)Starting development server...$(COLOR_RESET)"
	@export FLASK_ENV=development && export FLASK_DEBUG=1 && $(PYTHON_VENV) -m flask run --host=0.0.0.0 --port=5000

.PHONY: dev-api
dev-api: ## Start API server only
	@echo "$(COLOR_GREEN)Starting API server...$(COLOR_RESET)"
	@$(PYTHON_VENV) app/wsgi.py

.PHONY: notebook
notebook: ## Start Jupyter notebook server
	@echo "$(COLOR_GREEN)Starting Jupyter notebook...$(COLOR_RESET)"
	@$(VENV_BIN)/jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

.PHONY: shell
shell: ## Start Python interactive shell
	@$(PYTHON_VENV) -i -c "from app import create_app; app = create_app(); app.app_context().push()"

# ============================================
# Code Quality
# ============================================
.PHONY: format
format: ## Format code with black and isort
	@echo "$(COLOR_BLUE)Formatting code...$(COLOR_RESET)"
	@$(VENV_BIN)/$(BLACK) $(SRC_DIR) $(APP_DIR) $(TESTS_DIR)
	@$(VENV_BIN)/$(ISORT) $(SRC_DIR) $(APP_DIR) $(TESTS_DIR)
	@echo "$(COLOR_GREEN)✓ Code formatted$(COLOR_RESET)"

.PHONY: lint
lint: ## Run linting checks
	@echo "$(COLOR_BLUE)Running linters...$(COLOR_RESET)"
	@$(VENV_BIN)/$(FLAKE8) $(SRC_DIR) $(APP_DIR) $(TESTS_DIR) || true
	@$(VENV_BIN)/pylint $(SRC_DIR) $(APP_DIR) --exit-zero
	@echo "$(COLOR_GREEN)✓ Linting complete$(COLOR_RESET)"

.PHONY: type-check
type-check: ## Run type checking with mypy
	@echo "$(COLOR_BLUE)Running type checks...$(COLOR_RESET)"
	@$(VENV_BIN)/$(MYPY) $(SRC_DIR) $(APP_DIR) || true
	@echo "$(COLOR_GREEN)✓ Type checking complete$(COLOR_RESET)"

.PHONY: security
security: ## Run security checks
	@echo "$(COLOR_BLUE)Running security checks...$(COLOR_RESET)"
	@$(VENV_BIN)/$(BANDIT) -r $(SRC_DIR) $(APP_DIR) -f json -o security-report.json || true
	@$(VENV_BIN)/safety check --json || true
	@echo "$(COLOR_GREEN)✓ Security checks complete$(COLOR_RESET)"

.PHONY: check
check: format lint type-check ## Run all code quality checks
	@echo "$(COLOR_GREEN)✓ All checks complete$(COLOR_RESET)"

# ============================================
# Testing
# ============================================
.PHONY: test
test: ## Run all tests
	@echo "$(COLOR_BLUE)Running tests...$(COLOR_RESET)"
	@$(VENV_BIN)/$(PYTEST) $(TESTS_DIR) -v --tb=short

.PHONY: test-unit
test-unit: ## Run unit tests only
	@$(VENV_BIN)/$(PYTEST) $(TESTS_DIR) -m "unit" -v

.PHONY: test-integration
test-integration: ## Run integration tests only
	@$(VENV_BIN)/$(PYTEST) $(TESTS_DIR) -m "integration" -v

.PHONY: test-api
test-api: ## Run API tests only
	@$(VENV_BIN)/$(PYTEST) $(TESTS_DIR) -m "api" -v

.PHONY: test-ml
test-ml: ## Run ML pipeline tests only
	@$(VENV_BIN)/$(PYTEST) $(TESTS_DIR) -m "ml" -v

.PHONY: test-cov
test-cov: ## Run tests with coverage report
	@echo "$(COLOR_BLUE)Running tests with coverage...$(COLOR_RESET)"
	@$(VENV_BIN)/$(PYTEST) $(TESTS_DIR) --cov=$(SRC_DIR) --cov=$(APP_DIR) --cov-report=html --cov-report=term
	@echo "$(COLOR_GREEN)✓ Coverage report generated in htmlcov/$(COLOR_RESET)"

.PHONY: coverage
coverage: test-cov ## Generate and open coverage report
	@echo "$(COLOR_BLUE)Opening coverage report...$(COLOR_RESET)"
	@open htmlcov/index.html || xdg-open htmlcov/index.html || echo "Open htmlcov/index.html manually"

# ============================================
# Machine Learning Pipeline
# ============================================
.PHONY: data
data: ## Download and prepare datasets
	@echo "$(COLOR_BLUE)Preparing datasets...$(COLOR_RESET)"
	@$(PYTHON_VENV) -m src.data.ingestion
	@echo "$(COLOR_GREEN)✓ Data prepared$(COLOR_RESET)"

.PHONY: features
features: ## Generate features
	@echo "$(COLOR_BLUE)Generating features...$(COLOR_RESET)"
	@$(PYTHON_VENV) -m src.data.feature_engineering
	@echo "$(COLOR_GREEN)✓ Features generated$(COLOR_RESET)"

.PHONY: train
train: ## Train models
	@echo "$(COLOR_BLUE)Training models...$(COLOR_RESET)"
	@$(PYTHON_VENV) -m src.models.train --config $(CONFIG_DIR)/development.yaml
	@echo "$(COLOR_GREEN)✓ Models trained$(COLOR_RESET)"

.PHONY: evaluate
evaluate: ## Evaluate trained models
	@echo "$(COLOR_BLUE)Evaluating models...$(COLOR_RESET)"
	@$(PYTHON_VENV) -m src.models.evaluate
	@echo "$(COLOR_GREEN)✓ Evaluation complete$(COLOR_RESET)"

.PHONY: pipeline
pipeline: data features train evaluate ## Run complete ML pipeline
	@echo "$(COLOR_GREEN)✓ ML pipeline complete$(COLOR_RESET)"

# ============================================
# Docker Commands
# ============================================
.PHONY: build
build: ## Build all Docker images
	@echo "$(COLOR_BLUE)Building Docker images...$(COLOR_RESET)"
	@$(DOCKER_COMPOSE) build
	@echo "$(COLOR_GREEN)✓ Images built$(COLOR_RESET)"

.PHONY: build-api
build-api: ## Build API Docker image
	@docker build -f Dockerfile.api -t $(DOCKER_IMAGE_API):$(VERSION) -t $(DOCKER_IMAGE_API):latest .

.PHONY: build-worker
build-worker: ## Build worker Docker image
	@docker build -f Dockerfile.worker -t $(DOCKER_IMAGE_WORKER):$(VERSION) -t $(DOCKER_IMAGE_WORKER):latest .

.PHONY: up
up: ## Start all services with Docker Compose
	@echo "$(COLOR_GREEN)Starting all services...$(COLOR_RESET)"
	@$(DOCKER_COMPOSE) up -d
	@echo "$(COLOR_GREEN)✓ Services started$(COLOR_RESET)"
	@echo "$(COLOR_YELLOW)API: http://localhost:5000$(COLOR_RESET)"
	@echo "$(COLOR_YELLOW)Dashboard: http://localhost:8080$(COLOR_RESET)"

.PHONY: down
down: ## Stop all Docker services
	@echo "$(COLOR_BLUE)Stopping services...$(COLOR_RESET)"
	@$(DOCKER_COMPOSE) down
	@echo "$(COLOR_GREEN)✓ Services stopped$(COLOR_RESET)"

.PHONY: restart
restart: down up ## Restart all Docker services

.PHONY: logs
logs: ## View logs from all services
	@$(DOCKER_COMPOSE) logs -f

.PHONY: logs-api
logs-api: ## View API service logs
	@$(DOCKER_COMPOSE) logs -f api

.PHONY: logs-worker
logs-worker: ## View worker service logs
	@$(DOCKER_COMPOSE) logs -f worker

.PHONY: ps
ps: ## Show running containers
	@$(DOCKER_COMPOSE) ps

.PHONY: exec-api
exec-api: ## Execute bash in API container
	@$(DOCKER_COMPOSE) exec api bash

.PHONY: exec-worker
exec-worker: ## Execute bash in worker container
	@$(DOCKER_COMPOSE) exec worker bash

# ============================================
# Database Commands
# ============================================
.PHONY: db-init
db-init: ## Initialize database
	@echo "$(COLOR_BLUE)Initializing database...$(COLOR_RESET)"
	@$(DOCKER_COMPOSE) exec api flask db init
	@echo "$(COLOR_GREEN)✓ Database initialized$(COLOR_RESET)"

.PHONY: db-migrate
db-migrate: ## Create database migration
	@echo "$(COLOR_BLUE)Creating migration...$(COLOR_RESET)"
	@$(DOCKER_COMPOSE) exec api flask db migrate -m "$(msg)"

.PHONY: db-upgrade
db-upgrade: ## Apply database migrations
	@echo "$(COLOR_BLUE)Applying migrations...$(COLOR_RESET)"
	@$(DOCKER_COMPOSE) exec api flask db upgrade
	@echo "$(COLOR_GREEN)✓ Migrations applied$(COLOR_RESET)"

.PHONY: db-downgrade
db-downgrade: ## Rollback last migration
	@$(DOCKER_COMPOSE) exec api flask db downgrade

.PHONY: db-reset
db-reset: ## Reset database (WARNING: destructive)
	@echo "$(COLOR_YELLOW)WARNING: This will delete all data!$(COLOR_RESET)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		$(DOCKER_COMPOSE) exec db psql -U churn_user -d churn_db -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"; \
		$(MAKE) db-upgrade; \
	fi

# ============================================
# Pre-commit Hooks
# ============================================
.PHONY: pre-commit-install
pre-commit-install: ## Install pre-commit hooks
	@echo "$(COLOR_BLUE)Installing pre-commit hooks...$(COLOR_RESET)"
	@$(VENV_BIN)/pre-commit install
	@echo "$(COLOR_GREEN)✓ Pre-commit hooks installed$(COLOR_RESET)"

.PHONY: pre-commit-run
pre-commit-run: ## Run pre-commit on all files
	@$(VENV_BIN)/pre-commit run --all-files

# ============================================
# Deployment
# ============================================
.PHONY: deploy-staging
deploy-staging: ## Deploy to staging environment
	@echo "$(COLOR_BLUE)Deploying to staging...$(COLOR_RESET)"
	@export ENV=staging && $(DOCKER_COMPOSE) -f docker-compose.yml -f docker-compose.staging.yml up -d
	@echo "$(COLOR_GREEN)✓ Deployed to staging$(COLOR_RESET)"

.PHONY: deploy-prod
deploy-prod: ## Deploy to production environment
	@echo "$(COLOR_YELLOW)Deploying to production...$(COLOR_RESET)"
	@export ENV=production && $(DOCKER_COMPOSE) -f docker-compose.yml -f docker-compose.production.yml up -d
	@echo "$(COLOR_GREEN)✓ Deployed to production$(COLOR_RESET)"

.PHONY: push-images
push-images: ## Push Docker images to registry
	@echo "$(COLOR_BLUE)Pushing images to registry...$(COLOR_RESET)"
	@docker tag $(DOCKER_IMAGE_API):latest $(DOCKER_REGISTRY)/$(DOCKER_IMAGE_API):$(VERSION)
	@docker tag $(DOCKER_IMAGE_API):latest $(DOCKER_REGISTRY)/$(DOCKER_IMAGE_API):latest
	@docker push $(DOCKER_REGISTRY)/$(DOCKER_IMAGE_API):$(VERSION)
	@docker push $(DOCKER_REGISTRY)/$(DOCKER_IMAGE_API):latest
	@echo "$(COLOR_GREEN)✓ Images pushed$(COLOR_RESET)"

# ============================================
# Monitoring & Maintenance
# ============================================
.PHONY: health
health: ## Check service health
	@echo "$(COLOR_BLUE)Checking service health...$(COLOR_RESET)"
	@curl -s http://localhost:5000/api/v1/health | python -m json.tool || echo "API not responding"

.PHONY: metrics
metrics: ## View Prometheus metrics
	@curl -s http://localhost:5000/metrics

.PHONY: backup-db
backup-db: ## Backup database
	@echo "$(COLOR_BLUE)Backing up database...$(COLOR_RESET)"
	@mkdir -p backups
	@$(DOCKER_COMPOSE) exec -T db pg_dump -U churn_user churn_db > backups/backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "$(COLOR_GREEN)✓ Database backed up$(COLOR_RESET)"

.PHONY: restore-db
restore-db: ## Restore database from backup (file=backup.sql)
	@echo "$(COLOR_YELLOW)Restoring database from $(file)...$(COLOR_RESET)"
	@$(DOCKER_COMPOSE) exec -T db psql -U churn_user churn_db < $(file)
	@echo "$(COLOR_GREEN)✓ Database restored$(COLOR_RESET)"

# ============================================
# Cleanup
# ============================================
.PHONY: clean
clean: ## Clean cache files and build artifacts
	@echo "$(COLOR_BLUE)Cleaning up...$(COLOR_RESET)"
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/ .ruff_cache/
	@rm -rf *.egg-info
	@echo "$(COLOR_GREEN)✓ Cleaned up$(COLOR_RESET)"

.PHONY: clean-data
clean-data: ## Clean generated data files
	@echo "$(COLOR_YELLOW)Cleaning data files...$(COLOR_RESET)"
	@rm -rf $(DATA_DIR)/interim/* $(DATA_DIR)/processed/* $(DATA_DIR)/models/* $(DATA_DIR)/reports/*
	@echo "$(COLOR_GREEN)✓ Data files cleaned$(COLOR_RESET)"

.PHONY: clean-docker
clean-docker: ## Remove all Docker containers and volumes
	@echo "$(COLOR_YELLOW)Removing Docker containers and volumes...$(COLOR_RESET)"
	@$(DOCKER_COMPOSE) down -v --remove-orphans
	@echo "$(COLOR_GREEN)✓ Docker cleaned$(COLOR_RESET)"

.PHONY: clean-all
clean-all: clean clean-data clean-docker ## Deep clean everything
	@rm -rf $(VENV)
	@echo "$(COLOR_GREEN)✓ Complete cleanup done$(COLOR_RESET)"

# ============================================
# Documentation
# ============================================
.PHONY: docs
docs: ## Generate documentation
	@echo "$(COLOR_BLUE)Generating documentation...$(COLOR_RESET)"
	@cd docs && $(VENV_BIN)/sphinx-build -b html . _build/html
	@echo "$(COLOR_GREEN)✓ Documentation generated$(COLOR_RESET)"

.PHONY: docs-serve
docs-serve: docs ## Serve documentation locally
	@echo "$(COLOR_GREEN)Serving docs at http://localhost:8000$(COLOR_RESET)"
	@cd docs/_build/html && $(PYTHON_VENV) -m http.server 8000

# ============================================
# Utility Commands
# ============================================
.PHONY: requirements
requirements: ## Generate requirements.txt from pyproject.toml
	@$(VENV_BIN)/pip-compile pyproject.toml -o requirements.txt

.PHONY: version
version: ## Show current version
	@echo "$(COLOR_BOLD)Version: $(VERSION)$(COLOR_RESET)"

.PHONY: info
info: ## Show project information
	@echo "$(COLOR_BOLD)Project Information:$(COLOR_RESET)"
	@echo "  Name: $(PROJECT_NAME)"
	@echo "  Version: $(VERSION)"
	@echo "  Python: $(shell $(PYTHON) --version)"
	@echo "  Venv: $(VENV)"
	@echo "  Docker Compose: $(shell $(DOCKER_COMPOSE) --version 2>/dev/null || echo 'Not installed')"

# ============================================
# Quick Shortcuts
# ============================================
.PHONY: install-all
install-all: install-dev ## Install all dependencies (alias)

.PHONY: run
run: up ## Alias for 'up' command

.PHONY: stop
stop: down ## Alias for 'down' command
