.PHONY: help build up down logs clean dev prod debug legacy-run

# Default target
help:
	@echo "UAV Log Viewer Docker Commands:"
	@echo ""
	@echo "Development:"
	@echo "  make dev          - Start development environment"
	@echo "  make legacy-run    - Run with original Docker command (frontend only)"
	@echo "  make build        - Build all containers"
	@echo "  make up           - Start all services"
	@echo "  make down         - Stop all services"
	@echo "  make logs         - Show logs for all services"
	@echo "  make clean        - Remove containers, networks, and volumes"
	@echo ""
	@echo "Production:"
	@echo "  make prod         - Start production environment"
	@echo "  make prod-build   - Build production containers"
	@echo ""
	@echo "Debug:"
	@echo "  make debug        - Start with Redis Commander"
	@echo ""
	@echo "Individual Services:"
	@echo "  make backend      - Start only backend"
	@echo "  make frontend     - Start only frontend"
	@echo "  make redis        - Start only Redis"

# Development environment
dev:
	docker-compose --profile dev up -d

# Legacy run command (replicates your original Docker run)
legacy-run:
	@echo "Building Docker image..."
	docker build -t uavlogviewer .
	@echo "Loading environment variables..."
	@if [ -f .env ]; then \
		echo "Found root .env file, loading variables..."; \
		CESIUM_TOKEN=$$(awk '/^VUE_APP_CESIUM_TOKEN=/{flag=1; val=substr($$0,index($$0,"=")+1); next} /^[A-Z_]+=/{flag=0} flag{val=val$$0} END{gsub(/#.*/,"",val); gsub(/^[ \t]+|[ \t]+$$/,"",val); print val}' .env); \
	elif [ -f backend/.env ]; then \
		echo "Found backend/.env file, loading variables..."; \
		CESIUM_TOKEN=$$(awk '/^VUE_APP_CESIUM_TOKEN=/{flag=1; val=substr($$0,index($$0,"=")+1); next} /^[A-Z_]+=/{flag=0} flag{val=val$$0} END{gsub(/#.*/,"",val); gsub(/^[ \t]+|[ \t]+$$/,"",val); print val}' backend/.env); \
	else \
		echo "No .env file found. Please create one with your Cesium token:"; \
		echo "VUE_APP_CESIUM_TOKEN=your_cesium_ion_token_here"; \
		echo "Running without Cesium token..."; \
		docker run -it -p 8080:8080 -v $${PWD}:/usr/src/app uavlogviewer; \
		exit 0; \
	fi; \
	echo "Token length: $${#CESIUM_TOKEN}"; \
	echo "Running with volume mount and Cesium token..."; \
	if [ -n "$$CESIUM_TOKEN" ]; then \
		echo "Cesium token found, running with token..."; \
		docker run -e VUE_APP_CESIUM_TOKEN="$$CESIUM_TOKEN" -it -p 8080:8080 -v $${PWD}:/usr/src/app uavlogviewer; \
	else \
		echo "No Cesium token found, running without token..."; \
		docker run -it -p 8080:8080 -v $${PWD}:/usr/src/app uavlogviewer; \
	fi

# Production environment
prod:
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Debug environment with Redis Commander
debug:
	docker-compose --profile debug up -d

# Build containers
build:
	docker-compose build

# Build production containers
prod-build:
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml build

# Start all services
up:
	docker-compose up -d

# Stop all services
down:
	docker-compose down

# Show logs
logs:
	docker-compose logs -f

# Clean everything
clean:
	docker-compose down -v --remove-orphans
	docker system prune -f

# Individual services
backend:
	docker-compose up -d backend redis

frontend:
	docker-compose up -d frontend

redis:
	docker-compose up -d redis

# Utility commands
restart:
	docker-compose restart

status:
	docker-compose ps

# Database commands
redis-cli:
	docker-compose exec redis redis-cli

# Backend commands
backend-logs:
	docker-compose logs -f backend

backend-shell:
	docker-compose exec backend bash

# Frontend commands
frontend-logs:
	docker-compose logs -f frontend

frontend-shell:
	docker-compose exec frontend sh 