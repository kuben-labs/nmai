.PHONY: help dev dev-agent prod install clean kill-ports test-health

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m # No Color

# Local development URLs
AGENT_URL := http://localhost:8080
SOLVE_ENDPOINT := $(AGENT_URL)/solve

# Load environment variables from ai-accountant-mcp/.env if it exists
ifneq (,$(wildcard ai-accountant-mcp/.env))
    include ai-accountant-mcp/.env
    export
endif

# Load environment variables from .env if it exists
ifneq (,$(wildcard .env))
    include .env
    export
endif


help: ## Show this help message
	@echo "$(BLUE)AI Accounting Agent - Development Commands$(NC)"
	@echo ""
	@echo "$(YELLOW)Endpoints:$(NC)"
	@echo "  Agent Server:   $(AGENT_URL)"
	@echo "  Solve Endpoint: $(SOLVE_ENDPOINT)"
	@echo ""
	@echo "$(YELLOW)Architecture: Direct API (no MCP server needed)$(NC)"
	@echo "  Tools are generated from OpenAPI spec at startup"
	@echo "  Agent calls Tripletex API directly via httpx"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-25s$(NC) %s\n", $$1, $$2}'

# Development commands
dev: kill-ports dev-agent ## Start the agent server (direct API mode, no MCP)

dev-agent: ## Start Agent server (single worker, hot-reload)
	@echo "$(BLUE)Starting Agent server on $(AGENT_URL) (direct API mode, hot-reload)...$(NC)"
	@uv run uvicorn src.ai_accounting_agent.http_server:app --host 0.0.0.0 --port 8080 --reload

prod: kill-ports ## Start Agent server for production (4 workers, no reload)
	@echo "$(BLUE)Starting Agent server on $(AGENT_URL) (production, 4 workers)...$(NC)"
	@uv run uvicorn src.ai_accounting_agent.http_server:app --host 0.0.0.0 --port 8080 --workers 4

kill-ports: ## Free development ports
	@echo "$(YELLOW)Freeing development ports (8080)...$(NC)"
	@pids=$$(lsof -ti tcp:8080 2>/dev/null); \
	if [ -n "$$pids" ]; then \
		echo "Stopping processes $$pids on port 8080"; \
		echo "$$pids" | xargs kill; \
		sleep 1; \
	else \
		echo "No process found on port 8080"; \
	fi

dev-cli: ## Start CLI mode for testing agents locally
	@echo "$(BLUE)Starting Agent CLI mode...$(NC)"
	@uv run python -m ai_accounting_agent.main

install: ## Install all dependencies
	@echo "$(BLUE)Installing dependencies...$(NC)"
	@uv sync

clean: ## Clean all build artifacts and caches
	@echo "$(BLUE)Cleaning project...$(NC)"
	@rm -rf .venv __pycache__ .pytest_cache .ruff_cache *.egg-info dist build
	@rm -rf src/__pycache__ src/ai_accounting_agent/__pycache__
	@rm -rf ai-accountant-mcp/__pycache__ ai-accountant-mcp/src/__pycache__
	@rm -rf ai-accountant-mcp/.venv
	@echo "$(GREEN)✓ Clean complete$(NC)"

# Test commands
test-health: ## Check if agent server is running
	@echo "$(YELLOW)Testing service health...$(NC)"
	@echo ""
	@echo "$(BLUE)Checking Agent server...$(NC)"
	@curl -s http://localhost:8080/health 2>/dev/null || echo "Agent server not responding"

# Default target
.DEFAULT_GOAL := help
