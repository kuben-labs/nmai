.PHONY: help dev dev-all dev-mcp dev-agent install clean clean-ai-accountant

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m # No Color

# Local development URLs
MCP_URL := http://localhost:8083/mcp
AGENT_URL := http://localhost:8000
SOLVE_ENDPOINT := $(AGENT_URL)/solve

help: ## Show this help message
	@echo "$(BLUE)AI Accounting Agent - Development Commands$(NC)"
	@echo ""
	@echo "$(YELLOW)Endpoints:$(NC)"
	@echo "  MCP Server:     $(MCP_URL)"
	@echo "  Agent Server:   $(AGENT_URL)"
	@echo "  Solve Endpoint: $(SOLVE_ENDPOINT)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-25s$(NC) %s\n", $$1, $$2}'

# Development commands
dev: dev-all ## Start all services (alias for dev-all)

dev: ## Start all services in development mode
	@echo "$(YELLOW)Starting all services...$(NC)"
	@$(MAKE) -j3 kill-ports dev-mcp dev-agent


kill-ports: ## Free development ports (8000, 8083)
	@echo "$(YELLOW)Freeing development ports (8000, 8083)...$(NC)"
	@for port in 8000 8083; do \
		pids=$$(lsof -ti tcp:$$port 2>/dev/null); \
		if [ -n "$$pids" ]; then \
			echo "Stopping processes $$pids on port $$port"; \
			echo "$$pids" | xargs kill; \
		else \
			echo "No process found on port $$port"; \
		fi; \
	done

dev-mcp: ## Start MCP server only
	@echo "$(BLUE)Starting MCP server on $(MCP_URL)...$(NC)"
	@cd ai-accountant-mcp && uv run python src/ai_accountant_mcp/server.py

dev-agent: ## Start Agent server only
	@echo "$(BLUE)Starting Agent server on $(AGENT_URL)...$(NC)"
	@cd src && uv run uvicorn ai_accounting_agent.http_server:app --host 0.0.0.0 --port 8000 --reload

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
test-health: ## Check if services are running
	@echo "$(YELLOW)Testing service health...$(NC)"
	@echo ""
	@echo "$(BLUE)Checking MCP server...$(NC)"
	@curl -s http://localhost:8083/health 2>/dev/null || echo "MCP server not responding"
	@echo ""
	@echo "$(BLUE)Checking Agent server...$(NC)"
	@curl -s http://localhost:8000/health 2>/dev/null || echo "Agent server not responding"
# Default target
.DEFAULT_GOAL := help
