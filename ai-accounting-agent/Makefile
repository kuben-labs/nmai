.PHONY: help dev install

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m # No Color

# Local development URLs (override production URLs in .env files)
VIS_DOMAIN := http://localhost:3000

help: ## Show this help message
	@echo "$(BLUE)MCP Development Commands$(NC)"
	@echo "Local VIS domain: $(VIS_DOMAIN)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

# Development commands
dev: ## Start all services in development mode (requires tmux or multiple terminals)
	@echo "$(YELLOW)Starting all services...$(NC)"
	@$(MAKE) -j1 dev-ai-accountant-mcp

dev-ai-accountant-mcp: ## Start ai-accountant-mcp development server
	@echo "$(BLUE)Starting ai-accountant-mcp...$(NC)"
	@cd ai-accountant-mcp && uv run python src/ai_accountant_mcp/server.py

install: ## Install all (uv)
	@echo "$(BLUE)Installing dependencies...$(NC)"
	@$(MAKE) -j1 install-ai-accountant

install-ai-accountant: ## Install ai-accountant dependencies (uv)
	@echo "$(BLUE)Installing ai-accountant dependencies...$(NC)"
	@uv sync

clean: ## Clean all build artifacts and dependencies
	@echo "$(BLUE)Cleaning all build artifacts and dependencies...$(NC)"
	@$(MAKE) -j1 clean-ai-accountant

clean-ai-accountant: ## Clean ai-accountant build artifacts and dependencies
	@echo "$(BLUE)Cleaning ai-accountant...$(NC)"
	@rm -rf .venv __pycache__
	
# Default target
.DEFAULT_GOAL := help