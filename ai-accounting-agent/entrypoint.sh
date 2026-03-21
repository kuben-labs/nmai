#!/bin/bash
set -e

# No MCP server needed - agent calls Tripletex API directly via httpx
# The OpenAPI spec is fetched at startup and tools are generated in-process

echo "Starting Agent server (direct API mode, no MCP)..."
cd /app/ && python -m uvicorn src.ai_accounting_agent.http_server:app --host 0.0.0.0 --port 8080
