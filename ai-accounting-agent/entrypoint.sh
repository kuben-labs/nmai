#!/bin/bash
set -e

# Start MCP server in the background
echo "Starting MCP server..."
cd /app/ai-accountant-mcp && python src/ai_accountant_mcp/server.py &
MCP_PID=$!

# Give MCP server time to start
sleep 2

# Start Agent server in the foreground
echo "Starting Agent server..."
cd /app/ && python -m uvicorn src.ai_accounting_agent.http_server:app --host 0.0.0.0 --port 8080

# If agent exits, kill MCP server too
trap "kill $MCP_PID 2>/dev/null || true" EXIT
