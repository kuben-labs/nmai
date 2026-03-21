# AGENTS.md - AI Accounting Agent

This document provides guidelines for AI coding agents working in this repository.

## Project Overview

AI agent that completes accounting tasks in Tripletex. The agent receives task prompts
(in 7 languages: Norwegian, English, Spanish, Portuguese, Nynorsk, German, French),
uses the Tripletex API via MCP tools, and gets scored on correctness and efficiency.

**Architecture:**
- `src/ai_accounting_agent/` - Main agent (FastAPI HTTP server + coordinator)
- `ai-accountant-mcp/` - MCP server wrapping 800+ Tripletex API endpoints
- `docs/` - Competition documentation (task specs, scoring, examples)

## Build & Development Commands

### Installation
```bash
uv sync                    # Install all dependencies (uses uv package manager)
make install               # Alias for uv sync
```

### Running Services
```bash
make dev                   # Start both MCP server (8083) and Agent server (8080)
make dev-mcp               # Start MCP server only (http://localhost:8083/mcp)
make dev-agent             # Start Agent server only (http://localhost:8080)
make dev-cli               # Run CLI mode for local testing
make kill-ports            # Free ports 8080 and 8083
make test-health           # Check if services are running
```

### Docker
```bash
docker-compose up --build  # Build and run containerized agent
```

### Cleaning
```bash
make clean                 # Remove .venv, __pycache__, .pytest_cache, etc.
```

## Testing

No formal test suite exists yet. To verify functionality:
```bash
# Check services are healthy
curl http://localhost:8083/health   # MCP server health
curl http://localhost:8080/health   # Agent server health

# Test solve endpoint (requires running services)
curl -X POST http://localhost:8080/solve \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Test task", "tripletex_credentials": {"base_url": "...", "session_token": "..."}}'
```

## Code Style Guidelines

### Python Version
- Python 3.13+ required (see `.python-version`)

### Imports
Order imports as follows:
```python
# 1. Standard library
import os
import json
import asyncio
from typing import Dict, List, Any, Optional

# 2. Third-party packages
from fastapi import FastAPI, Request
from pydantic import BaseModel
from loguru import logger
from dotenv import load_dotenv

# 3. Local imports (relative within package)
from .coordinator import run_accounting_task
from .prompts import ACCOUNTING_SUBAGENT_SYSTEM_INSTRUCTIONS
```

### Type Annotations
- Always use type hints for function parameters and return values
- Use `Optional[T]` for nullable types
- Use `Dict`, `List`, `Any` from `typing` module
```python
async def run_task(
    prompt: str,
    files: List[Dict[str, str]] = None,
    credentials: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
```

### Naming Conventions
- **Classes**: PascalCase (`AccountingSubAgent`, `FileProcessor`)
- **Functions/Methods**: snake_case (`run_accounting_task`, `process_files`)
- **Constants**: UPPER_SNAKE_CASE (`SESSION_TOKEN`, `API_URL`)
- **Variables**: snake_case (`file_data`, `task_result`)
- **Private methods**: prefix with underscore (`_extract_content`, `_initialize_rag_filtering`)

### Async/Await
- Use `async def` for I/O-bound operations (API calls, file processing)
- All MCP tool interactions are async
- Coordinator and sub-agents use async execution

### Logging
Use loguru throughout:
```python
from loguru import logger

logger.info(f"Processing task: {task[:100]}...")
logger.debug(f"File size: {size} bytes")
logger.warning(f"Unknown mime type: {mime_type}")
logger.error(f"Failed to process: {e}", exc_info=True)
```

### Error Handling
- Wrap API calls in try/except blocks
- Log errors with context before re-raising or returning error dicts
- Return structured error responses with `success: False` and `error` message
```python
try:
    result = await self.run_query(prompt)
except Exception as e:
    logger.error(f"SubAgent {self.subtask_id} error: {e}")
    return {"success": False, "error": str(e)}
```

### Docstrings
Use Google-style docstrings:
```python
def process_attachment(filename: str, content_base64: str, mime_type: str) -> Dict[str, Any]:
    """
    Process a single attached file.

    Args:
        filename: Original filename
        content_base64: Base64-encoded content
        mime_type: MIME type of the file

    Returns:
        Dictionary with file metadata and extracted data
    """
```

### Pydantic Models
Use Pydantic for request/response validation:
```python
class SolveRequest(BaseModel):
    """Request to solve an accounting task."""
    prompt: str
    files: Optional[List[FileInput]] = None
    tripletex_credentials: TripletexCredentials
```

## Environment Variables

Copy `.env.example` to `.env` and configure:
```bash
LLM_PROVIDER=ollama           # LLM provider (ollama, openai, etc.)
LLM_MODEL=...                 # Model name
EMBEDDING_PROVIDER=ollama     # Embedding provider
EMBEDDING_DIMENSIONS=256      # Embedding dimensions
OLLAMA_BASE_URL=http://...    # Ollama endpoint (if using)
```

For MCP server (`ai-accountant-mcp/.env`):
```bash
SESSION_TOKEN=<tripletex-session-token>
API_URL=https://tx-proxy.ainm.no/v2
LOGIN_URL=<email>
WEB_UI=https://...
```

## Key Implementation Details

### Tripletex API Authentication
The MCP server handles auth internally via Basic Auth:
- Username: `0` (zero)
- Password: SESSION_TOKEN from environment

### /solve Endpoint Contract
- POST `/solve` with JSON body containing `prompt`, `files`, `tripletex_credentials`
- Must respond within 5 minutes (300s timeout)
- Return `{"status": "completed"}` on success

### RAG Tool Filtering
The codebase implements RAG-based tool filtering to reduce the 800+ MCP tools to
relevant subsets (~300-500) per task. This improves LLM context efficiency.

### File Processing
Supports PDF (pdfplumber/PyPDF2), images (pytesseract OCR), and text/CSV files.
Files are base64-decoded, saved to `/tmp`, and content extracted.

## Competition Scoring

- Field-by-field verification (correctness): 0-1 normalized
- Tier multiplier: Tier 1 (x1), Tier 2 (x2), Tier 3 (x3)
- Efficiency bonus (on perfect scores): fewer API calls + zero 4xx errors
- Max score per task: 6.0 (perfect Tier 3 + best efficiency)

## Common Tripletex API Patterns

```python
# List entities
GET /employee?fields=id,firstName,lastName,email

# Create entity
POST /customer with JSON body {"name": "...", "email": "...", "isCustomer": True}

# Linking entities
GET /customer → POST /order → POST /invoice

# Delete
DELETE /travelExpense/{id}
```

## Tips for Agents

1. Parse prompts fully before making API calls
2. Avoid trial-and-error - every 4xx error reduces efficiency score
3. Use MCP tools, not direct HTTP calls - auth is handled automatically
4. Read error messages carefully - validationMessages tell you what's wrong
5. Verify results with GET calls after creating entities
6. Handle all 7 languages in prompts
