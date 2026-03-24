"""AI Accounting Agent - Tripletex task automation with RAG-filtered OpenAPI tools.

Uses machine-core for model/embedding provider management, agent lifecycle,
OpenAPI tools generation, RAG tool filtering, and file processing.
Uses model-providers (via machine-core) for LLM and embedding provider resolution.
"""

from .coordinator import run_accounting_task
from .http_server import app

__version__ = "0.3.0"
__all__ = [
    "run_accounting_task",
    "app",
]
