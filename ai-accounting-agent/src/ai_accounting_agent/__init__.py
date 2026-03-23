"""AI Accounting Agent - Tripletex task automation with RAG-filtered OpenAPI tools.

Uses machine-core for model/embedding provider management and agent lifecycle.
Uses model-providers (via machine-core) for LLM and embedding provider resolution.
"""

from .coordinator import run_accounting_task
from .file_processor import FileProcessor
from .http_server import app

__version__ = "0.2.0"
__all__ = [
    "run_accounting_task",
    "FileProcessor",
    "app",
]
