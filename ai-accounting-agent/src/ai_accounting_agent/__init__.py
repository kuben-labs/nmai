"""AI Accounting Agent - Tripletex task automation with RAG-filtered MCP tools."""

from .coordinator import run_accounting_task
from .file_processor import FileProcessor
from .http_server import app

__version__ = "0.1.0"
__all__ = [
    "run_accounting_task",
    "FileProcessor",
    "app",
]
