"""AI Accounting Agent - Multi-agent framework for Tripletex task automation."""

from .planner import AccountingPlanner
from .task_splitter import TaskSplitter
from .coordinator import CoordinatorAgent, AccountingSubAgent, run_accounting_task
from .file_processor import FileProcessor
from .http_server import app

__version__ = "0.1.0"
__all__ = [
    "AccountingPlanner",
    "TaskSplitter",
    "CoordinatorAgent",
    "AccountingSubAgent",
    "run_accounting_task",
    "FileProcessor",
    "app",
]
