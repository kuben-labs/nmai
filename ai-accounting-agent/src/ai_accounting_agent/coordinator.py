"""Coordinator for accounting task execution with sub-agents.

This module provides direct integration with pydantic-ai and MCP toolsets
without relying on machine-core or model-providers packages.
"""

import json
import os
from typing import Dict, List, Any, Optional

from loguru import logger
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP

from .llm import get_google_model
from .embeddings import get_embedding_provider
from .prompts import (
    ACCOUNTING_SUBAGENT_SYSTEM_INSTRUCTIONS,
    ACCOUNTING_SUBAGENT_PROMPT_TEMPLATE,
    ACCOUNTING_COORDINATOR_SYSTEM_INSTRUCTIONS,
    ACCOUNTING_COORDINATOR_PROMPT_TEMPLATE,
)


def load_mcp_config(config_path: str = "mcp_accountant.json") -> List[Dict[str, Any]]:
    """Load MCP server configurations from a JSON file.

    Args:
        config_path: Path to the MCP config file

    Returns:
        List of server configurations
    """
    try:
        with open(config_path, "r") as f:
            config_data = json.load(f)

        servers = []
        mcp_servers = config_data.get("servers", {})

        for server_name, server_config in mcp_servers.items():
            servers.append(
                {
                    "name": server_name,
                    "type": server_config.get("type", "http"),
                    "url": server_config.get("url", ""),
                }
            )

        logger.info(f"Loaded {len(servers)} MCP server(s) from {config_path}")
        return servers

    except Exception as e:
        logger.error(f"Failed to load MCP config from {config_path}: {e}")
        return []


def setup_mcp_toolsets(
    server_configs: List[Dict[str, Any]],
    timeout: float = 300.0,
) -> List[MCPServerStreamableHTTP]:
    """Set up MCP toolsets from server configurations.

    Args:
        server_configs: List of server configurations
        timeout: Timeout for MCP connections in seconds

    Returns:
        List of configured MCP toolsets
    """
    toolsets = []

    for config in server_configs:
        try:
            server_type = config.get("type", "http")
            url = config.get("url", "")

            if server_type in ("http", "sse"):
                server = MCPServerStreamableHTTP(
                    url=url,
                    timeout=timeout,
                )
                toolsets.append(server)
                logger.info(f"Added {server_type} MCP server: {url}")
            else:
                logger.warning(f"Unknown server type: {server_type}")

        except Exception as e:
            logger.error(f"Failed to setup MCP server {config}: {e}")

    return toolsets


class AccountingSubAgent:
    """Sub-agent that executes a specific accounting subtask with RAG filtering."""

    def __init__(
        self,
        subtask_id: str,
        subtask_title: str,
        subtask_description: str,
        main_task: str,
        tripletex_credentials: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a sub-agent for a specific accounting subtask.

        Args:
            subtask_id: Unique identifier for the subtask
            subtask_title: Title of the subtask
            subtask_description: Detailed instructions for this subtask
            main_task: The overall accounting task
            tripletex_credentials: Tripletex API credentials (base_url, session_token)
        """
        self.subtask_id = subtask_id
        self.subtask_title = subtask_title
        self.tripletex_credentials = tripletex_credentials or {}
        self.use_rag_filtering = True
        self.rag_manager = None
        self._rag_initialized = False

        # Extract credentials for inclusion in prompt
        base_url = self.tripletex_credentials.get(
            "base_url", "https://api.tripletex.no/v2"
        )
        session_token = self.tripletex_credentials.get("session_token", "")

        self._user_prompt = ACCOUNTING_SUBAGENT_PROMPT_TEMPLATE.format(
            subtask_id=subtask_id,
            subtask_title=subtask_title,
            subtask_description=subtask_description,
            main_task=main_task,
            base_url=base_url,
            session_token=session_token,
        )

        # Set up model and toolsets
        self.model = get_google_model()

        # Load MCP config and create toolsets
        server_configs = load_mcp_config()
        self.toolsets = setup_mcp_toolsets(server_configs)
        self._original_toolsets = list(self.toolsets)

        # Create agent
        self.agent = Agent(
            model=self.model,
            toolsets=self.toolsets,
            system_prompt=ACCOUNTING_SUBAGENT_SYSTEM_INSTRUCTIONS,
            retries=3,
        )

        logger.info(
            f"SubAgent {subtask_id} initialized with {len(self.toolsets)} toolset(s)"
        )

    async def run(self, task: str = "") -> str:
        """Required method for running the agent."""
        return await self.execute()

    async def execute(self) -> str:
        """Execute this subtask using MCP tools with RAG filtering.

        Returns:
            Report of what was executed
        """
        logger.info(f"SubAgent {self.subtask_id} executing: {self.subtask_title}")

        try:
            # Initialize RAG on first execution
            if self.use_rag_filtering and not self._rag_initialized:
                await self._initialize_rag_filtering()

            result = await self.agent.run(self._user_prompt)

            # Extract the report
            if hasattr(result, "data"):
                report = getattr(result, "data")
            elif hasattr(result, "output"):
                report = getattr(result, "output")
            else:
                report = str(result)

            logger.info(f"SubAgent {self.subtask_id} completed execution")
            return report

        except Exception as e:
            logger.error(f"SubAgent {self.subtask_id} error: {e}")
            return f"Error in {self.subtask_id}: {str(e)}"

    async def _initialize_rag_filtering(self):
        """Initialize RAG filtering for this sub-agent."""
        try:
            logger.info(
                f"SubAgent {self.subtask_id}: Initializing RAG tool filtering..."
            )

            from .rag_tool_manager import create_rag_tool_manager

            # Get embedding provider
            try:
                embedding_provider = get_embedding_provider()
                logger.debug(
                    f"SubAgent {self.subtask_id}: Using Google embedding provider"
                )
            except Exception as e:
                logger.warning(
                    f"SubAgent {self.subtask_id}: Could not load embedding provider: {e}"
                )
                embedding_provider = None

            # Create RAG manager
            self.rag_manager = create_rag_tool_manager(
                embedding_provider=embedding_provider,
                top_k=100,
            )

            # Initialize RAG manager
            await self.rag_manager.initialize()

            # Index tools from original toolsets
            if self._original_toolsets:
                logger.debug(
                    f"SubAgent {self.subtask_id}: Indexing {len(self._original_toolsets)} toolset(s)..."
                )
                await self.rag_manager.index_toolsets(self._original_toolsets)  # type: ignore

                stats = self.rag_manager.get_statistics()
                logger.info(
                    f"SubAgent {self.subtask_id}: RAG initialized with {stats.get('total_tools', 0)} tools, "
                    f"will filter to ~300 most relevant per subtask"
                )

            # Create filtered toolsets for this subtask
            if self._original_toolsets and self.rag_manager:
                filtered_toolsets = await self.rag_manager.create_filtered_toolsets(
                    self._original_toolsets, self.subtask_title[:100]
                )  # type: ignore
                self.toolsets = filtered_toolsets

                # Recreate agent with filtered toolsets
                self.agent = Agent(
                    model=self.model,
                    toolsets=filtered_toolsets,
                    system_prompt=ACCOUNTING_SUBAGENT_SYSTEM_INSTRUCTIONS,
                    retries=3,
                )

                logger.info(
                    f"SubAgent {self.subtask_id}: Applied RAG filtering with "
                    f"{len(filtered_toolsets)} filtered toolset(s)"
                )

            self._rag_initialized = True
            return True

        except Exception as e:
            logger.warning(
                f"SubAgent {self.subtask_id}: RAG filtering initialization failed: {e}"
            )
            self.use_rag_filtering = False
            return False


class CoordinatorAgent:
    """Coordinator agent that orchestrates accounting sub-agents."""

    def __init__(
        self,
        task_summary: str,
        task_type: str,
        complexity: str,
        subtasks: List[Dict[str, Any]],
        base_url: str = "http://localhost:8083",
        tripletex_credentials: Optional[Dict[str, Any]] = None,
        use_rag_filtering: bool = True,
    ):
        """Initialize the coordinator agent.

        Args:
            task_summary: Summary of the accounting task
            task_type: Type of task (e.g., "Tier 1", "Tier 2", "Tier 3")
            complexity: Task complexity ("simple" or "complex")
            subtasks: List of subtask dictionaries
            base_url: Tripletex proxy base URL
            tripletex_credentials: Tripletex API credentials (base_url, session_token)
            use_rag_filtering: Whether to use RAG filtering for tool reduction
        """
        self.task_summary = task_summary
        self.task_type = task_type
        self.complexity = complexity
        self.subtasks = subtasks
        self.base_url = base_url
        self.tripletex_credentials = tripletex_credentials or {}
        self.use_rag_filtering = use_rag_filtering
        self.rag_manager = None
        self._rag_initialized = False

        # Set up model and toolsets
        self.model = get_google_model()

        # Load MCP config and create toolsets
        server_configs = load_mcp_config()
        self.toolsets = setup_mcp_toolsets(server_configs)
        self._original_toolsets = list(self.toolsets)

        # Create agent
        self.agent = Agent(
            model=self.model,
            toolsets=self.toolsets,
            system_prompt=ACCOUNTING_COORDINATOR_SYSTEM_INSTRUCTIONS,
            retries=3,
        )

        logger.info(f"Coordinator initialized with {len(self.toolsets)} toolset(s)")

    async def run(self, query: str = "") -> str:
        """Required method for running the agent."""
        return await self.coordinate_execution()

    async def coordinate_execution(self) -> str:
        """Execute the accounting task directly with RAG filtering.

        Returns:
            Summary of execution results
        """
        logger.info(f"Coordinator executing task: {self.task_summary[:100]}")

        try:
            # Initialize RAG filtering on first execution
            if self.use_rag_filtering and not self._rag_initialized:
                await self._initialize_rag_filtering()

            # Create the prompt for execution with credentials
            base_url = self.tripletex_credentials.get(
                "base_url", "https://api.tripletex.no/v2"
            )
            session_token = self.tripletex_credentials.get("session_token", "")

            execution_prompt = ACCOUNTING_COORDINATOR_PROMPT_TEMPLATE.format(
                task_description=self.task_summary,
                base_url=base_url,
                session_token=session_token,
            )

            # Execute task directly using the coordinator's tools (now RAG-filtered)
            logger.info("Executing task with RAG-filtered tools")
            result = await self.agent.run(execution_prompt)

            # Extract result
            if hasattr(result, "data"):
                report = getattr(result, "data")
            elif hasattr(result, "output"):
                report = getattr(result, "output")
            else:
                report = str(result)

            return f"""
# Task Execution Report

## Task
{self.task_summary}

## Result
{report}

## Status
Completed
"""

        except Exception as e:
            logger.error(f"Coordinator error: {e}", exc_info=True)
            raise

    async def _initialize_rag_filtering(self):
        """Initialize RAG filtering for this coordinator execution."""
        try:
            logger.info("Coordinator: Initializing RAG tool filtering...")

            from .rag_tool_manager import create_rag_tool_manager

            # Get embedding provider
            try:
                embedding_provider = get_embedding_provider()
                logger.debug("Coordinator: Using Google embedding provider")
            except Exception as e:
                logger.warning(f"Coordinator: Could not load embedding provider: {e}")
                embedding_provider = None

            # Create RAG manager
            self.rag_manager = create_rag_tool_manager(
                embedding_provider=embedding_provider,
                top_k=100,
            )

            # Initialize RAG manager
            await self.rag_manager.initialize()

            # Index tools from original toolsets
            if self._original_toolsets:
                logger.debug(
                    f"Coordinator: Indexing {len(self._original_toolsets)} toolset(s)..."
                )
                await self.rag_manager.index_toolsets(self._original_toolsets)  # type: ignore

                stats = self.rag_manager.get_statistics()
                logger.info(
                    f"Coordinator: RAG initialized with {stats.get('total_tools', 0)} tools"
                )

            # Create filtered toolsets based on task
            if self._original_toolsets and self.rag_manager:
                filtered_toolsets = await self.rag_manager.create_filtered_toolsets(
                    self._original_toolsets, self.task_summary[:200]
                )  # type: ignore
                self.toolsets = filtered_toolsets

                # Recreate agent with filtered toolsets
                self.agent = Agent(
                    model=self.model,
                    toolsets=filtered_toolsets,
                    system_prompt=ACCOUNTING_COORDINATOR_SYSTEM_INSTRUCTIONS,
                    retries=3,
                )

                logger.info(
                    f"Coordinator: Applied RAG filtering with {len(filtered_toolsets)} filtered toolset(s)"
                )

            self._rag_initialized = True
            return True

        except Exception as e:
            logger.warning(f"Coordinator: RAG filtering initialization failed: {e}")
            self.use_rag_filtering = False
            return False


async def run_accounting_task(
    prompt: str,
    file_content: str = "",
    tripletex_credentials: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the accounting task workflow - simplified direct execution.

    This function executes tasks directly without planning/splitting overhead:
    - Single main agent takes the original prompt
    - Can spawn 1-2 parallel sub-agents if task requires it
    - Direct tool invocation in Tripletex

    Args:
        prompt: The accounting task prompt
        file_content: Extracted content from attached files (PDF text, OCR, etc.)
        tripletex_credentials: Tripletex API credentials (base_url, session_token)

    Returns:
        Dictionary with task results
    """
    logger.info("Starting simplified accounting task workflow...")

    try:
        # Build complete task context including file content
        full_prompt = prompt
        if file_content:
            full_prompt = f"""{prompt}

ATTACHED FILE CONTENT:
{file_content}

Use the information from the attached files above to complete the task."""
            logger.info(f"Added {len(file_content)} chars of file content to prompt")

        # Single step: Execute the task directly
        logger.info("Executing task directly with main agent...")

        # Create main agent to execute the task
        coordinator = CoordinatorAgent(
            task_summary=full_prompt,
            task_type="Direct",
            complexity="single",
            subtasks=[
                {
                    "id": "main",
                    "title": "Execute Task",
                    "description": full_prompt,
                    "dependencies": None,
                }
            ],
            tripletex_credentials=tripletex_credentials or {},
        )

        execution_report = await coordinator.coordinate_execution()

        logger.info("Accounting task workflow completed successfully")

        return {
            "success": True,
            "status": "completed",
            "report": execution_report,
        }

    except Exception as e:
        logger.error(f"Accounting task workflow failed: {e}", exc_info=True)
        return {"success": False, "status": "failed", "error": str(e)}
