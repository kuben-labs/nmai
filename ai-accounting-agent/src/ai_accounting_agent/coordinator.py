"""Coordinator for accounting task execution with sub-agents."""

import json
import asyncio
from typing import Dict, List, Any, Optional
from loguru import logger
from machine_core.core.agent_base import BaseAgent
from .prompts import (
    ACCOUNTING_SUBAGENT_SYSTEM_INSTRUCTIONS,
    ACCOUNTING_SUBAGENT_PROMPT_TEMPLATE,
    ACCOUNTING_COORDINATOR_SYSTEM_INSTRUCTIONS,
    ACCOUNTING_COORDINATOR_PROMPT_TEMPLATE,
)
import os


class AccountingSubAgent(BaseAgent):
    """Sub-agent that executes a specific accounting subtask with RAG filtering."""

    def __init__(
        self,
        subtask_id: str,
        subtask_title: str,
        subtask_description: str,
        main_task: str,
        tripletex_credentials: Dict[str, Any] = None,
    ):
        """
        Initialize a sub-agent for a specific accounting subtask.

        Args:
            subtask_id: Unique identifier for the subtask
            subtask_title: Title of the subtask
            subtask_description: Detailed instructions for this subtask
            main_task: The overall accounting task
            tripletex_credentials: Tripletex API credentials (base_url, session_token)
        """
        # Extract credentials for inclusion in prompt
        # NOTE: These are for transparency; actual API calls use MCP server's .env credentials
        base_url = (tripletex_credentials or {}).get(
            "base_url", "https://api.tripletex.no/v2"
        )
        session_token = (tripletex_credentials or {}).get("session_token", "")

        prompt = ACCOUNTING_SUBAGENT_PROMPT_TEMPLATE.format(
            subtask_id=subtask_id,
            subtask_title=subtask_title,
            subtask_description=subtask_description,
            main_task=main_task,
            base_url=base_url,
            session_token=session_token,
        )

        super().__init__(
            model_name=os.getenv("LLM_MODEL"),
            system_prompt=ACCOUNTING_SUBAGENT_SYSTEM_INSTRUCTIONS,
            mcp_config_path="mcp_accountant.json",
        )
        self.subtask_id = subtask_id
        self.subtask_title = subtask_title
        self._user_prompt = prompt
        self.tripletex_credentials = tripletex_credentials or {}
        self.use_rag_filtering = True  # Enable RAG filtering
        self.rag_manager = None
        self._original_toolsets = list(self.toolsets) if self.toolsets else []
        self._rag_initialized = False

    async def run(self, task: str = None) -> str:
        """Required abstract method implementation."""
        return await self.execute()

    async def execute(self) -> str:
        """
        Execute this subtask using MCP tools with RAG filtering.

        Returns:
            Report of what was executed
        """
        logger.info(f"SubAgent {self.subtask_id} executing: {self.subtask_title}")

        try:
            # Initialize RAG on first execution
            if self.use_rag_filtering and not self._rag_initialized:
                await self._initialize_rag_filtering()

            result = await self.run_query(self._user_prompt)

            # Extract the report
            if hasattr(result, "data"):
                report = result.data
            elif hasattr(result, "output"):
                report = result.output
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
            from model_providers.embeddings import get_embedding_provider

            # Get embedding provider
            try:
                embedding_provider = get_embedding_provider()
                logger.debug(f"SubAgent {self.subtask_id}: Using embedding provider")
            except Exception as e:
                logger.warning(
                    f"SubAgent {self.subtask_id}: Could not load embedding provider: {e}"
                )
                embedding_provider = None

            # Create RAG manager
            self.rag_manager = create_rag_tool_manager(
                embedding_provider=embedding_provider,
                top_k=300,
            )

            # Initialize RAG manager
            await self.rag_manager.initialize()

            # Index tools from original toolsets
            if self._original_toolsets:
                logger.debug(
                    f"SubAgent {self.subtask_id}: Indexing {len(self._original_toolsets)} toolset(s)..."
                )
                await self.rag_manager.index_toolsets(self._original_toolsets)

                stats = self.rag_manager.get_statistics()
                logger.info(
                    f"SubAgent {self.subtask_id}: RAG initialized with {stats.get('total_tools', 0)} tools, "
                    f"will filter to ~300 most relevant per subtask"
                )

            # Create filtered toolsets for this subtask
            if self._original_toolsets and self.rag_manager:
                filtered_toolsets = await self.rag_manager.create_filtered_toolsets(
                    self._original_toolsets, self.subtask_title[:100]
                )
                self.toolsets = filtered_toolsets

                # Recreate agent with filtered toolsets
                from pydantic_ai import Agent

                self.agent = Agent(
                    model=self.model,
                    toolsets=filtered_toolsets,
                    system_prompt=ACCOUNTING_SUBAGENT_SYSTEM_INSTRUCTIONS,
                    retries=self.agent_config.max_tool_retries,
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


class CoordinatorAgent(BaseAgent):
    """Coordinator agent that orchestrates accounting sub-agents."""

    def __init__(
        self,
        task_summary: str,
        task_type: str,
        complexity: str,
        subtasks: List[Dict[str, Any]],
        base_url: str = "http://localhost:8083",
        tripletex_credentials: Dict[str, Any] = None,
        use_rag_filtering: bool = True,
    ):
        """
        Initialize the coordinator agent.

        Args:
            task_summary: Summary of the accounting task
            task_type: Type of task (e.g., "Tier 1", "Tier 2", "Tier 3")
            complexity: Task complexity ("simple" or "complex")
            subtasks: List of subtask dictionaries
            base_url: Tripletex proxy base URL
            tripletex_credentials: Tripletex API credentials (base_url, session_token)
            use_rag_filtering: Whether to use RAG filtering for tool reduction
        """
        subtasks_json = json.dumps(subtasks, indent=2, ensure_ascii=False)

        self.task_summary = task_summary
        self.task_type = task_type
        self.complexity = complexity
        self.subtasks = subtasks
        self.base_url = base_url
        self.tripletex_credentials = tripletex_credentials or {}

        super().__init__(
            model_name=os.getenv("LLM_MODEL"),
            system_prompt=ACCOUNTING_COORDINATOR_SYSTEM_INSTRUCTIONS,
            mcp_config_path="mcp_accountant.json",
        )

        self.use_rag_filtering = use_rag_filtering
        self.rag_manager = None
        self._original_toolsets = list(self.toolsets) if self.toolsets else []
        self._rag_initialized = False

    async def run(self, query: str = None) -> str:
        """Required abstract method implementation."""
        return await self.coordinate_execution()

    async def coordinate_execution(self) -> str:
        """
        Execute the accounting task directly with RAG filtering.

        Returns:
            Summary of execution results
        """
        logger.info(f"Coordinator executing task: {self.task_summary[:100]}")

        try:
            # Initialize RAG filtering on first execution
            if self.use_rag_filtering and not self._rag_initialized:
                await self._initialize_rag_filtering()

            # Create the prompt for execution with credentials
            # NOTE: These credentials are informational - the actual API calls use
            # credentials from the MCP server's .env file, not these values.
            # The LLM cannot use these credentials directly; they're provided for
            # transparency and future extensibility.
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
            result = await self.run_query(execution_prompt)

            # Extract result
            if hasattr(result, "data"):
                report = result.data
            elif hasattr(result, "output"):
                report = result.output
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
            logger.error(f"Coordinator error: {e}")
            raise

    async def _initialize_rag_filtering(self):
        """Initialize RAG filtering for this coordinator execution."""
        try:
            logger.info("Coordinator: Initializing RAG tool filtering...")

            from .rag_tool_manager import create_rag_tool_manager
            from model_providers.embeddings import get_embedding_provider

            # Get embedding provider
            try:
                embedding_provider = get_embedding_provider()
                logger.debug("Coordinator: Using embedding provider")
            except Exception as e:
                logger.warning(f"Coordinator: Could not load embedding provider: {e}")
                embedding_provider = None

            # Create RAG manager
            self.rag_manager = create_rag_tool_manager(
                embedding_provider=embedding_provider,
                top_k=500,  # Return top tools for filtering
            )

            # Initialize RAG manager
            await self.rag_manager.initialize()

            # Index tools from original toolsets
            if self._original_toolsets:
                logger.debug(
                    f"Coordinator: Indexing {len(self._original_toolsets)} toolset(s)..."
                )
                await self.rag_manager.index_toolsets(self._original_toolsets)

                stats = self.rag_manager.get_statistics()
                logger.info(
                    f"Coordinator: RAG initialized with {stats.get('total_tools', 0)} tools"
                )

            # Create filtered toolsets based on task
            if self._original_toolsets and self.rag_manager:
                filtered_toolsets = await self.rag_manager.create_filtered_toolsets(
                    self._original_toolsets, self.task_summary[:200]
                )
                self.toolsets = filtered_toolsets

                # Recreate agent with filtered toolsets
                from pydantic_ai import Agent

                self.agent = Agent(
                    model=self.model,
                    toolsets=filtered_toolsets,
                    system_prompt=ACCOUNTING_COORDINATOR_SYSTEM_INSTRUCTIONS,
                    retries=self.agent_config.max_tool_retries,
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
    files: list = None,
    tripletex_credentials: dict = None,
) -> Dict[str, Any]:
    """
    Run the accounting task workflow - simplified direct execution.

    This function executes tasks directly without planning/splitting overhead:
    - Single main agent takes the original prompt
    - Can spawn 1-2 parallel sub-agents if task requires it
    - Direct tool invocation in Tripletex

    Args:
        prompt: The accounting task prompt
        files: Optional list of attached files
        tripletex_credentials: Tripletex API credentials

    Returns:
        Dictionary with task results
    """
    logger.info("Starting simplified accounting task workflow...")

    try:
        # Single step: Execute the task directly
        logger.info("Executing task directly with main agent...")

        # Create main agent to execute the task
        coordinator = CoordinatorAgent(
            task_summary=prompt,
            task_type="Direct",
            complexity="single",
            subtasks=[
                {
                    "id": "main",
                    "title": "Execute Task",
                    "description": prompt,
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
        logger.error(f"Accounting task workflow failed: {e}")
        return {"success": False, "status": "failed", "error": str(e)}
