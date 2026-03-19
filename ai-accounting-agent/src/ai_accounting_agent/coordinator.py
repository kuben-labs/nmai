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
    """Sub-agent that executes a specific accounting subtask."""

    def __init__(
        self,
        subtask_id: str,
        subtask_title: str,
        subtask_description: str,
        main_task: str,
    ):
        """
        Initialize a sub-agent for a specific accounting subtask.

        Args:
            subtask_id: Unique identifier for the subtask
            subtask_title: Title of the subtask
            subtask_description: Detailed instructions for this subtask
            main_task: The overall accounting task
        """
        prompt = ACCOUNTING_SUBAGENT_PROMPT_TEMPLATE.format(
            subtask_id=subtask_id,
            subtask_title=subtask_title,
            subtask_description=subtask_description,
            main_task=main_task,
        )

        super().__init__(
            model_name=os.getenv("LLM_MODEL"),
            system_prompt=ACCOUNTING_SUBAGENT_SYSTEM_INSTRUCTIONS,
            mcp_config_path="mcp_accountant.json",
        )

        self.subtask_id = subtask_id
        self.subtask_title = subtask_title
        self._user_prompt = prompt

    async def run(self, task: str = None) -> str:
        """Required abstract method implementation."""
        return await self.execute()

    async def execute(self) -> str:
        """
        Execute this subtask using MCP tools.

        Returns:
            Report of what was executed
        """
        logger.info(f"SubAgent {self.subtask_id} executing: {self.subtask_title}")

        try:
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


class CoordinatorAgent(BaseAgent):
    """Coordinator agent that orchestrates accounting sub-agents."""

    def __init__(
        self,
        task_summary: str,
        task_type: str,
        complexity: str,
        subtasks: List[Dict[str, Any]],
        base_url: str = "http://localhost:8083",
    ):
        """
        Initialize the coordinator agent.

        Args:
            task_summary: Summary of the accounting task
            task_type: Type of task (e.g., "Tier 1", "Tier 2", "Tier 3")
            complexity: Task complexity ("simple" or "complex")
            subtasks: List of subtask dictionaries
            base_url: Tripletex proxy base URL
        """
        subtasks_json = json.dumps(subtasks, indent=2, ensure_ascii=False)

        self.task_summary = task_summary
        self.task_type = task_type
        self.complexity = complexity
        self.subtasks = subtasks
        self.base_url = base_url

        super().__init__(
            model_name=os.getenv("LLM_MODEL"),
            system_prompt=ACCOUNTING_COORDINATOR_SYSTEM_INSTRUCTIONS,
            mcp_config_path="mcp_accountant.json",
        )

    async def run(self, query: str = None) -> str:
        """Required abstract method implementation."""
        return await self.coordinate_execution()

    async def coordinate_execution(self) -> str:
        """
        Coordinate all sub-agents and verify results.

        Returns:
            Summary of execution results
        """
        logger.info(f"Coordinator starting execution of {len(self.subtasks)} subtasks")
        logger.info(f"Task type: {self.task_type}, Complexity: {self.complexity}")

        try:
            # For simple tasks, execute directly
            if self.complexity.lower() == "simple":
                logger.info("Simple task - executing directly without splitting")
                return await self._execute_simple_task()

            # For complex tasks, coordinate sub-agents
            logger.info("Complex task - coordinating sub-agents")
            return await self._execute_complex_task()

        except Exception as e:
            logger.error(f"Coordinator error: {e}")
            raise

    async def _execute_simple_task(self) -> str:
        """Execute a simple task (1-2 API calls)."""
        try:
            # For simple tasks, we still use a single sub-agent
            if self.subtasks:
                first_subtask = self.subtasks[0]
                subagent = AccountingSubAgent(
                    subtask_id=first_subtask.get("id", "main"),
                    subtask_title=first_subtask.get("title", "Main Task"),
                    subtask_description=first_subtask.get("description", ""),
                    main_task=self.task_summary,
                )

                logger.info("Executing simple task...")
                result = await subagent.execute()

                return f"""
# Task Execution Report

## Task Summary
{self.task_summary}

## Result
{result}

## Status
Completed
"""

        except Exception as e:
            logger.error(f"Error executing simple task: {e}")
            raise

    async def _execute_complex_task(self) -> str:
        """Execute a complex task with multiple parallel sub-agents."""
        try:
            # Create sub-agents for each subtask
            subagents = []
            for subtask in self.subtasks:
                subagent = AccountingSubAgent(
                    subtask_id=subtask.get("id", "unknown"),
                    subtask_title=subtask.get("title", "Subtask"),
                    subtask_description=subtask.get("description", ""),
                    main_task=self.task_summary,
                )
                subagents.append((subtask, subagent))

            # Execute sub-agents
            # Note: In a real system, we'd respect dependencies, but for now we run all in parallel
            logger.info(f"Starting {len(subagents)} sub-agents in parallel...")

            tasks = [agent.execute() for _, agent in subagents]
            sub_reports = await asyncio.gather(*tasks)

            # Compile results
            logger.info("Coordinator compiling execution reports...")

            execution_summary = "## Sub-Task Execution Results\n\n"
            for (subtask, _), report in zip(subagents, sub_reports):
                execution_summary += f"### {subtask.get('id', 'unknown')}: {subtask.get('title', 'Task')}\n"
                execution_summary += f"{report}\n\n"

            # Create final verification summary
            final_report = f"""
# Accounting Task Execution Report

## Task Summary
{self.task_summary}

## Classification
- Type: {self.task_type}
- Complexity: {self.complexity}

## Execution Status

{execution_summary}

## Overall Status
Task execution coordinated and completed.
"""

            logger.info("Coordinator completed execution")
            return final_report

        except Exception as e:
            logger.error(f"Error executing complex task: {e}")
            raise


async def run_accounting_task(
    prompt: str,
    files: list = None,
    tripletex_credentials: dict = None,
    planner_class=None,
    splitter_class=None,
) -> Dict[str, Any]:
    """
    Run the complete accounting task workflow.

    This function orchestrates the full workflow:
    1. Plan the task
    2. Split into subtasks (if complex)
    3. Coordinate execution

    Args:
        prompt: The accounting task prompt
        files: Optional list of attached files
        tripletex_credentials: Tripletex API credentials
        planner_class: The planner agent class to use
        splitter_class: The task splitter class to use

    Returns:
        Dictionary with task results
    """
    # Import here to avoid circular imports
    if planner_class is None:
        from .planner import AccountingPlanner

        planner_class = AccountingPlanner

    if splitter_class is None:
        from .task_splitter import TaskSplitter

        splitter_class = TaskSplitter

    logger.info("Starting accounting task workflow...")

    try:
        # Step 1: Generate plan
        logger.info("Step 1: Planning accounting task...")
        planner = planner_class()
        task_plan = await planner.plan_task(
            prompt=prompt,
            files=files or [],
            tripletex_credentials=tripletex_credentials or {},
        )
        logger.debug(f"Task plan:\n{task_plan}")

        # Step 2: Determine if task needs splitting
        # For now, we'll always try to split if it's complex
        logger.info("Step 2: Analyzing task complexity...")

        # Check if plan mentions "complex"
        is_complex = "complex" in task_plan.lower()

        subtasks = []
        if is_complex:
            logger.info("Complex task detected - splitting into subtasks...")
            splitter = splitter_class()
            subtasks = await splitter.split_into_subtasks(task_plan)
            logger.info(f"Task split into {len(subtasks)} subtasks")
        else:
            logger.info("Simple task detected - creating single subtask...")
            subtasks = [
                {
                    "id": "main",
                    "title": "Main Task",
                    "description": task_plan,
                    "dependencies": None,
                }
            ]

        # Step 3: Coordinate execution
        logger.info("Step 3: Coordinating task execution...")
        task_type = "Tier 1"  # Default, would be extracted from plan in real system
        complexity = "complex" if is_complex else "simple"

        coordinator = CoordinatorAgent(
            task_summary=prompt,
            task_type=task_type,
            complexity=complexity,
            subtasks=subtasks,
        )
        execution_report = await coordinator.coordinate_execution()

        logger.info("Accounting task workflow completed successfully")

        return {
            "success": True,
            "status": "completed",
            "plan": task_plan,
            "subtasks": subtasks,
            "report": execution_report,
        }

    except Exception as e:
        logger.error(f"Accounting task workflow failed: {e}")
        return {"success": False, "status": "failed", "error": str(e)}
