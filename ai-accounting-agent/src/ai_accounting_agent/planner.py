"""Accounting task planner for Tripletex automation."""

import os
from loguru import logger
from machine_core.core.agent_base import BaseAgent
from .prompts import (
    ACCOUNTING_PLANNER_SYSTEM_INSTRUCTIONS,
    ACCOUNTING_PLANNER_PROMPT_TEMPLATE,
)
from typing import Optional


class AccountingPlanner(BaseAgent):
    """Planner agent that analyzes accounting tasks and creates execution plans."""

    def __init__(self):
        """Initialize the accounting planner agent."""
        super().__init__(
            model_name=os.getenv("LLM_MODEL"),
            system_prompt=ACCOUNTING_PLANNER_SYSTEM_INSTRUCTIONS,
            mcp_config_path="mcp_accountant.json",
        )

    async def run(self, task: str) -> str:
        """Required abstract method implementation."""
        return await self.plan_task(task, [], {}, "http://localhost:8083")

    async def plan_task(
        self,
        prompt: str,
        files: list = None,
        tripletex_credentials: dict = None,
        base_url: str = "http://localhost:8083",
    ) -> str:
        """
        Analyze an accounting task and create a detailed execution plan.

        Args:
            prompt: The task description (in any language)
            files: Optional list of attached files with metadata
            tripletex_credentials: Dict with 'base_url' and 'session_token'
            base_url: Tripletex proxy base URL

        Returns:
            A detailed plan as a string
        """
        logger.info(f"Planning accounting task: {prompt[:100]}...")

        if files is None:
            files = []
        if tripletex_credentials is None:
            tripletex_credentials = {}

        # Format file information
        file_info = ""
        if files:
            file_list = []
            for f in files:
                file_list.append(
                    f"  - {f.get('filename', 'unknown')}: {f.get('mime_type', 'unknown')}"
                )
            file_info = "\n".join(file_list)

        # Create the planning prompt
        planning_prompt = ACCOUNTING_PLANNER_PROMPT_TEMPLATE.format(
            prompt=prompt,
            file_count=len(files),
            file_info=file_info if files else "None",
            base_url=tripletex_credentials.get("base_url", base_url),
        )

        try:
            result = await self.run_query(planning_prompt)

            # Extract the plan
            if hasattr(result, "data"):
                plan = result.data
            elif hasattr(result, "output"):
                plan = result.output
            else:
                plan = str(result)

            logger.info("Accounting task plan generated successfully")
            logger.debug(f"Plan:\n{plan}")
            return plan

        except Exception as e:
            logger.error(f"Error planning accounting task: {e}")
            raise
