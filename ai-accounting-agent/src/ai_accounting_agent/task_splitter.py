"""Task splitter for breaking down complex accounting tasks."""

import os
import json
from loguru import logger
from machine_core.core.agent_base import BaseAgent
from .prompts import (
    ACCOUNTING_TASK_SPLITTER_SYSTEM_INSTRUCTIONS,
    ACCOUNTING_TASK_SPLITTER_PROMPT_TEMPLATE,
)
from typing import List, Dict, Any
from pydantic import BaseModel, Field


class Subtask(BaseModel):
    """An accounting subtask."""

    id: str = Field(
        ...,
        description="Short identifier for the subtask (e.g. 'create_employee', 'create_invoice')",
    )
    title: str = Field(..., description="Short descriptive title of the subtask")
    description: str = Field(
        ..., description="Clear, detailed instructions for executing this subtask"
    )
    dependencies: List[str] = Field(
        default_factory=list, description="List of subtask IDs this task depends on"
    )


class SubtaskList(BaseModel):
    """List of accounting subtasks."""

    subtasks: List[Subtask] = Field(
        ..., description="List of subtasks that together accomplish the accounting task"
    )


class TaskSplitter(BaseAgent):
    """Agent that splits complex accounting tasks into executable subtasks."""

    def __init__(self):
        super().__init__(
            model_name=os.getenv("LLM_MODEL"),
            system_prompt=ACCOUNTING_TASK_SPLITTER_SYSTEM_INSTRUCTIONS,
            mcp_config_path="mcp_accountant.json",
        )

    async def run(self, planning_context: str) -> List[Dict]:
        """Required abstract method implementation."""
        return await self.split_into_subtasks(planning_context)

    async def split_into_subtasks(self, planning_context: str) -> List[Dict[str, Any]]:
        """
        Split a complex accounting plan into discrete subtasks.

        Args:
            planning_context: The planning context from the planner agent

        Returns:
            List of subtask dictionaries with id, title, description, and dependencies
        """
        logger.info("Splitting accounting task into subtasks...")

        try:
            # Create the splitting prompt
            query = ACCOUNTING_TASK_SPLITTER_PROMPT_TEMPLATE.format(
                planning_context=planning_context
            )

            result = await self.run_query(query)

            # Extract the result
            if hasattr(result, "data"):
                content = result.data
            elif hasattr(result, "output"):
                content = result.output
            else:
                content = str(result)

            # Parse JSON from the response
            subtasks = self._parse_subtasks(content)

            logger.info(f"Generated {len(subtasks)} subtasks:")
            for task in subtasks:
                logger.info(f"  - {task['id']}: {task['title']}")

            return subtasks

        except Exception as e:
            logger.error(f"Error splitting accounting task: {e}")
            raise

    def _parse_subtasks(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Parse JSON subtasks from the response.

        Args:
            response_text: The raw response from the LLM

        Returns:
            List of parsed subtasks
        """
        try:
            # Clean up the response
            content = response_text.strip()

            # Remove markdown code blocks if present
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                content = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                content = content[start:end].strip()

            # Parse the JSON
            parsed = json.loads(content)
            subtasks = parsed.get("subtasks", [])

            # Validate structure
            for subtask in subtasks:
                if not all(key in subtask for key in ["id", "title", "description"]):
                    raise ValueError(f"Subtask missing required fields: {subtask}")
                # Ensure dependencies is a list
                if "dependencies" not in subtask:
                    subtask["dependencies"] = None

            return subtasks

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response was: {response_text}")
            # Fallback: create a single comprehensive task
            logger.warning("Using fallback single subtask")
            return [
                {
                    "id": "comprehensive",
                    "title": "Execute Accounting Task",
                    "description": response_text,
                    "dependencies": None,
                }
            ]
        except Exception as e:
            logger.error(f"Error parsing subtasks: {e}")
            raise
