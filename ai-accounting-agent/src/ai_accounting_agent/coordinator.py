"""Coordinator for accounting task execution.

Simplified single-agent architecture:
- One agent with RAG-filtered MCP tools (full schemas preserved)
- Uses pydantic-ai's native FilteredToolset for tool filtering
- No sub-agent hierarchy or planning overhead
"""

import json
import os
import traceback
from typing import Dict, List, Any, Optional, Set

from loguru import logger
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
)
from pydantic_ai._agent_graph import (
    CallToolsNode,
    ModelRequestNode,
    UserPromptNode,
)
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.mcp import MCPServerStreamableHTTP

from .llm import get_google_model
from .embeddings import get_embedding_provider
from .prompts import (
    ACCOUNTING_SYSTEM_INSTRUCTIONS,
    ACCOUNTING_TASK_PROMPT_TEMPLATE,
)
from .rag_tool_manager import create_rag_tool_manager

# Module-level cache for the RAG manager (shared across requests)
_rag_manager = None
_rag_initialized = False


def load_mcp_config(config_path: str = "mcp_accountant.json") -> List[Dict[str, Any]]:
    """Load MCP server configurations from a JSON file."""
    try:
        with open(config_path, "r") as f:
            config_data = json.load(f)

        servers = []
        for server_name, server_config in config_data.get("servers", {}).items():
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


def get_mcp_toolset(timeout: float = 300.0) -> MCPServerStreamableHTTP:
    """Create the MCP toolset connection.

    Args:
        timeout: Timeout for MCP connections in seconds

    Returns:
        Configured MCP toolset
    """
    server_configs = load_mcp_config()
    if not server_configs:
        raise ValueError("No MCP servers configured")

    config = server_configs[0]  # We only have one MCP server
    url = config.get("url", "")

    server = MCPServerStreamableHTTP(
        url=url,
        timeout=timeout,
        max_retries=5,
    )
    logger.info(f"MCP toolset configured: {url}")
    return server


async def _get_rag_manager():
    """Get or create the shared RAG manager (cached across requests)."""
    global _rag_manager, _rag_initialized

    if _rag_manager is not None and _rag_initialized:
        return _rag_manager

    logger.info("Initializing shared RAG tool manager...")

    try:
        embedding_provider = get_embedding_provider()
        logger.debug("Using Google embedding provider")
    except Exception as e:
        logger.warning(f"Could not load embedding provider: {e}")
        embedding_provider = None

    _rag_manager = create_rag_tool_manager(
        embedding_provider=embedding_provider,
        top_k=100,
    )
    await _rag_manager.initialize()
    _rag_initialized = True

    return _rag_manager


async def _index_and_get_relevant_tools(
    mcp_toolset: MCPServerStreamableHTTP,
    task_prompt: str,
    top_k: int = 100,
) -> Set[str]:
    """Index MCP tools (if needed) and get relevant tool names for a task.

    Args:
        mcp_toolset: The MCP toolset to index
        task_prompt: The task prompt to find relevant tools for
        top_k: Number of relevant tools to return

    Returns:
        Set of relevant tool names
    """
    rag_manager = await _get_rag_manager()

    # Index tools (skips if already indexed)
    await rag_manager.index_toolsets([mcp_toolset])

    stats = rag_manager.get_statistics()
    logger.info(f"RAG index: {stats.get('total_tools', 0)} tools indexed")

    # Get relevant tool names
    relevant_names = await rag_manager.get_relevant_names(task_prompt, top_k=top_k)
    logger.info(f"RAG filter: {len(relevant_names)} tools relevant for this task")

    return relevant_names


def _simplify_schema_for_gemini(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Simplify a JSON schema so Gemini can process it.

    Gemini rejects schemas that are too complex (nested $ref, deep $defs, etc.).
    This function resolves $ref references inline and limits nesting depth,
    while preserving top-level property names, types, and descriptions.

    The key improvement over the old sanitize_schema: instead of replacing
    $ref with "Complex object", we resolve the reference and include the
    actual property names and types from the referenced definition.
    """
    # Extract $defs for inline resolution (check multiple locations)
    defs = schema.get("$defs", {})
    if not defs:
        defs = schema.get("definitions", {})

    def _resolve(s: Any, depth: int = 0, seen: Optional[set] = None) -> Any:
        if seen is None:
            seen = set()

        if not isinstance(s, dict):
            return s

        # Resolve $ref by inlining the referenced definition
        if "$ref" in s:
            ref_path = s["$ref"]  # e.g. "#/$defs/SomeType" or "#/definitions/X"
            ref_name = ref_path.rsplit("/", 1)[-1] if "/" in ref_path else ref_path

            # Prevent infinite recursion on circular refs
            if ref_name in seen:
                return {"type": "object"}

            if ref_name in defs:
                seen = seen | {ref_name}
                return _resolve(defs[ref_name], depth, seen)
            else:
                # $ref not found in $defs — remove it and return generic object
                return {"type": "object"}

        # At depth > 3, flatten complex nested types to simple objects
        if depth > 3:
            schema_type = s.get("type")
            if schema_type == "object":
                return {"type": "object"}
            elif schema_type == "array":
                return {"type": "array", "items": {"type": "object"}}
            # For simple types (string, integer, etc.) keep as-is
            # But strip any remaining $ref or $defs in the value
            return {
                k: v for k, v in s.items() if k not in ("$ref", "$defs", "definitions")
            }

        # Recursively process dict values
        result = {}
        for k, v in s.items():
            if k in ("$defs", "definitions"):
                continue  # Strip $defs from output (already resolved inline)
            elif isinstance(v, dict):
                result[k] = _resolve(v, depth + 1, seen)
            elif isinstance(v, list):
                result[k] = [
                    _resolve(item, depth + 1, seen) if isinstance(item, dict) else item
                    for item in v
                ]
            else:
                result[k] = v

        return result

    simplified = _resolve(schema)

    # Final safety pass: ensure absolutely no $ref remains anywhere
    # (Gemini will reject the entire request if even one $ref is undefined)
    simplified_str = json.dumps(simplified, default=str)
    if "$ref" in simplified_str:
        logger.warning(
            "Schema still contains $ref after simplification, doing aggressive cleanup"
        )
        simplified = json.loads(simplified_str.replace('"$ref"', '"_removed_ref"'))

    return simplified


async def run_accounting_task(
    prompt: str,
    file_content: str = "",
    tripletex_credentials: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the accounting task - single agent with filtered MCP tools.

    This function:
    1. Creates MCP toolset connection
    2. Uses RAG to find relevant tools (embedding search)
    3. Creates a FilteredToolset that preserves full parameter schemas
    4. Runs a single agent to execute the task

    Args:
        prompt: The accounting task prompt
        file_content: Extracted content from attached files
        tripletex_credentials: Tripletex API credentials

    Returns:
        Dictionary with task results
    """
    logger.info("Starting accounting task execution...")
    credentials = tripletex_credentials or {}

    # Build full prompt with file content
    full_prompt = prompt
    if file_content:
        full_prompt = f"""{prompt}

ATTACHED FILE CONTENT:
{file_content}

Use the information from the attached files above to complete the task."""
        logger.info(f"Added {len(file_content)} chars of file content to prompt")

    # Create MCP toolset
    mcp_toolset = get_mcp_toolset()

    try:
        # Get relevant tool names via RAG search
        relevant_names = await _index_and_get_relevant_tools(
            mcp_toolset, full_prompt, top_k=100
        )

        # Create filtered toolset using pydantic-ai's native FilteredToolset
        # Then apply schema simplification via prepared() so Gemini can handle it.
        # This resolves $ref inline (preserving property names/types) instead of
        # replacing them with "Complex object" like the old approach did.
        from dataclasses import replace as dc_replace
        from pydantic_ai._run_context import RunContext as PydanticRunContext

        async def _simplify_tool_defs(
            ctx: PydanticRunContext, tool_defs: list[ToolDefinition]
        ) -> list[ToolDefinition]:
            simplified = []
            for td in tool_defs:
                if td.parameters_json_schema:
                    new_schema = _simplify_schema_for_gemini(td.parameters_json_schema)
                    simplified.append(dc_replace(td, parameters_json_schema=new_schema))
                else:
                    simplified.append(td)
            return simplified

        filtered_toolset = mcp_toolset.filtered(
            lambda ctx, tool_def: tool_def.name in relevant_names
        ).prepared(_simplify_tool_defs)

        logger.info(
            f"Created FilteredToolset: {len(relevant_names)} tools "
            f"(schemas simplified with $ref resolution for Gemini)"
        )

        # Build the execution prompt with credentials
        base_url = credentials.get("base_url", "https://api.tripletex.no/v2")
        session_token = credentials.get("session_token", "")

        execution_prompt = ACCOUNTING_TASK_PROMPT_TEMPLATE.format(
            task_description=full_prompt,
            base_url=base_url,
            session_token=session_token,
        )

        # Create and run the agent
        model = get_google_model()
        agent = Agent(
            model=model,
            toolsets=[filtered_toolset],
            system_prompt=ACCOUNTING_SYSTEM_INSTRUCTIONS,
            retries=30,
        )

        logger.info("Executing task with agent (full tool schemas preserved)...")
        step_num = 0
        try:
            # Use iter() for step-by-step execution with logging
            async with agent.iter(execution_prompt) as agent_run:
                async for node in agent_run:
                    step_num += 1
                    node_type = type(node).__name__
                    logger.info(f"[Step {step_num}] {node_type}")

                    # Log tool calls from CallToolsNode
                    if isinstance(node, CallToolsNode):
                        # The node has the model response with tool calls
                        try:
                            resp = getattr(node, "model_response", None)
                            if resp:
                                for part in getattr(resp, "parts", []):
                                    if isinstance(part, ToolCallPart):
                                        args_str = json.dumps(part.args, default=str)[
                                            :500
                                        ]
                                        logger.info(
                                            f"[Step {step_num}] TOOL CALL: "
                                            f"{part.tool_name}({args_str})"
                                        )
                        except Exception as log_err:
                            logger.debug(
                                f"[Step {step_num}] Could not log tool call: {log_err}"
                            )

                    # Log model request nodes (contain tool returns)
                    if isinstance(node, ModelRequestNode):
                        try:
                            request = getattr(node, "request", None)
                            if request:
                                for part in getattr(request, "parts", []):
                                    if isinstance(part, ToolReturnPart):
                                        content_str = str(part.content)[:500]
                                        logger.info(
                                            f"[Step {step_num}] TOOL RESULT "
                                            f"({part.tool_name}): {content_str}"
                                        )
                                    elif isinstance(part, RetryPromptPart):
                                        logger.warning(
                                            f"[Step {step_num}] RETRY: {part.content}"
                                        )
                        except Exception as log_err:
                            logger.debug(
                                f"[Step {step_num}] Could not log request: {log_err}"
                            )

                    # Log any node attrs for debugging
                    try:
                        node_attrs = [
                            a
                            for a in dir(node)
                            if not a.startswith("_") and a not in ("run",)
                        ]
                        logger.debug(f"[Step {step_num}] Node attrs: {node_attrs}")
                    except Exception:
                        pass

                # Get final result
                result = agent_run.result
                logger.info(f"Agent completed after {step_num} steps")

        except Exception as run_error:
            logger.error(
                f"Agent execution failed at step {step_num}: "
                f"{type(run_error).__name__}: {run_error}",
            )
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise

        # Extract result
        if hasattr(result, "data"):
            report = getattr(result, "data")
        elif hasattr(result, "output"):
            report = getattr(result, "output")
        else:
            report = str(result)

        logger.info("Accounting task completed successfully")

        return {
            "success": True,
            "status": "completed",
            "report": report,
        }

    except Exception as e:
        logger.error(
            f"Accounting task failed: {type(e).__name__}: {e}",
        )
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        return {"success": False, "status": "failed", "error": str(e)}
