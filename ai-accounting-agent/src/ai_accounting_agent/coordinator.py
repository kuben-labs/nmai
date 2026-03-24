"""Coordinator for accounting task execution.

Direct OpenAPI tools architecture (no MCP):
- Fetches OpenAPI spec at startup (cached)
- Generates pydantic-ai Tools from endpoints
- Agent calls Tripletex API directly via httpx
- No separate MCP server process needed

Uses machine-core for model/embedding setup, agent lifecycle,
OpenAPI tools generation, RAG tool filtering, and file processing.
Uses model-providers for LLM and embedding provider resolution.
"""

import asyncio
import base64
import json
import os
import traceback
from typing import Dict, List, Any, Optional, Set

from loguru import logger
from pydantic_ai import Tool
from pydantic_ai.messages import (
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
)
from pydantic_ai._agent_graph import (
    CallToolsNode,
    ModelRequestNode,
)

from model_providers import get_embedding_provider, EmbeddingProviderConfig

from machine_core import (
    AgentCore,
    AgentConfig,
    generate_tools_from_openapi,
    fetch_openapi_spec,
    Embedder,
    VectorStore,
    ToolFilterManager,
)

from .prompts import (
    ACCOUNTING_SYSTEM_INSTRUCTIONS,
    ACCOUNTING_TASK_PROMPT_TEMPLATE,
)

# Module-level caches (protected by narrow init locks for concurrent requests)
_tool_filter_manager = None
_tool_filter_initialized = False
_tool_filter_lock = asyncio.Lock()
_openapi_spec_cache: Optional[Dict[str, Any]] = None
_openapi_spec_lock = asyncio.Lock()

# Shared AgentCore instance (initialized once, rebuilt per request with new tools)
_agent_core = None
_agent_core_lock = asyncio.Lock()


# Essential tools that must ALWAYS be available regardless of RAG filtering.
ESSENTIAL_TOOLS = {
    "Department_search",
    "Department_post",
    "Department_get",
    "Employee_search",
    "Employee_post",
    "Employee_get",
    "EmployeeEmployment_search",
    "EmployeeEmployment_post",
    "EmployeeEmployment_put",
    "EmployeeEmploymentDetails_search",
    "EmployeeEmploymentDetails_post",
    "EmployeeEmploymentDetails_put",
    "EmployeeEmploymentOccupationCode_search",
    "EmployeeStandardTime_post",
    "EmployeeStandardTime_put",
    "EmployeeStandardTime_search",
    "Customer_search",
    "Customer_post",
    "Customer_get",
    "Supplier_search",
    "Supplier_post",
    "Product_search",
    "Product_post",
    "Contact_search",
    "Contact_post",
    "LedgerAccount_search",
    "LedgerAccount_post",
    "Order_search",
    "Order_post",
    "Order_get",
    "OrderOrderline_post",
    "OrderOrderline_get",
    "Invoice_search",
    "Invoice_post",
    "Invoice_get",
    "InvoiceSend_send",
    "InvoicePayment_payment",
    "InvoicePaymentType_search",
    "IncomingInvoiceSearch_search",
    "IncomingInvoice_post",
    "IncomingInvoice_get",
    "IncomingInvoiceAddPayment_addPayment",
    "LedgerVoucher_search",
    "LedgerVoucher_post",
    "LedgerVoucher_get",
    "TravelExpense_search",
    "TravelExpense_post",
    "TravelExpense_delete",
    "TravelExpense_get",
    "TravelExpenseCost_post",
    "TravelExpenseCost_search",
    "TravelExpenseDeliver_deliver",
    "TravelExpenseCreateVouchers_createVouchers",
    "Project_search",
    "Project_post",
    "ProjectParticipant_post",
    "SupplierInvoice_search",
    "SupplierInvoice_get",
    "SupplierInvoiceAddPayment_addPayment",
    "SupplierInvoiceApprove_approve",
    "SalaryType_search",
    "SalaryTransaction_post",
    "Division_search",
    "Division_post",
    "LedgerVatType_search",
    "LedgerAccount_put",
    "Company_get",
    "InvoiceSend_send",
    # Credit notes and reversals
    "InvoiceCreateCreditNote_createCreditNote",
    "LedgerVoucherReverse_reverse",
    # Bank reconciliation
    "BankStatement_search",
    "BankStatement_get",
    "BankStatementImport_importBankStatement",
    "BankStatementTransaction_search",
    "BankReconciliation_search",
    "BankReconciliation_post",
    "BankReconciliation_get",
    "BankReconciliationMatch_post",
    "BankReconciliationMatch_search",
    "BankReconciliationMatchSuggest_suggest",
    "BankReconciliationPaymentType_search",
    # Ledger postings and open posts
    "LedgerPosting_search",
    "LedgerPostingOpenPost_openPost",
    "LedgerPaymentTypeOut_search",
    # Ledger/balance
    "Ledger_search",
    "LedgerVoucherType_search",
    # Activity
    "Activity_search",
    "Activity_post",
    "ProjectProjectActivity_post",
}


def _make_auth_headers(session_token: str) -> Dict[str, str]:
    """Create Tripletex Basic Auth headers: base64("0:" + session_token).

    This is the only Tripletex-specific auth logic. machine-core's
    generate_tools_from_openapi() accepts generic auth_headers.
    """
    credentials = f"0:{session_token}"
    encoded = base64.b64encode(credentials.encode()).decode()
    return {"Authorization": f"Basic {encoded}"}


async def _get_agent_core():
    """Get or create the shared AgentCore instance.

    The AgentCore is created once (initializes model + embedding providers),
    then rebuilt per-request with different tools via rebuild_agent().
    """
    global _agent_core

    if _agent_core is not None:
        return _agent_core

    async with _agent_core_lock:
        if _agent_core is not None:
            return _agent_core

        logger.info("Initializing shared AgentCore (model + embedding providers)...")

        # Configure for high retry count (accounting tasks are complex)
        config = AgentConfig(
            max_tool_retries=30,
            timeout=300.0,
        )

        # Create AgentCore with dynamic tools mode (no MCP).
        # We pass an empty tools list here; it will be rebuilt per-request.
        _agent_core = AgentCore(
            tools=[],
            system_prompt=ACCOUNTING_SYSTEM_INSTRUCTIONS,
            agent_config=config,
        )

        logger.info(
            f"AgentCore initialized: provider_type={_agent_core.provider_type}, "
            f"embedding={'available' if _agent_core.embedding else 'unavailable'}"
        )
        return _agent_core


async def _get_openapi_spec(base_url: str, session_token: str) -> Dict[str, Any]:
    """Get OpenAPI spec, using cache if available.

    We fetch the spec from the sandbox URL (always available) and cache it.
    The spec is the same regardless of which Tripletex instance we connect to.
    Thread-safe: uses _openapi_spec_lock to prevent duplicate fetches.
    """
    global _openapi_spec_cache

    # Fast path: no lock needed if already cached
    if _openapi_spec_cache is not None:
        return _openapi_spec_cache

    async with _openapi_spec_lock:
        # Double-check after acquiring lock (another coroutine may have filled it)
        if _openapi_spec_cache is not None:
            return _openapi_spec_cache

        # Fetch from sandbox (always available, spec is universal)
        sandbox_url = os.getenv("API_URL", "https://kkpqfuj-amager.tripletex.dev/v2")
        sandbox_token = os.getenv("SESSION_TOKEN", "")
        auth_headers = _make_auth_headers(sandbox_token) if sandbox_token else None

        try:
            spec = await fetch_openapi_spec(sandbox_url, auth_headers=auth_headers)
            _openapi_spec_cache = spec
            logger.info(f"Fetched OpenAPI spec: {len(spec.get('paths', {}))} paths")
            return spec
        except Exception as e:
            logger.error(f"Failed to fetch OpenAPI spec: {e}")
            raise


async def _get_tool_filter_manager():
    """Get or create the shared ToolFilterManager (cached across requests).

    Uses machine-core's Embedder + VectorStore + ToolFilterManager.
    Thread-safe: uses _tool_filter_lock to prevent duplicate initialization.
    """
    global _tool_filter_manager, _tool_filter_initialized

    # Fast path: no lock needed if already initialized
    if _tool_filter_manager is not None and _tool_filter_initialized:
        return _tool_filter_manager

    async with _tool_filter_lock:
        # Double-check after acquiring lock
        if _tool_filter_manager is not None and _tool_filter_initialized:
            return _tool_filter_manager

        logger.info("Initializing shared ToolFilterManager...")

        # Use model-providers for embedding resolution.
        try:
            resolved_embedding = get_embedding_provider()
            embedder = Embedder(resolved_embedding)
            logger.debug(f"Using embedding provider: {resolved_embedding.model_name}")
        except Exception as e:
            logger.warning(f"Could not load embedding provider: {e}")
            embedder = Embedder(None)

        vector_store = VectorStore(
            db_path=os.getenv("VECTOR_STORE_PATH", ".vector_store"),
            embedder=embedder,
        )

        _tool_filter_manager = ToolFilterManager(
            embedder=embedder,
            vector_store=vector_store,
        )
        _tool_filter_initialized = True

        logger.info(
            f"ToolFilterManager initialized "
            f"(indexed: {_tool_filter_manager.is_indexed}, "
            f"tools: {_tool_filter_manager.tool_count})"
        )
        return _tool_filter_manager


async def _get_relevant_tool_names(task_prompt: str, spec: Dict[str, Any]) -> Set[str]:
    """Get relevant tool names via RAG + essential tools.

    If the RAG index is empty (first run or cleared), rebuilds it
    from the OpenAPI spec.
    """
    manager = await _get_tool_filter_manager()

    # If no tools indexed, index from OpenAPI spec
    if not manager.is_indexed:
        logger.info("Tool filter index empty, indexing from OpenAPI spec...")
        await manager.index_openapi(spec)

    top_k = int(os.getenv("TOP_K", "200"))
    result = await manager.filter(
        task_prompt, top_k=top_k, essential_tools=ESSENTIAL_TOOLS
    )

    logger.info(
        f"Tool filter: {len(result.names)} tools selected "
        f"(sources: {{{', '.join(f'{k}: {len(v)}' for k, v in result.by_source.items())}}})"
    )
    return result.names


async def run_accounting_task(
    prompt: str,
    file_content: str = "",
    tripletex_credentials: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the accounting task with direct OpenAPI tools (no MCP).

    Uses machine-core's AgentCore for model setup and agent lifecycle.
    Per-request: rebuilds agent with RAG-filtered tools via rebuild_agent().

    1. Fetch OpenAPI spec (cached)
    2. RAG filter to select relevant tools
    3. Generate pydantic-ai Tools from spec
    4. Rebuild agent with those tools
    5. Run agent with step-by-step logging

    Args:
        prompt: The accounting task prompt
        file_content: Extracted content from attached files
        tripletex_credentials: Tripletex API credentials

    Returns:
        Dictionary with task results
    """
    logger.info("Starting accounting task execution...")
    credentials = tripletex_credentials or {}
    base_url = credentials.get("base_url", "")
    session_token = credentials.get("session_token", "")

    if not base_url or not session_token:
        logger.error("Missing base_url or session_token!")
        return {"success": False, "status": "failed", "error": "Missing credentials"}

    # Build full prompt with file content
    full_prompt = prompt
    if file_content:
        full_prompt = f"""{prompt}

ATTACHED FILE CONTENT:
{file_content}

Use the information from the attached files above to complete the task."""
        logger.info(f"Added {len(file_content)} chars of file content to prompt")

    try:
        # 1. Get shared AgentCore (model + embedding initialized once)
        agent_core = await _get_agent_core()

        # 2. Get OpenAPI spec (cached)
        spec = await _get_openapi_spec(base_url, session_token)

        # 3. Get relevant tool names via RAG
        relevant_names = await _get_relevant_tool_names(full_prompt, spec)

        # 4. Generate tools from spec (filtered to relevant ones)
        # Construct Tripletex auth headers (this is the only Tripletex-specific part)
        auth_headers = _make_auth_headers(session_token)
        tools = generate_tools_from_openapi(
            spec=spec,
            base_url=base_url,
            auth_headers=auth_headers,
            tool_filter=relevant_names,
        )

        if not tools:
            logger.error("No tools generated from OpenAPI spec!")
            return {"success": False, "status": "failed", "error": "No tools available"}

        logger.info(f"Generated {len(tools)} direct API tools (no MCP)")

        # 5. Build execution prompt
        execution_prompt = ACCOUNTING_TASK_PROMPT_TEMPLATE.format(
            task_description=full_prompt,
        )

        # 6. Rebuild agent with per-request tools (reuses model from AgentCore)
        agent_core.rebuild_agent(tools=tools, retries=30)

        # 7. Execute with step-by-step logging via agent.iter()
        logger.info("Executing task with direct API tools...")
        step_num = 0
        try:
            async with agent_core.agent.iter(execution_prompt) as agent_run:
                async for node in agent_run:
                    step_num += 1
                    node_type = type(node).__name__
                    logger.info(f"[Step {step_num}] {node_type}")

                    # Log tool calls
                    if isinstance(node, CallToolsNode):
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
                        except Exception:
                            pass

                    # Log tool results and retries
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
                                            f"[Step {step_num}] RETRY: {str(part.content)[:300]}"
                                        )
                        except Exception:
                            pass

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
        logger.info(f"Agent final output: {str(report)[:1000]}")

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
