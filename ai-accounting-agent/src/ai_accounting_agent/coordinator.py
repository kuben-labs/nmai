"""Coordinator for accounting task execution.

Direct OpenAPI tools architecture (no MCP):
- Fetches OpenAPI spec at startup (cached)
- Generates pydantic-ai Tools from endpoints
- Agent calls Tripletex API directly via httpx
- No separate MCP server process needed
"""

import asyncio
import json
import os
import traceback
from typing import Dict, List, Any, Optional, Set

from loguru import logger
from pydantic_ai import Agent, Tool
from pydantic_ai.messages import (
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

from .llm import get_google_model
from .embeddings import get_embedding_provider
from .prompts import (
    ACCOUNTING_SYSTEM_INSTRUCTIONS,
    ACCOUNTING_TASK_PROMPT_TEMPLATE,
)
from .rag_tool_manager import create_rag_tool_manager
from .openapi_tools import generate_tools_from_openapi, fetch_openapi_spec

# Module-level caches (protected by narrow init locks for concurrent requests)
_rag_manager = None
_rag_initialized = False
_rag_init_lock = asyncio.Lock()
_openapi_spec_cache: Optional[Dict[str, Any]] = None
_openapi_spec_lock = asyncio.Lock()
_tools_cache: Dict[str, List[Tool]] = {}  # keyed by base_url+token hash


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

        try:
            spec = await fetch_openapi_spec(sandbox_url, sandbox_token)
            _openapi_spec_cache = spec
            logger.info(f"Fetched OpenAPI spec: {len(spec.get('paths', {}))} paths")
            return spec
        except Exception as e:
            logger.error(f"Failed to fetch OpenAPI spec: {e}")
            raise


async def _get_rag_manager():
    """Get or create the shared RAG manager (cached across requests).

    Thread-safe: uses _rag_init_lock to prevent duplicate initialization.
    """
    global _rag_manager, _rag_initialized

    # Fast path: no lock needed if already initialized
    if _rag_manager is not None and _rag_initialized:
        return _rag_manager

    async with _rag_init_lock:
        # Double-check after acquiring lock
        if _rag_manager is not None and _rag_initialized:
            return _rag_manager

        logger.info("Initializing shared RAG tool manager...")

        try:
            embedding_provider = get_embedding_provider()
            logger.debug("Using Google embedding provider")
        except Exception as e:
            logger.warning(f"Could not load embedding provider: {e}")
            embedding_provider = None

        top_k = int(os.getenv("TOP_K", "200"))
        _rag_manager = create_rag_tool_manager(
            embedding_provider=embedding_provider,
            top_k=top_k,
        )
        await _rag_manager.initialize()
        _rag_initialized = True

        return _rag_manager


async def _get_relevant_tool_names(task_prompt: str, spec: Dict[str, Any]) -> Set[str]:
    """Get relevant tool names via RAG + essential tools.

    If the RAG index is empty (first run or cleared), rebuilds it
    from the OpenAPI spec.
    """
    rag_manager = await _get_rag_manager()

    # If no tools indexed, rebuild from OpenAPI spec
    stats = rag_manager.get_statistics()
    if stats.get("total_tools", 0) == 0:
        logger.info("RAG index empty, rebuilding from OpenAPI spec...")
        from .rag_tool_filter import index_openapi_tools

        if rag_manager.vector_store and rag_manager.embedder:
            await index_openapi_tools(
                spec=spec,
                vector_store=rag_manager.vector_store,
                embedder=rag_manager.embedder,
            )
        else:
            logger.warning(
                "No vector store or embedder, returning essential tools only"
            )
            return ESSENTIAL_TOOLS

    top_k = int(os.getenv("TOP_K", "200"))
    relevant_names = await rag_manager.get_relevant_names(task_prompt, top_k=top_k)
    relevant_names = relevant_names | ESSENTIAL_TOOLS

    logger.info(
        f"Tool filter: {len(relevant_names)} tools selected "
        f"(RAG: 200 + essential: {len(ESSENTIAL_TOOLS)} with overlap)"
    )
    return relevant_names


async def run_accounting_task(
    prompt: str,
    file_content: str = "",
    tripletex_credentials: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the accounting task with direct OpenAPI tools (no MCP).

    1. Fetch OpenAPI spec (cached)
    2. RAG filter to select relevant tools
    3. Generate pydantic-ai Tools from spec
    4. Run agent with those tools

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
        # 1. Get OpenAPI spec (cached)
        spec = await _get_openapi_spec(base_url, session_token)

        # 2. Get relevant tool names via RAG
        relevant_names = await _get_relevant_tool_names(full_prompt, spec)

        # 3. Generate tools from spec (filtered to relevant ones)
        tools = generate_tools_from_openapi(
            spec=spec,
            base_url=base_url,
            session_token=session_token,
            tool_filter=relevant_names,
        )

        if not tools:
            logger.error("No tools generated from OpenAPI spec!")
            return {"success": False, "status": "failed", "error": "No tools available"}

        logger.info(f"Generated {len(tools)} direct API tools (no MCP)")

        # 4. Build execution prompt
        execution_prompt = ACCOUNTING_TASK_PROMPT_TEMPLATE.format(
            task_description=full_prompt,
        )

        # 5. Create and run agent
        model = get_google_model()
        agent = Agent(
            model=model,
            tools=tools,
            system_prompt=ACCOUNTING_SYSTEM_INSTRUCTIONS,
            retries=30,
        )

        logger.info("Executing task with direct API tools...")
        step_num = 0
        try:
            async with agent.iter(execution_prompt) as agent_run:
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
