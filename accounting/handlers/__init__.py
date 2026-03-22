"""Deterministic task handlers for common accounting tasks.

For recognized task patterns, these handlers:
1. Use a single LLM call to extract structured data from the prompt
2. Execute the correct API calls in Python (no LLM guessing)
"""

import base64
import json
import logging

from ._detection import detect_task_type
from ._schemas import SCHEMAS
from ._helpers import extract_structured

from .supplier_invoice import handle_supplier_invoice
from .employee import handle_employee_creation
from .customer_invoice import handle_customer_invoice
from .credit_note import handle_credit_note
from .travel_expense import handle_travel_expense
from .salary import handle_salary
from .project_lifecycle import handle_project_lifecycle
from .ledger_analysis import handle_ledger_analysis
from .bank_reconciliation import handle_bank_reconciliation
from .year_end_closing import handle_year_end_closing
from .voucher_correction import handle_voucher_correction

logger = logging.getLogger(__name__)

HANDLERS = {
    "supplier_invoice": handle_supplier_invoice,
    "employee_creation": handle_employee_creation,
    "customer_invoice": handle_customer_invoice,
    "credit_note": handle_credit_note,
    "travel_expense": handle_travel_expense,
    "salary": handle_salary,
    "project_lifecycle": handle_project_lifecycle,
    "ledger_analysis": handle_ledger_analysis,
    "bank_reconciliation": handle_bank_reconciliation,
    "year_end_closing": handle_year_end_closing,
    "voucher_correction": handle_voucher_correction,
}


def try_handle(client, model, prompt, files, tripletex) -> dict | None:
    """Try to handle the task deterministically.

    Returns {"status": "completed"} on success, or None to fall back to LLM consensus.
    """
    task_type = detect_task_type(prompt)
    if not task_type or task_type not in HANDLERS:
        logger.info(f"No deterministic handler for task (detected: {task_type})")
        return None

    logger.info(f"Deterministic handler: {task_type}")

    schema = SCHEMAS[task_type]
    params = extract_structured(client, model, prompt, files, schema)
    if not params:
        logger.warning(f"LLM extraction failed for {task_type} — falling back")
        return None

    logger.info(f"Extracted: {json.dumps(params, default=str, ensure_ascii=False)[:500]}")

    # Inject CSV content for bank reconciliation
    if task_type == "bank_reconciliation" and files:
        for f in files:
            fn = f.get("filename", "")
            mime = f.get("mime_type", "")
            if fn.endswith(".csv") or "csv" in mime or "text" in mime:
                try:
                    raw = base64.b64decode(f.get("content_base64", ""))
                    params["_csv_content"] = raw.decode("utf-8", errors="replace")
                    break
                except Exception:
                    pass

    handler = HANDLERS[task_type]
    try:
        result = handler(tripletex, params)
    except Exception as e:
        logger.exception(f"Handler {task_type} crashed: {e}")
        return None

    logger.info(f"Handler result: {json.dumps(result, default=str)[:300]}")

    if result and result.get("success"):
        return {"status": "completed"}

    logger.warning(f"Handler failed: {result.get('error')} — falling back to LLM consensus")
    return None
