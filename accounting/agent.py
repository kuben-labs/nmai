"""Claude-powered agentic loop for Tripletex accounting tasks."""

import json
import logging
import base64
import os
from datetime import date, datetime

import anthropic

from tripletex import TripletexClient
from prompts_v2 import DOMAINS, build_doer_prompt, FILE_HANDLING_PROMPT

logger = logging.getLogger(__name__)

# Keywords → domain mapping for task classification
DOMAIN_KEYWORDS = {
    "customer": ["kunde", "customer", "client", "cliente", "kundefaktura"],
    "supplier": ["leverandør", "supplier", "fournisseur", "proveedor", "fornecedor", "lieferant",
                 "leverandørkostnad", "leverandørfaktura"],
    "employee": ["ansatt", "employee", "empleado", "empregado", "mitarbeiter", "employé",
                 "onboarding", "incorpora", "einarbeitung", "intégration",
                 "rolle", "role", "admin", "kontoadministrator", "regnskapsfører",
                 "revisor", "auditor", "faktureringsansvarlig", "personalansvarlig"],
    "invoice": ["faktura", "invoice", "factura", "fatura", "rechnung", "facture",
                "ordre", "order", "bestilling", "pedido", "bestellung", "commande",
                "kreditnota", "credit note", "nota de crédito", "gutschrift",
                "purring", "reminder", "recordatorio", "rappel", "mahnung"],
    "voucher": ["bilag", "voucher", "journal", "konto", "account",
                "debet", "debit", "kredit", "credit",
                "bokfør", "registrer", "postering",
                "kvittering", "receipt", "recibo", "quittung", "reçu",
                "kostnad", "expense", "utgift", "despesa", "gasto", "aufwand", "dépense",
                "korreksjon", "correction", "feil", "error"],
    "travel": ["reise", "travel", "viaje", "viagem", "voyage",
               "diett", "per diem", "dietpenger", "kjøregodtgjørelse", "mileage",
               "reiseregning"],
    "project": ["prosjekt", "project", "proyecto", "projeto", "projekt", "projet",
                "timesheet", "timer", "timar", "hours", "horas", "stunden", "heures"],
    "salary": ["lønn", "salary", "salario", "gehalt", "salaire",
               "payroll", "lønnskjøring", "lønnsslipp"],
    "dimension": ["dimensjon", "dimension", "dimensión", "dimensão"],
    "department": ["avdeling", "department", "departamento", "abteilung", "département"],
    "company": ["selskap", "company", "empresa", "unternehmen", "entreprise", "firma"],
    "contact": ["kontakt", "contact", "contacto"],
    "bank_reconciliation": ["bankavsteming", "reconciliation", "conciliación", "reconciliação",
                            "abstimmung", "rapprochement", "kontoutskrift", "bank statement"],
    "closing": ["avskrivning", "depreciation", "depreciación", "abschreibung", "amortissement",
                "årsavslutning", "year-end", "periodisering", "accrual",
                "skattekostnad", "tax provision", "skatt"],
}

# Domains that should co-include other domains
DOMAIN_DEPS = {
    "travel": ["employee"],
    "salary": ["employee"],
    "dimension": ["voucher"],
    "contact": ["customer"],
    "bank_reconciliation": ["invoice", "supplier"],
    "closing": ["voucher"],
}


VERIFY_PROMPT = """VERIFICATION CHECK: Before finishing, verify your work:
1. GET back the key entities you created/modified
2. Compare each field against the original task requirements
3. Check: correct amounts? correct dates? correct names? correct accounts? correct department/project links?
4. If a field is wrong, fix it with PUT (do NOT delete and recreate)
5. If everything is correct, confirm and stop

IMPORTANT: Do NOT delete any entities during verification. Only use PUT to fix wrong values. Do this now — GET the entities and verify."""


def classify_task(prompt: str, has_files: bool) -> list[str]:
    """Classify task into relevant domains using keyword matching."""
    prompt_lower = prompt.lower()
    domains = set()

    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(kw in prompt_lower for kw in keywords):
            domains.add(domain)

    # Receipt/expense from attached image → need supplier + department + voucher
    if has_files and domains & {"voucher"}:
        domains.update(["supplier", "department"])

    # Project lifecycle often needs invoice + supplier for costs
    if "project" in domains and any(w in prompt_lower for w in ["faktura", "invoice", "kostnad", "cost"]):
        domains.update(["invoice", "customer", "supplier", "voucher"])

    # Invoice tasks need customer
    if "invoice" in domains:
        domains.add("customer")

    # Add dependency domains
    for domain in list(domains):
        if domain in DOMAIN_DEPS:
            domains.update(DOMAIN_DEPS[domain])

    # Fallback: if nothing matched, include common domains
    if not domains:
        domains = set(DOMAINS.keys())

    return sorted(domains)


TOOLS = [
    {
        "name": "tripletex_get",
        "description": "Make a GET request to the Tripletex API. Use this to query/search for entities. GET requests are free and don't affect scoring. Always use the `fields` parameter to limit response size.",
        "input_schema": {
            "type": "object",
            "properties": {
                "endpoint": {
                    "type": "string",
                    "description": "API endpoint path, e.g. '/employee', '/customer', '/invoice', '/ledger/vatType'. Do NOT include the base URL."
                },
                "params": {
                    "type": "object",
                    "description": "Query parameters as key-value pairs, e.g. {\"fields\": \"id,name\", \"count\": 100}",
                    "additionalProperties": True
                }
            },
            "required": ["endpoint"]
        }
    },
    {
        "name": "tripletex_post",
        "description": "Make a POST request to the Tripletex API. Use this to CREATE new entities. Each POST counts as a write call and affects your efficiency score. Make sure all required fields are included to avoid 4xx errors.",
        "input_schema": {
            "type": "object",
            "properties": {
                "endpoint": {
                    "type": "string",
                    "description": "API endpoint path, e.g. '/employee', '/customer', '/order', '/invoice'"
                },
                "data": {
                    "type": "object",
                    "description": "JSON body for the POST request",
                    "additionalProperties": True
                }
            },
            "required": ["endpoint", "data"]
        }
    },
    {
        "name": "tripletex_put",
        "description": "Make a PUT request to the Tripletex API. Use this to UPDATE existing entities. Each PUT counts as a write call. Include the entity ID in the endpoint path or body as required.",
        "input_schema": {
            "type": "object",
            "properties": {
                "endpoint": {
                    "type": "string",
                    "description": "API endpoint path, e.g. '/employee/123', '/customer/456'"
                },
                "data": {
                    "type": "object",
                    "description": "JSON body for the PUT request (usually the full updated entity)",
                    "additionalProperties": True
                }
            },
            "required": ["endpoint", "data"]
        }
    },
    {
        "name": "tripletex_delete",
        "description": "Make a DELETE request to the Tripletex API. Use this to DELETE entities. Include the entity ID in the endpoint path. Each DELETE counts as a write call.",
        "input_schema": {
            "type": "object",
            "properties": {
                "endpoint": {
                    "type": "string",
                    "description": "API endpoint path with ID, e.g. '/travelExpense/123', '/ledger/voucher/456'"
                }
            },
            "required": ["endpoint"]
        }
    },
]


def extract_file_content(files: list) -> str:
    """Extract text content from attached files for the LLM."""
    if not files:
        return ""

    parts = [FILE_HANDLING_PROMPT, "\n## ATTACHED FILES\n"]

    for f in files:
        filename = f.get("filename", "unknown")
        mime_type = f.get("mime_type", "")
        content_b64 = f.get("content_base64", "")

        parts.append(f"\n### File: {filename} ({mime_type})\n")

        if mime_type == "application/pdf":
            try:
                raw = base64.b64decode(content_b64)
                from PyPDF2 import PdfReader
                import io
                reader = PdfReader(io.BytesIO(raw))
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                if text.strip():
                    parts.append(f"Extracted text:\n```\n{text}\n```\n")
                else:
                    parts.append("(PDF contained no extractable text - may be image-based)\n")
            except Exception as e:
                parts.append(f"(Could not extract PDF text: {e})\n")

        elif mime_type and mime_type.startswith("image/"):
            parts.append("(Image file attached as inline image)\n")

        elif mime_type and ("csv" in mime_type or "text" in mime_type or filename.endswith(".csv")):
            try:
                raw = base64.b64decode(content_b64)
                text = raw.decode("utf-8", errors="replace")
                parts.append(f"Content:\n```\n{text}\n```\n")
            except Exception as e:
                parts.append(f"(Could not decode text: {e})\n")

    return "\n".join(parts)


def build_user_message(prompt: str, files: list) -> list:
    """Build the user message content blocks, including images and PDFs for vision."""
    content_blocks = []

    # Add files as native content blocks (vision/document)
    for f in files:
        mime_type = f.get("mime_type", "")
        if mime_type and mime_type.startswith("image/"):
            content_blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": f["content_base64"],
                }
            })
        elif mime_type == "application/pdf":
            content_blocks.append({
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": f["content_base64"],
                }
            })

    # Build the text part
    file_text = extract_file_content(files)
    today = date.today().isoformat()
    text = f"Today's date is {today}.\n\nTASK:\n{prompt}"
    if file_text:
        text += f"\n\n{file_text}"

    content_blocks.append({"type": "text", "text": text})
    return content_blocks


def execute_tool(client: TripletexClient, tool_name: str, tool_input: dict) -> str:
    """Execute a Tripletex API tool call and return the result as a string."""
    try:
        if tool_name == "tripletex_get":
            result = client.get(tool_input["endpoint"], tool_input.get("params"))
        elif tool_name == "tripletex_post":
            result = client.post(tool_input["endpoint"], tool_input.get("data"))
        elif tool_name == "tripletex_put":
            result = client.put(tool_input["endpoint"], tool_input.get("data"))
        elif tool_name == "tripletex_delete":
            result = client.delete(tool_input["endpoint"])
        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        result_str = json.dumps(result, default=str, ensure_ascii=False)
        # Truncate very large responses but keep useful info
        # Use higher limit for GET (free) vs write calls
        max_len = 50000 if tool_name == "tripletex_get" else 15000
        if len(result_str) > max_len:
            if "body" in result and "values" in result.get("body", {}):
                values = result["body"]["values"]
                # For GETs, keep more results to avoid repeated lookups
                keep = 100 if tool_name == "tripletex_get" else 20
                if len(values) > keep:
                    result["body"]["values"] = values[:keep]
                    result["body"]["_note"] = f"Showing {keep} of {len(values)} results. Use more specific search params to narrow down."
                result_str = json.dumps(result, default=str, ensure_ascii=False)
            if len(result_str) > max_len:
                result_str = result_str[:max_len] + "\n... (truncated)"

        return result_str
    except Exception as e:
        logger.exception(f"Tool execution error: {tool_name}")
        return json.dumps({"error": str(e)})


def structure_task(client, model: str, user_content: list) -> str:
    """Phase 1: Call the structurer model to extract and organize task info."""
    try:
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            system=STRUCTURER_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
        )
        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text
        return text.strip()
    except Exception as e:
        logger.warning(f"Structurer failed: {e}")
        return ""  # Fall back to no structured plan


STRUCTURER_SYSTEM = """You are a task analyzer for Tripletex accounting. Your job is to read the task and any attached files, then output a clean structured breakdown.

OUTPUT FORMAT — respond with ONLY this JSON:
{
  "objective": "One sentence: what needs to be done",
  "extracted_data": {
    "names": [...],
    "amounts": [...],
    "dates": [...],
    "accounts": [...],
    "other": {...}
  },
  "steps": [
    "Step 1: description with exact values to use",
    "Step 2: ...",
    ...
  ],
  "warnings": ["Any tricky aspects to watch out for"]
}

RULES:
- Extract ALL values from the task text and any attached files (receipts, PDFs, CSVs)
- Pre-compute all calculations (VAT splits, currency conversions, depreciation, etc.)
- List steps in execution order with exact field values
- If a file is attached, extract every relevant value (supplier name, date, amounts, items, VAT)
- Be specific: "Create employee with firstName='Ola', lastName='Nordmann'" not "Create the employee"
"""


def run_agent(prompt: str, files: list, base_url: str, session_token: str) -> dict:
    """Run the two-phase agent: structurer → executor."""
    tripletex = TripletexClient(base_url, session_token)

    project_id = os.getenv("GCP_PROJECT_ID", "ai-nm26osl-1759")
    region = os.getenv("CLAUDE_VERTEX_REGION", "global")
    model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

    logger.info(f"Using Claude {model} via Vertex AI (project={project_id}, region={region})")
    client = anthropic.AnthropicVertex(project_id=project_id, region=region)

    # Phase 1: Structure the task
    user_content = build_user_message(prompt, files)
    structured_plan = structure_task(client, model, user_content)
    logger.info(f"STRUCTURED_PLAN: {structured_plan[:2000]}")

    # Classify task and build focused system prompt
    domains = classify_task(prompt, bool(files))
    system_prompt = build_doer_prompt(domains)
    logger.info(f"Task domains: {domains} (prompt size: {len(system_prompt)} chars)")

    # Phase 2: Execute with the structured plan injected
    today = date.today().isoformat()
    executor_text = f"Today's date is {today}.\n\nORIGINAL TASK:\n{prompt}\n\nSTRUCTURED PLAN (from analysis):\n{structured_plan}\n\nExecute this plan now. Follow the steps exactly. Do NOT skip any step."

    # Build executor message — include files for vision but use structured text
    executor_content = []
    for f in files:
        mime_type = f.get("mime_type", "")
        if mime_type and mime_type.startswith("image/"):
            executor_content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": mime_type, "data": f["content_base64"]}
            })
        elif mime_type == "application/pdf":
            executor_content.append({
                "type": "document",
                "source": {"type": "base64", "media_type": "application/pdf", "data": f["content_base64"]}
            })

    file_text = extract_file_content(files)
    if file_text:
        executor_text += f"\n\n{file_text}"
    executor_content.append({"type": "text", "text": executor_text})

    messages = [{"role": "user", "content": executor_content}]

    max_iterations = 30
    iteration = 0
    verified = False

    # Track all API calls for task summary
    api_calls = []
    errors = []

    try:
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"Agent iteration {iteration}/{max_iterations}")

            response = client.messages.create(
                model=model,
                max_tokens=4096,
                system=system_prompt,
                tools=TOOLS,
                messages=messages,
            )

            logger.info(f"  Stop reason: {response.stop_reason}")

            if response.stop_reason == "end_turn":
                # Check if we did any writes — if so, run verification
                write_calls_so_far = [c for c in api_calls if c["tool"] != "tripletex_get"]
                if write_calls_so_far and not verified:
                    verified = True
                    logger.info("Injecting verification step")
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({"role": "user", "content": VERIFY_PROMPT})
                    continue  # One more iteration for verification
                logger.info("Agent completed task (end_turn)")
                break

            if response.stop_reason != "tool_use":
                logger.info(f"Agent stopped with reason: {response.stop_reason}")
                break

            # Process tool calls
            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            tool_results = []
            for block in assistant_content:
                if block.type == "tool_use":
                    logger.info(f"  Tool call: {block.name}({json.dumps(block.input, default=str)[:200]})")
                    result = execute_tool(tripletex, block.name, block.input)

                    # Parse result to track status
                    try:
                        result_parsed = json.loads(result)
                        status_code = result_parsed.get("status_code", 0)
                    except Exception:
                        status_code = 0

                    call_record = {
                        "tool": block.name,
                        "endpoint": block.input.get("endpoint", ""),
                        "status_code": status_code,
                    }
                    api_calls.append(call_record)

                    if status_code >= 400:
                        err_body = ""
                        try:
                            err_body = json.dumps(result_parsed.get("body", {}), ensure_ascii=False)[:500]
                        except Exception:
                            pass
                        errors.append({
                            "tool": block.name,
                            "endpoint": block.input.get("endpoint", ""),
                            "status_code": status_code,
                            "error_body": err_body,
                            "input_data": json.dumps(block.input.get("data", {}), default=str, ensure_ascii=False)[:500],
                        })

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            messages.append({"role": "user", "content": tool_results})

    except Exception as e:
        logger.exception(f"Agent error at iteration {iteration}")
        errors.append({"tool": "agent", "endpoint": "", "status_code": 0, "error_body": str(e), "input_data": ""})
    finally:
        tripletex.close()

    # Determine success: check if last write calls succeeded (recovered from earlier errors)
    write_errors = [e for e in errors if e["tool"] != "tripletex_get"]
    write_calls = [c for c in api_calls if c["tool"] != "tripletex_get"]
    # Task is FAILED only if it never completed a successful write, or the last write call failed
    if write_calls:
        last_write = write_calls[-1]
        has_errors = last_write["status_code"] >= 400
    else:
        has_errors = len(write_errors) > 0

    # Log structured task summary for easy querying
    summary = {
        "task_summary": True,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "prompt": prompt[:2000],
        "status": "FAILED" if has_errors else "OK",
        "iterations": iteration,
        "total_calls": len(api_calls),
        "write_calls": len([c for c in api_calls if c["tool"] != "tripletex_get"]),
        "error_count": len(errors),
        "write_error_count": len(write_errors),
        "errors": errors[:20],
        "api_calls": api_calls,
    }
    logger.info(f"TASK_RESULT: {json.dumps(summary, ensure_ascii=False)}")

    return {"status": "completed"}
