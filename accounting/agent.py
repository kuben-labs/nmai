"""Claude-powered planner-executor agent for Tripletex accounting tasks.

Architecture:
1. Planner (1 call): Breaks task into subtasks, tags each with domains
2. Executor (fresh chat per step): Gets ONLY the API docs for tagged domains
3. Context passing: Entity IDs flow between steps via $variable substitution
"""

import json
import logging
import base64
import os
import re
from datetime import date, datetime

import anthropic

from tripletex import TripletexClient
from prompts_v2 import DOMAINS, build_doer_prompt, PLANNER_PROMPT, FILE_HANDLING_PROMPT

logger = logging.getLogger(__name__)

# Keywords → domain mapping for task classification (fallback if planner doesn't tag)
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

DOMAIN_DEPS = {
    "travel": ["employee"],
    "salary": ["employee"],
    "dimension": ["voucher"],
    "contact": ["customer"],
    "bank_reconciliation": ["invoice", "supplier"],
    "closing": ["voucher"],
}


STEP_PREFIX = """You are executing a single focused subtask in Tripletex.
Complete ONLY the described subtask — nothing more, nothing less.
Use EXACTLY the values specified in the subtask description. Do not add fields or steps not mentioned.

RULES:
- GET calls are free — use them to look up IDs before writes
- Make exactly the write calls described in this subtask — no more, no fewer
- If the subtask says to create multiple entities, create them all in this step
- $variable references have been replaced with actual IDs in the context below
- If a GET returns existing entities, use their IDs — don't create duplicates
- Use the EXACT names, amounts, dates, and values from the subtask description
- When done, stop. Do not continue to other tasks.
"""


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
        max_len = 50000 if tool_name == "tripletex_get" else 15000
        if len(result_str) > max_len:
            if "body" in result and "values" in result.get("body", {}):
                values = result["body"]["values"]
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


def classify_task(prompt: str, has_files: bool) -> list[str]:
    """Classify task into relevant domains using keyword matching."""
    prompt_lower = prompt.lower()
    domains = set()

    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(kw in prompt_lower for kw in keywords):
            domains.add(domain)

    if has_files and domains & {"voucher"}:
        domains.update(["supplier", "department"])

    if "project" in domains and any(w in prompt_lower for w in ["faktura", "invoice", "kostnad", "cost"]):
        domains.update(["invoice", "customer", "supplier", "voucher"])

    if "invoice" in domains:
        domains.add("customer")

    for domain in list(domains):
        if domain in DOMAIN_DEPS:
            domains.update(DOMAIN_DEPS[domain])

    if not domains:
        domains = set(DOMAINS.keys())

    return sorted(domains)


def create_plan(client, model: str, prompt: str, files: list) -> list:
    """Call the planner to break the task into subtasks."""
    user_content = build_user_message(prompt, files)

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=PLANNER_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )

    text = ""
    for block in response.content:
        if hasattr(block, "text"):
            text += block.text

    # Parse JSON — handle markdown code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        plan_data = json.loads(json_match.group(1))
    else:
        plan_data = json.loads(text.strip())

    subtasks = plan_data.get("subtasks", [])

    # Validate and normalize
    for s in subtasks:
        if "id" not in s:
            s["id"] = subtasks.index(s) + 1
        if "depends_on" not in s:
            s["depends_on"] = []
        if "output_key" not in s:
            s["output_key"] = None
        if "domains" not in s:
            s["domains"] = []

    return subtasks


def execute_step(client, model: str, tripletex: TripletexClient,
                 subtask: dict, context: dict, files: list) -> tuple:
    """Execute a single subtask in a fresh chat with domain-focused prompt.

    Returns: (api_calls, errors, created_id)
    """
    api_calls = []
    errors = []
    created_id = None

    task_text = subtask["task"]

    # Replace $variable references with actual values from previous steps
    for key, value in context.items():
        task_text = task_text.replace(f"${key}", str(value))

    # Build context string showing available entity IDs
    today = date.today().isoformat()
    if context:
        ctx_lines = [f"  {k} = {v}" for k, v in context.items()]
        ctx_str = "\n".join(ctx_lines)
        user_text = f"Today's date is {today}.\n\nSUBTASK:\n{task_text}\n\nCONTEXT (entity IDs from previous steps):\n{ctx_str}"
    else:
        user_text = f"Today's date is {today}.\n\nSUBTASK:\n{task_text}"

    # Build user content with files if this is the first step or step needs files
    user_content = []
    # Include image/PDF files for vision on steps that might need them
    for f in files:
        mime_type = f.get("mime_type", "")
        if mime_type and mime_type.startswith("image/"):
            user_content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": mime_type, "data": f["content_base64"]}
            })
        elif mime_type == "application/pdf":
            user_content.append({
                "type": "document",
                "source": {"type": "base64", "media_type": "application/pdf", "data": f["content_base64"]}
            })

    # Add file text extraction if there are files
    file_text = extract_file_content(files)
    if file_text:
        user_text += f"\n\n{file_text}"

    user_content.append({"type": "text", "text": user_text})

    # Build focused system prompt from tagged domains
    domains = subtask.get("domains", [])
    if domains:
        domain_docs = build_doer_prompt(domains)
    else:
        # Fallback: use keyword classifier on the subtask text
        fallback_domains = classify_task(task_text, bool(files))
        domain_docs = build_doer_prompt(fallback_domains)

    system_prompt = STEP_PREFIX + "\n\n" + domain_docs

    # Fresh chat for this step
    messages = [{"role": "user", "content": user_content}]

    max_iterations = 10
    for iteration in range(max_iterations):
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_prompt,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            break

        if response.stop_reason != "tool_use":
            break

        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        tool_results = []
        for block in assistant_content:
            if block.type == "tool_use":
                logger.info(f"    [step {subtask['id']}] {block.name}({json.dumps(block.input, default=str)[:200]})")
                result = execute_tool(tripletex, block.name, block.input)

                try:
                    result_parsed = json.loads(result)
                    status_code = result_parsed.get("status_code", 0)
                except Exception:
                    status_code = 0
                    result_parsed = {}

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

                # Extract created entity ID from successful POST/PUT
                if status_code in (200, 201) and block.name in ("tripletex_post", "tripletex_put"):
                    try:
                        body = result_parsed.get("body", {})
                        value = body.get("value", {})
                        if isinstance(value, dict) and "id" in value:
                            created_id = value["id"]
                    except Exception:
                        pass

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        messages.append({"role": "user", "content": tool_results})

    return api_calls, errors, created_id


def run_agent(prompt: str, files: list, base_url: str, session_token: str) -> dict:
    """Run the planner-executor agent to complete an accounting task.

    1. Planner breaks task into subtasks with domains and dependencies
    2. Each subtask runs in a fresh chat with focused system prompt
    3. Entity IDs flow between steps via context dict
    """
    tripletex = TripletexClient(base_url, session_token)

    project_id = os.getenv("GCP_PROJECT_ID", "ai-nm26osl-1759")
    region = os.getenv("CLAUDE_VERTEX_REGION", "global")
    model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

    logger.info(f"Planner-executor agent using {model} via Vertex AI (project={project_id}, region={region})")
    client = anthropic.AnthropicVertex(project_id=project_id, region=region)

    api_calls = []
    errors = []

    try:
        # Phase 1: Create plan
        plan = create_plan(client, model, prompt, files)
        logger.info(f"Plan created: {len(plan)} subtasks")
        for s in plan:
            logger.info(f"  Step {s['id']}: {s['task'][:120]}... domains={s.get('domains', [])} depends={s['depends_on']} output_key={s.get('output_key')}")

        # Phase 2: Execute subtasks in dependency order
        context = {}  # output_key → entity_id
        completed = set()

        while len(completed) < len(plan):
            # Find steps whose dependencies are all completed
            ready = [
                s for s in plan
                if s["id"] not in completed
                and all(d in completed for d in s["depends_on"])
            ]

            if not ready:
                logger.warning("No ready subtasks but not all complete — circular dependency?")
                break

            for subtask in ready:
                logger.info(f"Executing step {subtask['id']}/{len(plan)}: {subtask['task'][:120]}...")

                step_calls, step_errors, created_id = execute_step(
                    client, model, tripletex, subtask, context, files
                )
                api_calls.extend(step_calls)
                errors.extend(step_errors)

                # Store created entity ID for downstream steps
                if subtask.get("output_key") and created_id is not None:
                    context[subtask["output_key"]] = created_id
                    logger.info(f"  -> {subtask['output_key']} = {created_id}")

                completed.add(subtask["id"])

    except Exception as e:
        logger.exception("Planner-executor agent error")
        errors.append({"tool": "agent", "endpoint": "", "status_code": 0,
                       "error_body": str(e), "input_data": ""})
    finally:
        tripletex.close()

    # Determine success
    write_errors = [e for e in errors if e["tool"] != "tripletex_get"]
    write_calls = [c for c in api_calls if c["tool"] != "tripletex_get"]

    if write_calls:
        last_write = write_calls[-1]
        has_errors = last_write["status_code"] >= 400
    else:
        has_errors = len(write_errors) > 0

    summary = {
        "task_summary": True,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "prompt": prompt[:2000],
        "status": "FAILED" if has_errors else "OK",
        "plan_steps": len(plan) if 'plan' in locals() else 0,
        "total_calls": len(api_calls),
        "write_calls": len(write_calls),
        "error_count": len(errors),
        "write_error_count": len(write_errors),
        "errors": errors[:20],
        "api_calls": api_calls,
    }
    logger.info(f"TASK_RESULT: {json.dumps(summary, ensure_ascii=False)}")

    return {"status": "completed"}
