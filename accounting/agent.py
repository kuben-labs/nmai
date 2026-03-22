"""Claude-powered consensus agent for Tripletex accounting tasks.

Architecture:
1. Three models think about the task in parallel (no tools)
2. Reconciler model merges their plans into one consensus
3. Executor model runs the consensus plan with tools
"""

import json
import logging
import base64
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime

import anthropic

from tripletex import TripletexClient
from prompts import SYSTEM_PROMPT, FILE_HANDLING_PROMPT

logger = logging.getLogger(__name__)

THINKER_SYSTEM = SYSTEM_PROMPT + """

## YOUR ROLE: PLANNER (NO TOOL ACCESS)
You are PLANNING ONLY. You do NOT have tools. Do NOT attempt any API calls.
The task may be in Norwegian, English, Spanish, Portuguese, German, French, or Nynorsk — understand it in any language.

Output a CONCRETE step-by-step plan:
1. Translate/understand the task requirements (list ALL values: names, amounts, dates, accounts, percentages)
2. Pre-compute ALL calculations (VAT splits, depreciation, net/gross, etc.)
3. List the EXACT API calls in dependency order — each step must only use IDs from previous steps
4. For each call: endpoint, method (GET/POST/PUT), and exact field values

DEPENDENCY ORDER IS CRITICAL:
- You MUST create entities before referencing their IDs
- Example: POST /customer FIRST → get customer_id → then POST /order with that customer_id → get order_id → then POST /invoice with that order_id
- Never reference an ID you haven't obtained yet

If files are attached, extract ALL values from them (amounts, dates, names, supplier, items).
Be specific — use exact values copied from the task. Do not be vague or generic.
"""

RECONCILER_PROMPT = """You are merging three independent plans for the same Tripletex accounting task into one consensus plan.

The task may be in any language (Norwegian, English, Spanish, Portuguese, German, French, Nynorsk).
Below are three plans from different analysts who all analyzed the same task.

Your job:
1. Compare all three plans
2. Identify what they AGREE on — this is likely correct
3. Where they DISAGREE, pick the approach that is most complete and uses the exact values from the task
4. Output ONE final consensus plan

CRITICAL RULES for the consensus plan:
- Steps MUST be in strict dependency order (create parent entities before children)
- Include EXACT field values, amounts, dates, calculations, and endpoint paths
- Every step must specify: endpoint, method, and exact data/params
- The executor will follow your plan LITERALLY — be precise
- Do NOT omit any step that all three plans agree on
- If a calculation differs between plans, recompute it yourself from the original task values
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
                    result["body"]["_note"] = f"Showing {keep} of {len(values)} results."
                result_str = json.dumps(result, default=str, ensure_ascii=False)
            if len(result_str) > max_len:
                result_str = result_str[:max_len] + "\n... (truncated)"

        return result_str
    except Exception as e:
        logger.exception(f"Tool execution error: {tool_name}")
        return json.dumps({"error": str(e)})


def gather_preflight_data(tripletex: TripletexClient, prompt: str) -> str:
    """Gather real data from Tripletex before planning. GET calls are free.

    Based on task keywords, fetch relevant existing data so the thinkers
    can plan with actual IDs, names, and values.
    """
    p = prompt.lower()
    data_parts = []

    # Always useful: existing employees and departments (small, needed for most tasks)
    result = tripletex.get("/employee", {"fields": "id,firstName,lastName,email", "count": 50})
    if result["status_code"] == 200:
        employees = result["body"].get("values", [])
        if employees:
            data_parts.append(f"EXISTING EMPLOYEES ({len(employees)}):")
            for e in employees[:20]:
                data_parts.append(f"  id={e['id']}, {e.get('firstName','')} {e.get('lastName','')}, email={e.get('email','')}")

    result = tripletex.get("/department", {"fields": "id,name,departmentNumber", "count": 50})
    if result["status_code"] == 200:
        depts = result["body"].get("values", [])
        if depts:
            data_parts.append(f"\nEXISTING DEPARTMENTS ({len(depts)}):")
            for d in depts[:20]:
                data_parts.append(f"  id={d['id']}, name={d.get('name','')}, number={d.get('departmentNumber','')}")

    # Accounts — needed for vouchers, closing, analysis
    if any(kw in p for kw in ["konto", "account", "voucher", "bilag", "journal",
                               "closing", "clôture", "cierre", "encerramento",
                               "depreci", "avskriv", "accrual", "periodiser",
                               "skatt", "tax", "kostnad", "expense", "despesa",
                               "gasto", "aufwand", "dépense", "razão", "ledger",
                               "correction", "korreksjon", "feil", "error"]):
        result = tripletex.get("/ledger/account", {"fields": "id,number,name", "count": 1000})
        if result["status_code"] == 200:
            accounts = result["body"].get("values", [])
            data_parts.append(f"\nALL ACCOUNTS ({len(accounts)}):")
            for a in accounts:
                data_parts.append(f"  id={a['id']}, {a.get('number','')} {a.get('name','')}")

    # Existing customers
    if any(kw in p for kw in ["kunde", "customer", "client", "faktura", "invoice",
                               "ordre", "order", "rechnung", "facture", "factura",
                               "kreditnota", "credit note", "gutschrift", "nota de crédito",
                               "betaling", "payment", "purring", "reminder"]):
        result = tripletex.get("/customer", {"fields": "id,name,email,organizationNumber", "count": 50})
        if result["status_code"] == 200:
            customers = result["body"].get("values", [])
            if customers:
                data_parts.append(f"\nEXISTING CUSTOMERS ({len(customers)}):")
                for c in customers[:20]:
                    data_parts.append(f"  id={c['id']}, name={c.get('name','')}, org={c.get('organizationNumber','')}")

    # Existing suppliers
    if any(kw in p for kw in ["leverandør", "supplier", "fournisseur", "proveedor",
                               "lieferant", "fornecedor"]):
        result = tripletex.get("/supplier", {"fields": "id,name,email", "count": 50})
        if result["status_code"] == 200:
            suppliers = result["body"].get("values", [])
            if suppliers:
                data_parts.append(f"\nEXISTING SUPPLIERS ({len(suppliers)}):")
                for s in suppliers[:20]:
                    data_parts.append(f"  id={s['id']}, name={s.get('name','')}")

    # Existing projects
    if any(kw in p for kw in ["prosjekt", "project", "proyecto", "projeto", "projekt", "projet"]):
        result = tripletex.get("/project", {"fields": "id,name,projectManager,startDate", "count": 50})
        if result["status_code"] == 200:
            projects = result["body"].get("values", [])
            if projects:
                data_parts.append(f"\nEXISTING PROJECTS ({len(projects)}):")
                for pr in projects[:20]:
                    data_parts.append(f"  id={pr['id']}, name={pr.get('name','')}")

    # Existing invoices (for payment/reconciliation/credit note tasks)
    if any(kw in p for kw in ["betaling", "payment", "reconcili", "avsteming", "innbetaling",
                               "purring", "reminder", "kreditnota", "credit note", "gutschrift",
                               "nota de crédito", "reverse", "reverser", "storno"]):
        result = tripletex.get("/invoice", {"fields": "id,invoiceNumber,amount,amountOutstanding,customer",
                                             "invoiceDateFrom": "2020-01-01", "invoiceDateTo": "2030-12-31", "count": 50})
        if result["status_code"] == 200:
            invoices = result["body"].get("values", [])
            if invoices:
                data_parts.append(f"\nEXISTING INVOICES ({len(invoices)}):")
                for inv in invoices[:20]:
                    cust = inv.get("customer", {})
                    data_parts.append(f"  id={inv['id']}, #{inv.get('invoiceNumber','')}, amount={inv.get('amount','')}, outstanding={inv.get('amountOutstanding','')}, customer={cust.get('name','')}")

    # Salary tasks: need salary types, divisions, employments
    if any(kw in p for kw in ["lønn", "salary", "payroll", "lønnskjøring", "gehalt",
                               "gehaltsabrechnung", "salario", "salaire", "nómina"]):
        result = tripletex.get("/salary/type", {"fields": "id,name,number", "count": 100})
        if result["status_code"] == 200:
            types = result["body"].get("values", [])
            if types:
                data_parts.append(f"\nSALARY TYPES ({len(types)}):")
                for t in types[:30]:
                    data_parts.append(f"  id={t['id']}, name={t.get('name','')}, number={t.get('number','')}")

        result = tripletex.get("/division", {"fields": "id,name,organizationNumber", "count": 10})
        if result["status_code"] == 200:
            divs = result["body"].get("values", [])
            if divs:
                data_parts.append(f"\nEXISTING DIVISIONS ({len(divs)}):")
                for d in divs[:10]:
                    data_parts.append(f"  id={d['id']}, name={d.get('name','')}, orgNr={d.get('organizationNumber','')}")
            else:
                data_parts.append(f"\nNO DIVISIONS EXIST — you must create one before salary processing")
                data_parts.append(f"  POST /division needs: name, startDate, organizationNumber (9 digits, different from company), municipality (object with id)")

    # Voucher analysis — for comparison/correction tasks, compute aggregations in Python
    is_analysis = any(kw in p for kw in ["aumentar", "increase", "økte", "økt", "største",
                                          "largest", "biggest", "mayor", "maior", "größt",
                                          "analyse", "analys", "analiz", "analy"])
    is_correction = any(kw in p for kw in ["korreksjon", "correction", "feil", "error",
                                            "corrección", "correção", "korrektur"])
    is_ledger = any(kw in p for kw in ["razão", "ledger", "hovedbok", "grand livre",
                                        "libro mayor", "hauptbuch"])

    if is_analysis or is_correction or is_ledger:
        result = tripletex.get("/ledger/voucher", {"fields": "id,date,description,postings",
                                                    "dateFrom": "2025-01-01", "dateTo": "2026-12-31", "count": 500})
        if result["status_code"] == 200:
            vouchers = result["body"].get("values", [])
            if vouchers:
                # For correction tasks: show raw vouchers
                if is_correction:
                    data_parts.append(f"\nEXISTING VOUCHERS ({len(vouchers)}):")
                    for v in vouchers[:50]:
                        postings_summary = []
                        for post in (v.get("postings", []) or [])[:5]:
                            acct = post.get("account", {})
                            postings_summary.append(f"{acct.get('number','?')}:{post.get('amount',0)}")
                        data_parts.append(f"  id={v['id']}, date={v.get('date','')}, desc={v.get('description','')[:60]}, postings=[{', '.join(postings_summary)}]")

                # For analysis tasks: compute per-account per-month aggregations
                if is_analysis or is_ledger:
                    # Aggregate: {(account_number, account_name, account_id): {month: total_debit}}
                    from collections import defaultdict
                    account_month_totals = defaultdict(lambda: defaultdict(float))
                    account_info = {}  # account_number -> (name, id)

                    for v in vouchers:
                        v_date = v.get("date", "")
                        if not v_date or len(v_date) < 7:
                            continue
                        month_key = v_date[:7]  # "2026-01"
                        for post in (v.get("postings", []) or []):
                            acct = post.get("account", {})
                            acct_num = acct.get("number", "")
                            acct_name = acct.get("name", "")
                            acct_id = acct.get("id", "")
                            amount = post.get("amount", 0) or 0
                            if acct_num:
                                account_month_totals[acct_num][month_key] += amount
                                account_info[acct_num] = (acct_name, acct_id)

                    # Build summary sorted by account number
                    data_parts.append(f"\nACCOUNT TOTALS BY MONTH (computed from {len(vouchers)} vouchers):")
                    data_parts.append("  Account | Name | " + " | ".join(sorted(set(
                        m for totals in account_month_totals.values() for m in totals
                    ))))

                    months = sorted(set(m for totals in account_month_totals.values() for m in totals))

                    for acct_num in sorted(account_month_totals.keys()):
                        name, aid = account_info[acct_num]
                        month_vals = [f"{account_month_totals[acct_num].get(m, 0):.2f}" for m in months]
                        data_parts.append(f"  {acct_num} (id={aid}) {name}: {' | '.join(month_vals)}")

                    # Compute month-over-month changes for expense accounts (4xxx-8xxx)
                    if len(months) >= 2:
                        changes = []
                        for acct_num, totals in account_month_totals.items():
                            try:
                                num = int(acct_num)
                            except ValueError:
                                continue
                            if 4000 <= num <= 8999:  # Expense/cost accounts
                                sorted_months = sorted(totals.keys())
                                if len(sorted_months) >= 2:
                                    m1, m2 = sorted_months[-2], sorted_months[-1]
                                    val1 = totals[m1]
                                    val2 = totals[m2]
                                    diff = val2 - val1
                                    name, aid = account_info[acct_num]
                                    changes.append((diff, acct_num, name, aid, val1, val2, m1, m2))

                        changes.sort(reverse=True)
                        data_parts.append(f"\nEXPENSE ACCOUNT CHANGES (sorted by largest increase):")
                        for diff, acct_num, name, aid, val1, val2, m1, m2 in changes[:10]:
                            data_parts.append(f"  {acct_num} (id={aid}) {name}: {m1}={val1:.2f} → {m2}={val2:.2f}, change=+{diff:.2f}")

                        data_parts.append(f"\nTOP 3 EXPENSE ACCOUNTS WITH LARGEST INCREASE:")
                        for i, (diff, acct_num, name, aid, val1, val2, m1, m2) in enumerate(changes[:3], 1):
                            data_parts.append(f"  #{i}: {acct_num} {name} (id={aid}), increase=+{diff:.2f} ({m1}: {val1:.2f} → {m2}: {val2:.2f})")

    # Year-end / closing tasks: compute income statement so tax provision can be calculated
    is_closing = any(kw in p for kw in ["årsavslutning", "årsoppgjer", "year-end", "year end",
                                         "clôture", "cierre", "encerramento", "jahresabschluss",
                                         "avskriv", "depreci", "skatt", "tax", "impuesto", "imposto",
                                         "steuern", "impôt", "periodiser", "accrual"])
    if is_closing:
        # Fetch all vouchers for the closing year to compute income statement
        result = tripletex.get("/ledger/voucher", {"fields": "id,date,description,postings",
                                                    "dateFrom": "2025-01-01", "dateTo": "2025-12-31", "count": 500})
        if result["status_code"] == 200:
            vouchers = result["body"].get("values", [])
            from collections import defaultdict
            account_totals = defaultdict(float)
            account_info = {}

            for v in vouchers:
                for post in (v.get("postings", []) or []):
                    acct = post.get("account", {})
                    acct_num = str(acct.get("number", ""))
                    amount = post.get("amount", 0) or 0
                    if acct_num:
                        account_totals[acct_num] += amount
                        account_info[acct_num] = (acct.get("name", ""), acct.get("id", ""))

            # Compute income statement
            revenue = 0.0  # 3xxx accounts (credit = negative in Norwegian accounting)
            expenses = 0.0  # 4xxx-7xxx accounts (debit = positive)
            financial = 0.0  # 8xxx accounts

            for acct_num, total in account_totals.items():
                try:
                    num = int(acct_num)
                except ValueError:
                    continue
                if 3000 <= num <= 3999:
                    revenue += total  # Revenue is typically negative (credit)
                elif 4000 <= num <= 7999:
                    expenses += total  # Expenses are positive (debit)
                elif 8000 <= num <= 8999:
                    financial += total

            # Profit = -(revenue) - expenses - financial  (revenue is negative/credit)
            profit_before_closing = -(revenue) - expenses - financial

            data_parts.append(f"\nINCOME STATEMENT FOR 2025 (pre-computed from {len(vouchers)} vouchers):")
            data_parts.append(f"  Total revenue (3xxx accounts): {-revenue:.2f} NOK")
            data_parts.append(f"  Total expenses (4xxx-7xxx accounts): {expenses:.2f} NOK")
            data_parts.append(f"  Financial items (8xxx accounts): {financial:.2f} NOK")
            data_parts.append(f"  PROFIT BEFORE CLOSING ADJUSTMENTS: {profit_before_closing:.2f} NOK")
            data_parts.append(f"")
            data_parts.append(f"  NOTE FOR TAX PROVISION:")
            data_parts.append(f"  After you add depreciation expenses and reverse prepaid costs,")
            data_parts.append(f"  recalculate: adjusted_profit = {profit_before_closing:.2f} - total_depreciation - prepaid_reversal")
            data_parts.append(f"  Then: tax_provision = adjusted_profit × 0.22")
            data_parts.append(f"")

            # Show relevant account balances
            data_parts.append(f"  KEY ACCOUNT BALANCES:")
            for acct_num in sorted(account_totals.keys()):
                try:
                    num = int(acct_num)
                except ValueError:
                    continue
                if num in [1200, 1209, 1210, 1240, 1250, 1700, 2920, 6010, 8700] or (3000 <= num <= 3999) or (total := account_totals[acct_num]) != 0 and 4000 <= num <= 8999:
                    name, aid = account_info.get(acct_num, ("", ""))
                    data_parts.append(f"    {acct_num} (id={aid}) {name}: {account_totals[acct_num]:.2f}")

    if data_parts:
        return "\n".join(data_parts)
    return ""


def think_about_task(client, model: str, user_content: list, thinker_id: int) -> str:
    """One thinker model analyzes the task and outputs a plan (no tools)."""
    try:
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            system=THINKER_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
        )
        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text
        logger.info(f"Thinker {thinker_id} plan: {text[:200]}...")
        return text.strip()
    except Exception as e:
        logger.warning(f"Thinker {thinker_id} failed: {e}")
        return f"(Thinker {thinker_id} failed: {e})"


def reconcile_plans(client, model: str, plans: list[str], prompt: str) -> str:
    """Reconciler merges three plans into one consensus."""
    plans_text = ""
    for i, plan in enumerate(plans, 1):
        plans_text += f"\n--- PLAN {i} ---\n{plan}\n"

    user_text = f"ORIGINAL TASK:\n{prompt}\n\n{plans_text}\n\nCreate the consensus plan now."

    try:
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            system=RECONCILER_PROMPT,
            messages=[{"role": "user", "content": user_text}],
        )
        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text
        logger.info(f"Consensus plan: {text[:300]}...")
        return text.strip()
    except Exception as e:
        logger.warning(f"Reconciler failed: {e}")
        return plans[0] if plans else ""


def run_agent(prompt: str, files: list, base_url: str, session_token: str) -> dict:
    """Run the consensus agent: 3 thinkers -> reconciler -> executor.

    1. Three models independently plan the task (no tools, parallel)
    2. Reconciler merges plans into consensus
    3. Executor runs consensus with the full system prompt + tools
    """
    tripletex = TripletexClient(base_url, session_token)

    project_id = os.getenv("GCP_PROJECT", os.getenv("GCP_PROJECT_ID", "ai-nm26osl-1759"))
    region = os.getenv("GCP_REGION", os.getenv("CLAUDE_VERTEX_REGION", "global"))
    model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

    logger.info(f"Consensus agent using {model} via Vertex AI (project={project_id}, region={region})")
    client = anthropic.AnthropicVertex(project_id=project_id, region=region)

    api_calls = []
    errors = []
    iteration = 0
    verified = False

    try:
        # Phase 0: Try deterministic handler first (most reliable for known task types)
        from handlers import try_handle
        deterministic_result = try_handle(client, model, prompt, files, tripletex)
        if deterministic_result:
            logger.info("Task handled deterministically — skipping LLM consensus")
            return deterministic_result

        # Phase 0b: Gather existing data from Tripletex (GET calls are free)
        logger.info("Phase 0b: Gathering preflight data...")
        preflight_data = gather_preflight_data(tripletex, prompt)
        if preflight_data:
            logger.info(f"Preflight data: {len(preflight_data)} chars")
        else:
            logger.info("No relevant preflight data found")

        # Phase 1: Three thinkers analyze in parallel
        user_content = build_user_message(prompt, files)
        # Inject preflight data into the user message
        if preflight_data:
            # Find the text block and append preflight data
            for block in user_content:
                if isinstance(block, dict) and block.get("type") == "text":
                    block["text"] += f"\n\n## EXISTING DATA IN TRIPLETEX (from preflight GET calls)\n{preflight_data}"
                    break

        logger.info("Phase 1: Launching 3 thinkers in parallel...")

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                pool.submit(think_about_task, client, model, user_content, i): i
                for i in range(1, 4)
            }
            plans = [None, None, None]
            for future in as_completed(futures):
                idx = futures[future] - 1
                plans[idx] = future.result()

        logger.info("Phase 1 complete: 3 plans received")

        # Phase 2: Reconcile into consensus
        logger.info("Phase 2: Reconciling plans...")
        consensus = reconcile_plans(client, model, plans, prompt)
        logger.info(f"Phase 2 complete: consensus plan ({len(consensus)} chars)")

        # Phase 3: Execute with tools
        logger.info("Phase 3: Executing consensus plan...")
        today = date.today().isoformat()
        preflight_section = f"\n\nEXISTING DATA IN TRIPLETEX:\n{preflight_data}" if preflight_data else ""
        executor_text = (
            f"Today's date is {today}.\n\n"
            f"ORIGINAL TASK:\n{prompt}\n\n"
            f"CONSENSUS PLAN (agreed by 3 independent analysts):\n{consensus}\n\n"
            f"Execute this plan now step by step IN ORDER. Each step depends on the previous ones.\n"
            f"- Follow the steps EXACTLY as written\n"
            f"- Use the EXACT values from the plan (names, amounts, dates, accounts)\n"
            f"- Do NOT skip any step\n"
            f"- Do NOT reorder steps — the dependency order is critical\n"
            f"- After each POST, note the returned ID — you'll need it for later steps\n"
            f"- Use entity IDs from the preflight data below — do NOT create duplicates"
            f"{preflight_section}"
        )

        # Build executor message with files for vision
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

        max_iterations = 25
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"Executor iteration {iteration}/{max_iterations}")

            response = client.messages.create(
                model=model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

            logger.info(f"  Stop reason: {response.stop_reason}")

            if response.stop_reason == "end_turn":
                # Verification: if we made write calls, verify the results
                write_calls_so_far = [c for c in api_calls if c["tool"] != "tripletex_get"]
                if write_calls_so_far and not verified:
                    verified = True
                    logger.info(f"Executor end_turn after {len(write_calls_so_far)} write calls — running verification...")
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({"role": "user", "content": [{
                        "type": "text",
                        "text": (
                            "VERIFICATION CHECK: Before finishing, verify your work:\n"
                            "1. GET back the key entities you created/modified (use the IDs from your POST/PUT responses)\n"
                            "2. Compare each field against the original task requirements\n"
                            "3. Check: correct amounts? correct dates? correct names? correct accounts? correct department/project links?\n"
                            "4. If a field is wrong, fix it with PUT (do NOT delete and recreate)\n"
                            "5. If everything is correct, confirm and stop\n\n"
                            "IMPORTANT: Do NOT delete any entities during verification. Only use PUT to fix wrong values.\n"
                            "Do this now — GET the entities and verify."
                        )
                    }]})
                    continue
                logger.info("Executor completed (end_turn)")
                break

            if response.stop_reason != "tool_use":
                logger.info(f"Executor stopped: {response.stop_reason}")
                break

            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            tool_results = []
            for block in assistant_content:
                if block.type == "tool_use":
                    logger.info(f"  Tool: {block.name}({json.dumps(block.input, default=str)[:200]})")
                    result = execute_tool(tripletex, block.name, block.input)

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
        logger.exception("Consensus agent error")
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
        "iterations": iteration,
        "total_calls": len(api_calls),
        "write_calls": len(write_calls),
        "error_count": len(errors),
        "write_error_count": len(write_errors),
        "errors": errors[:20],
        "api_calls": api_calls,
    }
    logger.info(f"TASK_RESULT: {json.dumps(summary, ensure_ascii=False)}")

    return {"status": "completed"}
