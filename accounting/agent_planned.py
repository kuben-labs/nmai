"""Planner-doer agent for Tripletex accounting tasks.

Architecture:
1. Planner (1 call): Breaks task into subtasks, tags each with domains
2. Doers (mini agent loop each): Get ONLY the API docs for their tagged domains
3. Coordinator (code): Passes entity IDs between doers, tracks errors
"""

import json
import logging
import os
import re
from datetime import date, datetime

import anthropic

from tripletex import TripletexClient
from prompts_v2 import PLANNER_PROMPT, FILE_HANDLING_PROMPT, build_doer_prompt
from agent import TOOLS, execute_tool, extract_file_content, build_user_message

logger = logging.getLogger(__name__)

DOER_PREFIX = """You are executing a single focused subtask in Tripletex.
Complete ONLY the described subtask. Do not add extra steps.
When you create or find the needed entity, note its ID and stop.

RULES:
- GET calls are free — use them to look up IDs
- Make exactly ONE write call (POST/PUT/DELETE) per subtask unless the task requires multiple
- $variable references have been replaced with actual IDs in the context below
- If a GET returns existing entities, use their IDs — don't create duplicates
"""


def run_agent_planned(prompt: str, files: list, base_url: str, session_token: str) -> dict:
    """Run the planner-doer agent to complete an accounting task."""
    tripletex = TripletexClient(base_url, session_token)

    project_id = os.getenv("GCP_PROJECT_ID", "ai-nm26osl-1759")
    region = os.getenv("CLAUDE_VERTEX_REGION", "global")
    model = os.getenv("CLAUDE_MODEL", "claude-opus-4-6")

    logger.info(f"Planned agent using {model} via Vertex AI (project={project_id}, region={region})")
    client = anthropic.AnthropicVertex(project_id=project_id, region=region)

    api_calls = []
    errors = []

    try:
        # Phase 1: Create plan
        plan = create_plan(client, model, prompt, files)
        logger.info(f"Plan: {len(plan)} subtasks")
        for s in plan:
            domains = s.get("domains", [])
            logger.info(f"  Step {s['id']}: {s['task'][:80]}... domains={domains} depends={s['depends_on']}")

        # Phase 2: Execute subtasks in dependency order
        context = {}
        completed = set()

        while len(completed) < len(plan):
            ready = [
                s for s in plan
                if s["id"] not in completed
                and all(d in completed for d in s["depends_on"])
            ]

            if not ready:
                logger.warning("No ready subtasks but not all complete — circular dependency?")
                break

            for subtask in ready:
                logger.info(f"Executing subtask {subtask['id']}: {subtask['task'][:80]}...")
                step_calls, step_errors, created_id = execute_subtask(
                    client, model, tripletex, subtask, context
                )
                api_calls.extend(step_calls)
                errors.extend(step_errors)

                if subtask.get("output_key") and created_id is not None:
                    context[subtask["output_key"]] = created_id
                    logger.info(f"  -> {subtask['output_key']} = {created_id}")

                completed.add(subtask["id"])

    except Exception as e:
        logger.exception("Planned agent error")
        errors.append({"tool": "agent", "endpoint": "", "status_code": 0,
                       "error_body": str(e), "input_data": ""})
    finally:
        tripletex.close()

    # Determine success
    write_errors = [e for e in errors if e["tool"] != "tripletex_get"]
    write_calls_list = [c for c in api_calls if c["tool"] != "tripletex_get"]

    if write_calls_list:
        last_write = write_calls_list[-1]
        has_errors = last_write["status_code"] >= 400
    else:
        has_errors = len(write_errors) > 0

    summary = {
        "task_summary": True,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "prompt": prompt[:500],
        "status": "FAILED" if has_errors else "OK",
        "iterations": len(plan),
        "total_calls": len(api_calls),
        "write_calls": len(write_calls_list),
        "error_count": len(errors),
        "write_error_count": len(write_errors),
        "errors": errors[:20],
        "api_calls": api_calls,
    }
    logger.info(f"TASK_RESULT: {json.dumps(summary, ensure_ascii=False)}")

    return {"status": "completed"}


def create_plan(client, model, prompt: str, files: list) -> list:
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


def execute_subtask(client, model, tripletex, subtask: dict, context: dict):
    """Execute a single subtask with a domain-focused system prompt."""
    api_calls = []
    errors = []
    created_id = None

    task_text = subtask["task"]

    # Replace $variable references with actual values
    for key, value in context.items():
        task_text = task_text.replace(f"${key}", str(value))

    # Build context string
    if context:
        ctx_lines = [f"  {k} = {v}" for k, v in context.items()]
        ctx_str = "\n".join(ctx_lines)
        user_text = f"Today's date is {date.today().isoformat()}.\n\nSUBTASK:\n{task_text}\n\nCONTEXT (entity IDs from previous steps):\n{ctx_str}"
    else:
        user_text = f"Today's date is {date.today().isoformat()}.\n\nSUBTASK:\n{task_text}"

    # Build focused system prompt from tagged domains
    domains = subtask.get("domains", [])
    domain_docs = build_doer_prompt(domains) if domains else build_doer_prompt(list(
        # Fallback: give all domains if planner didn't tag
        ["customer", "supplier", "employee", "invoice", "voucher", "travel",
         "project", "salary", "dimension", "department", "company"]
    ))
    doer_system = DOER_PREFIX + "\n\n" + domain_docs

    messages = [{"role": "user", "content": user_text}]

    max_iterations = 10
    for iteration in range(max_iterations):
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=doer_system,
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
                logger.info(f"    [{subtask['id']}] {block.name}({json.dumps(block.input, default=str)[:200]})")
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
