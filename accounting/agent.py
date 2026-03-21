"""Claude-powered agentic loop for Tripletex accounting tasks."""

import json
import logging
import base64
import os
from datetime import date, datetime

import anthropic

from tripletex import TripletexClient
from prompts import SYSTEM_PROMPT, FILE_HANDLING_PROMPT

logger = logging.getLogger(__name__)

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
    """Build the user message content blocks, including images for vision."""
    content_blocks = []

    # Add any image files as vision content
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


def run_agent(prompt: str, files: list, base_url: str, session_token: str) -> dict:
    """Run the agentic loop to complete an accounting task."""
    tripletex = TripletexClient(base_url, session_token)

    project_id = os.getenv("GCP_PROJECT_ID", "ai-nm26osl-1759")
    region = os.getenv("CLAUDE_VERTEX_REGION", "global")
    model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

    logger.info(f"Using Claude {model} via Vertex AI (project={project_id}, region={region})")
    client = anthropic.AnthropicVertex(project_id=project_id, region=region)

    user_content = build_user_message(prompt, files)
    messages = [{"role": "user", "content": user_content}]

    max_iterations = 30
    iteration = 0

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
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

            logger.info(f"  Stop reason: {response.stop_reason}")

            if response.stop_reason == "end_turn":
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
        "prompt": prompt[:500],
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
