"""Generate pydantic-ai Tools directly from an OpenAPI spec.

No MCP server needed. Each OpenAPI endpoint becomes a Tool.from_schema
that makes direct HTTP calls via httpx with Basic Auth.

This eliminates:
- MCP server process management
- Output schema validation errors
- URL rewriting complexity
- Credential passing between processes
"""

import json
import base64
from typing import Dict, List, Any, Optional, Tuple

import httpx
from loguru import logger
from pydantic_ai import Tool


def _make_auth_header(session_token: str) -> str:
    """Create Tripletex Basic Auth header: base64("0:" + session_token)."""
    credentials = f"0:{session_token}"
    encoded = base64.b64encode(credentials.encode()).decode()
    return f"Basic {encoded}"


def _simplify_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Simplify a JSON schema so Gemini can process it.

    Resolves $ref inline, strips $defs, limits depth.
    """
    defs = schema.get("$defs", {}) or schema.get("definitions", {})

    def _resolve(s: Any, depth: int = 0, seen: Optional[set] = None) -> Any:
        if seen is None:
            seen = set()
        if not isinstance(s, dict):
            return s

        if "$ref" in s:
            ref_path = s["$ref"]
            ref_name = ref_path.rsplit("/", 1)[-1] if "/" in ref_path else ref_path
            if ref_name in seen:
                return {"type": "object"}
            if ref_name in defs:
                seen = seen | {ref_name}
                return _resolve(defs[ref_name], depth, seen)
            return {"type": "object"}

        if depth > 3:
            schema_type = s.get("type")
            if schema_type == "object":
                return {"type": "object"}
            elif schema_type == "array":
                return {"type": "array", "items": {"type": "object"}}
            return {
                k: v for k, v in s.items() if k not in ("$ref", "$defs", "definitions")
            }

        result = {}
        for k, v in s.items():
            if k in ("$defs", "definitions"):
                continue
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
    # Safety: strip any remaining $ref
    s = json.dumps(simplified, default=str)
    if "$ref" in s:
        simplified = json.loads(s.replace('"$ref"', '"_removed_ref"'))
    return simplified


def _extract_params_schema(
    operation: Dict[str, Any], spec: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[str], Optional[str]]:
    """Extract a unified JSON schema for an endpoint's parameters + body.

    Returns (schema, query_param_names, body_content_type).
    """
    properties = {}
    required = []
    query_params = []

    # Path and query parameters
    for param in operation.get("parameters", []):
        # Resolve $ref in parameters
        if "$ref" in param:
            ref_path = param["$ref"]
            parts = ref_path.strip("#/").split("/")
            resolved = spec
            for part in parts:
                resolved = resolved.get(part, {})
            param = resolved

        name = param.get("name", "")
        if not name:
            continue

        param_schema = param.get("schema", {"type": "string"})
        description = param.get("description", "")

        if description:
            param_schema = {**param_schema, "description": description}

        properties[name] = param_schema

        if param.get("required", False):
            required.append(name)

        if param.get("in") in ("query", "path"):
            query_params.append(name)

    # Request body
    body_content_type = None
    request_body = operation.get("requestBody", {})
    if request_body:
        content = request_body.get("content", {})
        # Match content-type flexibly (e.g., "application/json; charset=utf-8")
        matched_ct = None
        for ct_key in content:
            if "json" in ct_key.lower():
                matched_ct = ct_key
                break
            elif "form" in ct_key.lower():
                matched_ct = ct_key
                break
        if matched_ct:
            ct = matched_ct
            body_schema = content[ct].get("schema", {})
            body_content_type = ct

            # If body has properties, merge them
            if "properties" in body_schema:
                for prop_name, prop_schema in body_schema["properties"].items():
                    if prop_name not in properties:
                        properties[prop_name] = prop_schema
                for req in body_schema.get("required", []):
                    if req not in required:
                        required.append(req)
            elif body_schema.get("type") == "object" or "$ref" in body_schema:
                # Resolve $ref for body
                resolved_body = body_schema
                if "$ref" in body_schema:
                    ref_path = body_schema["$ref"]
                    ref_name = ref_path.rsplit("/", 1)[-1]
                    all_defs = spec.get("components", {}).get("schemas", {})
                    if ref_name in all_defs:
                        resolved_body = all_defs[ref_name]

                if "properties" in resolved_body:
                    for prop_name, prop_schema in resolved_body["properties"].items():
                        if prop_name not in properties:
                            properties[prop_name] = prop_schema
                    for req in resolved_body.get("required", []):
                        if req not in required:
                            required.append(req)
            else:
                # Treat as single "body" parameter
                properties["body"] = body_schema
                if request_body.get("required", False):
                    required.append("body")

    schema = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required

    return schema, query_params, body_content_type


def generate_tools_from_openapi(
    spec: Dict[str, Any],
    base_url: str,
    session_token: str,
    tool_filter: Optional[set] = None,
) -> List[Tool]:
    """Generate pydantic-ai Tools from an OpenAPI spec.

    Args:
        spec: Parsed OpenAPI spec JSON
        base_url: Tripletex API base URL (e.g., "https://tx-proxy.../v2")
        session_token: Tripletex session token for Basic Auth
        tool_filter: Optional set of tool names to include (None = all)

    Returns:
        List of Tool objects ready to use with Agent
    """
    tools = []
    used_names: set = set()
    all_defs = spec.get("components", {}).get("schemas", {})
    # Merge top-level $defs if any
    top_defs = spec.get("$defs", {})

    paths = spec.get("paths", {})
    auth_header = _make_auth_header(session_token)

    for path, path_item in paths.items():
        for method in ["get", "post", "put", "delete", "patch"]:
            operation = path_item.get(method)
            if not operation:
                continue

            operation_id = operation.get("operationId", "")
            if not operation_id:
                continue

            # Convert operationId to tool name
            # e.g. "[Employee_post]" -> "Employee_post"
            tool_name = operation_id.strip("[]").replace(" ", "_")

            # Gemini requires: start with [a-zA-Z_], then [a-zA-Z0-9_.:- ], max 64 chars
            import re

            tool_name = re.sub(r"[^a-zA-Z0-9_.:\-]", "_", tool_name)
            if tool_name and not re.match(r"^[a-zA-Z_]", tool_name):
                tool_name = f"t_{tool_name}"
            if len(tool_name) > 64:
                tool_name = tool_name[:64]

            # Deduplicate
            if tool_name in used_names:
                for i in range(2, 100):
                    candidate = f"{tool_name[:61]}_{i}"
                    if candidate not in used_names:
                        tool_name = candidate
                        break
            used_names.add(tool_name)

            # Filter if specified — check both sanitized name and original operationId
            original_name = operation_id.strip("[]").replace(" ", "_")
            if (
                tool_filter
                and tool_name not in tool_filter
                and original_name not in tool_filter
            ):
                continue

            description = operation.get("summary", "") or operation.get(
                "description", ""
            )
            if not description:
                description = f"{method.upper()} {path}"

            # Truncate long descriptions (Gemini has limits)
            if len(description) > 500:
                description = description[:497] + "..."

            # Extract parameter schema
            try:
                raw_schema, query_params, body_content_type = _extract_params_schema(
                    operation, spec
                )
            except Exception as e:
                logger.debug(f"Skipping {tool_name}: schema extraction failed: {e}")
                continue

            # Simplify schema for Gemini
            # Inject component schemas as $defs for resolution
            if all_defs:
                raw_schema["$defs"] = all_defs
            if top_defs:
                raw_schema.setdefault("$defs", {}).update(top_defs)

            simplified_schema = _simplify_schema(raw_schema)

            # Build path parameters set
            path_params = set()
            import re

            for match in re.finditer(r"\{(\w+)\}", path):
                path_params.add(match.group(1))

            # Create the tool function (closure captures method, path, etc.)
            def _make_tool_fn(
                _method: str,
                _path: str,
                _path_params: set,
                _query_params: list,
                _body_content_type: Optional[str],
                _tool_name: str,
            ):
                async def tool_fn(**kwargs) -> str:
                    """Execute a Tripletex API call."""
                    try:
                        # Build URL with path parameters
                        url_path = _path
                        for pp in _path_params:
                            if pp in kwargs:
                                url_path = url_path.replace(
                                    f"{{{pp}}}", str(kwargs.pop(pp))
                                )

                        full_url = f"{base_url.rstrip('/')}{url_path}"

                        # Separate query params from body
                        query = {}
                        body = {}
                        for k, v in kwargs.items():
                            if k in _query_params or k in _path_params:
                                query[k] = v
                            else:
                                body[k] = v

                        # For GET/DELETE, all non-path params go to query
                        if _method in ("get", "delete"):
                            query.update(body)
                            body = {}

                        headers = {
                            "Authorization": auth_header,
                            "Content-Type": "application/json",
                        }

                        async with httpx.AsyncClient(timeout=30.0) as client:
                            response = await client.request(
                                method=_method.upper(),
                                url=full_url,
                                params=query if query else None,
                                json=body
                                if body and _method in ("post", "put", "patch")
                                else None,
                                headers=headers,
                            )

                        # Return the response regardless of status code
                        # (let the LLM see error messages and adapt)
                        try:
                            result = response.json()
                        except Exception:
                            result = response.text

                        if response.status_code >= 400:
                            return json.dumps(
                                {
                                    "error": True,
                                    "status_code": response.status_code,
                                    "response": result,
                                },
                                default=str,
                            )

                        return json.dumps(result, default=str)

                    except httpx.TimeoutException:
                        return json.dumps(
                            {"error": True, "message": "Request timed out"}
                        )
                    except Exception as e:
                        return json.dumps({"error": True, "message": str(e)})

                return tool_fn

            fn = _make_tool_fn(
                method, path, path_params, query_params, body_content_type, tool_name
            )

            try:
                tool = Tool.from_schema(
                    function=fn,
                    name=tool_name,
                    description=description,
                    json_schema=simplified_schema,
                    takes_ctx=False,
                )
                tools.append(tool)
            except Exception as e:
                logger.debug(f"Skipping {tool_name}: Tool creation failed: {e}")
                continue

    logger.info(f"Generated {len(tools)} tools from OpenAPI spec ({len(paths)} paths)")
    return tools


async def fetch_openapi_spec(api_url: str, session_token: str) -> Dict[str, Any]:
    """Fetch the OpenAPI spec from a Tripletex instance.

    Args:
        api_url: Base API URL (e.g., "https://kkpqfuj-amager.tripletex.dev/v2")
        session_token: Session token for auth

    Returns:
        Parsed OpenAPI spec as dict
    """
    openapi_url = f"{api_url.rstrip('/')}/openapi.json"
    auth_header = _make_auth_header(session_token)

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            openapi_url,
            headers={"Authorization": auth_header},
        )
        response.raise_for_status()
        return response.json()
