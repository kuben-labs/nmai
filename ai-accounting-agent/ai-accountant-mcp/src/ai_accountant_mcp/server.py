"""
FastMCP Server for Tripletex API v2

Automatically generates MCP tools from the Tripletex OpenAPI specification.
All 800+ API endpoints are exposed as callable MCP tools.

DYNAMIC CREDENTIALS SUPPORT:
This server now supports dynamic credentials per request. Credentials are
read from environment variables on EACH API call:
- TRIPLETEX_BASE_URL: API base URL (e.g., https://tx-proxy.ainm.no/v2)
- TRIPLETEX_SESSION_TOKEN: Session token for authentication

If dynamic credentials are not set, falls back to static .env credentials.
"""

import os
import json
import base64
import httpx
from fastmcp import FastMCP
from loguru import logger
from dotenv import load_dotenv
from typing import Any, Optional

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# Configuration - Supports Dynamic Overrides
# ============================================================================


# In-process dynamic credentials (updated via /update-credentials HTTP endpoint)
# This is process-safe: the agent calls the MCP server's HTTP endpoint to update these.
_dynamic_base_url: Optional[str] = None
_dynamic_session_token: Optional[str] = None


def set_dynamic_credentials(base_url: str, session_token: str):
    """Update dynamic credentials in this process (called via HTTP endpoint)."""
    global _dynamic_base_url, _dynamic_session_token
    _dynamic_base_url = base_url
    _dynamic_session_token = session_token
    # Also set env vars so os.getenv fallback works consistently
    os.environ["TRIPLETEX_BASE_URL"] = base_url
    os.environ["TRIPLETEX_SESSION_TOKEN"] = session_token
    logger.info(
        f"Dynamic credentials updated: base_url={base_url}, token={session_token[:20]}..."
    )


def get_credentials():
    """
    Get Tripletex credentials, preferring dynamic runtime values.

    This is called on EACH request to allow dynamic credential updates.

    Priority:
    1. In-process dynamic credentials (set via /update-credentials endpoint)
    2. TRIPLETEX_* env vars
    3. Static .env values (fallback)
    """
    # In-process dynamic credentials (highest priority)
    base_url = _dynamic_base_url
    session_token = _dynamic_session_token

    # Fallback to env vars
    if not base_url:
        base_url = os.getenv("TRIPLETEX_BASE_URL")
    if not session_token:
        session_token = os.getenv("TRIPLETEX_SESSION_TOKEN")

    # Fallback to static credentials
    if not base_url:
        base_url = os.getenv("API_URL", "https://kkpqfuj-amager.tripletex.dev/v2")
    if not session_token:
        session_token = os.getenv("SESSION_TOKEN")

    return base_url, session_token


def encode_session_token(token: str) -> str:
    """
    Encode session token in Tripletex Basic Auth format.

    Tripletex format: Authorization: Basic base64("0:" + SESSION_TOKEN)
    The username is always "0", password is the session token.
    """
    if not token:
        raise ValueError("SESSION_TOKEN not set")

    credentials = f"0:{token}"
    encoded = base64.b64encode(credentials.encode()).decode()
    return f"Basic {encoded}"


# Initial static credentials (for server startup / OpenAPI spec fetch)
SESSION_TOKEN = os.getenv("SESSION_TOKEN")
API_URL = os.getenv("API_URL", "https://kkpqfuj-amager.tripletex.dev/v2")
OPENAPI_URL = f"{API_URL}/openapi.json"
LOGIN_EMAIL = os.getenv("LOGIN_URL")  # Email address (login)
WEB_UI = os.getenv("WEB_UI", "https://kkpqfuj-amager.tripletex.dev/")


# ============================================================================
# Dynamic Credentials HTTP Client
# ============================================================================


class DynamicAuthClient(httpx.AsyncClient):
    """
    HTTP client that reads credentials on EACH request.

    This allows credentials to be updated via environment variables
    between requests without restarting the server.

    IMPORTANT: FastMCP's OpenAPI provider calls client.send(request) not
    client.request(), and reads client.base_url to build URLs. So we must:
    1. Update self._base_url when dynamic credentials change
    2. Override send() to inject auth headers on prepared requests
    """

    def __init__(self, fallback_base_url: str, fallback_token: str, **kwargs):
        # Set base_url and default auth headers for FastMCP compatibility
        auth_header = encode_session_token(fallback_token) if fallback_token else ""
        headers = {"Authorization": auth_header, "Content-Type": "application/json"}
        super().__init__(
            base_url=fallback_base_url, timeout=30.0, headers=headers, **kwargs
        )
        self._fallback_base_url = fallback_base_url
        self._fallback_token = fallback_token
        self._current_base_url = fallback_base_url
        self._current_token = fallback_token

    def _sync_credentials(self):
        """Sync dynamic credentials into the client's base_url and headers.

        Called before every request to ensure FastMCP's OpenAPI provider
        reads the correct base_url and auth headers.
        """
        base_url, token = get_credentials()
        base_url = base_url or self._fallback_base_url
        token = token or self._fallback_token

        # Update base_url if it changed (FastMCP reads this via client.base_url)
        if base_url != self._current_base_url:
            logger.info(f"DynamicAuthClient: Switching base_url to {base_url}")
            # Ensure trailing slash for correct URL joining by httpx
            base_url_with_slash = base_url.rstrip("/") + "/"
            self._base_url = httpx.URL(base_url_with_slash)
            self._current_base_url = base_url

        # Update default auth headers if token changed
        if token != self._current_token:
            logger.info(f"DynamicAuthClient: Updating auth token")
            auth_header = encode_session_token(token)
            self.headers["Authorization"] = auth_header
            self._current_token = token

    async def send(self, request: httpx.Request, **kwargs) -> httpx.Response:
        """Override send() - FastMCP's OpenAPI provider uses this path.

        FastMCP builds a full request using client.base_url and then calls
        client.send(request). We must sync credentials before sending.
        """
        self._sync_credentials()

        # If the request was built with the old base_url, rebuild the URL
        request_url = str(request.url)
        if (
            self._current_base_url != self._fallback_base_url
            and self._fallback_base_url in request_url
        ):
            new_url = request_url.replace(
                self._fallback_base_url.rstrip("/"),
                self._current_base_url.rstrip("/"),
            )
            logger.debug(
                f"DynamicAuthClient.send: Rewriting URL {request_url[:80]} -> {new_url[:80]}"
            )
            request.url = httpx.URL(new_url)

        # Inject current auth headers into the request
        auth_header = encode_session_token(self._current_token)
        request.headers["Authorization"] = auth_header
        request.headers["Content-Type"] = "application/json"

        logger.debug(
            f"DynamicAuthClient.send: {request.method} {str(request.url)[:120]}"
        )
        return await super().send(request, **kwargs)

    async def request(self, method: str, url: Any, **kwargs) -> httpx.Response:
        """Override request() for direct API calls (fallback path)."""
        self._sync_credentials()

        url_str = str(url)

        # Inject auth headers
        headers = dict(kwargs.get("headers", {}) or {})
        headers["Authorization"] = encode_session_token(self._current_token)
        headers["Content-Type"] = "application/json"
        kwargs["headers"] = headers

        # If URL is relative, it will use self._base_url (already synced above)
        logger.debug(f"DynamicAuthClient.request: {method} {url_str[:100]}")
        return await super().request(method, url_str, **kwargs)


# ============================================================================
# Configuration Validation
# ============================================================================


def validate_config():
    """Validate that all required environment variables are set for startup."""
    errors = []

    if not SESSION_TOKEN:
        errors.append("SESSION_TOKEN not set in .env (needed for startup)")
    if not API_URL:
        errors.append("API_URL not set in .env")

    if errors:
        logger.warning("Configuration warnings:")
        for error in errors:
            logger.warning(f"  ⚠ {error}")
        logger.info("")
        logger.info("Note: Dynamic credentials can be set via TRIPLETEX_* env vars")
        logger.info("  TRIPLETEX_BASE_URL=<url>")
        logger.info("  TRIPLETEX_SESSION_TOKEN=<token>")
        return False

    logger.info("✓ Configuration validation passed:")
    logger.info(f"  - API_URL: {API_URL}")
    if LOGIN_EMAIL:
        logger.info(f"  - LOGIN_EMAIL: {LOGIN_EMAIL}")
    if SESSION_TOKEN:
        token_preview = (
            SESSION_TOKEN[:20] + "..." + SESSION_TOKEN[-10:]
            if len(SESSION_TOKEN) > 30
            else SESSION_TOKEN
        )
        logger.info(f"  - SESSION_TOKEN: {token_preview}")
    return True


# ============================================================================
# Initialize MCP Server from OpenAPI Spec
# ============================================================================


def create_mcp_server():
    """
    Create and configure the MCP server from Tripletex OpenAPI spec.

    Key points:
    1. Validates configuration from .env
    2. Fetches OpenAPI spec from Tripletex
    3. Creates DynamicAuthClient that reads credentials per-request
    4. Generates MCP tools from all OpenAPI endpoints
    5. Returns the configured MCP server

    The DynamicAuthClient will read TRIPLETEX_* env vars on each request,
    allowing credentials to be updated without restarting the server.
    """
    logger.info("Initializing Tripletex MCP Server...")
    logger.info("")

    # Validate configuration (warnings only, don't fail)
    has_static_config = validate_config()
    logger.info("")

    # Try to fetch OpenAPI spec
    openapi_spec = None

    # First try with static credentials
    if SESSION_TOKEN and API_URL:
        auth_header = encode_session_token(SESSION_TOKEN)
        logger.info(f"Fetching OpenAPI spec from {OPENAPI_URL}")
        try:
            response = httpx.get(
                OPENAPI_URL, headers={"Authorization": auth_header}, timeout=30.0
            )
            response.raise_for_status()
            openapi_spec = response.json()
            logger.info("✓ OpenAPI spec loaded successfully")
        except Exception as e:
            logger.warning(f"⚠ Could not fetch OpenAPI spec: {e}")

    # Try loading from cached file if API fetch failed
    if openapi_spec is None:
        cached_spec_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "data", "raw", "openapi.json"
        )
        if os.path.exists(cached_spec_path):
            logger.info(f"Loading OpenAPI spec from cache: {cached_spec_path}")
            try:
                with open(cached_spec_path, "r") as f:
                    openapi_spec = json.load(f)
                logger.info("✓ OpenAPI spec loaded from cache")
            except Exception as e:
                logger.error(f"✗ Failed to load cached OpenAPI spec: {e}")
                raise
        else:
            raise ValueError(
                "No OpenAPI spec available (API fetch failed and no cache)"
            )

    # Create dynamic auth client
    # This client reads credentials from env vars on EACH request
    client = DynamicAuthClient(
        fallback_base_url=API_URL or "",
        fallback_token=SESSION_TOKEN or "",
    )

    logger.info("✓ Dynamic auth HTTP client created")
    logger.info("  Credentials will be read from env vars on each request")

    # Generate MCP from OpenAPI spec
    logger.info("Generating MCP tools from OpenAPI spec...")
    try:
        mcp = FastMCP.from_openapi(
            openapi_spec=openapi_spec,
            client=client,
            name="tripletex-mcp",
            version="0.1.0",
            tags={"tripletex", "accounting", "api-v2"},
        )

        # CRITICAL: Strip outputSchema from all tools.
        # Tripletex API responses often have null fields where the OpenAPI
        # spec says object is expected. The MCP SDK validates responses against
        # outputSchema and rejects valid API responses (e.g., created entity
        # with null manager field). This causes the agent to never see successful
        # results and lose created entity IDs, causing cascading failures.
        try:
            tool_manager = getattr(mcp, "_tool_manager", None)
            if tool_manager:
                tools_dict = getattr(tool_manager, "_tools", {})
                stripped_count = 0
                for tool in tools_dict.values():
                    # FastMCP Tool model has 'output_schema' field
                    if (
                        hasattr(tool, "output_schema")
                        and tool.output_schema is not None
                    ):
                        tool.output_schema = None
                        stripped_count += 1
                logger.info(
                    f"✓ Stripped output_schema from {stripped_count} tools (prevents false validation errors)"
                )
            else:
                logger.warning("⚠ Could not find tool_manager to strip output_schema")
        except Exception as e:
            logger.warning(f"⚠ Failed to strip output_schema: {e} (non-fatal)")

        endpoint_count = len(openapi_spec.get("paths", {}))
        logger.info(f"✓ MCP server created with {endpoint_count} endpoints")
        logger.info(f"✓ Generated ~800 MCP tools from OpenAPI")

    except Exception as e:
        logger.error(f"✗ Failed to generate MCP from OpenAPI: {e}")
        raise

    return mcp


# ============================================================================
# Create Server Instance
# ============================================================================

try:
    mcp = create_mcp_server()

    # Add custom HTTP endpoints for credential management and health checks
    from starlette.requests import Request
    from starlette.responses import JSONResponse as StarletteJSONResponse

    @mcp.custom_route("/update-credentials", methods=["POST"])
    async def update_credentials(request: Request) -> StarletteJSONResponse:
        """Update Tripletex credentials dynamically (called by agent per-request)."""
        try:
            body = await request.json()
            base_url = body.get("base_url", "")
            session_token = body.get("session_token", "")
            if not base_url or not session_token:
                return StarletteJSONResponse(
                    {"error": "base_url and session_token required"}, status_code=400
                )
            set_dynamic_credentials(base_url, session_token)
            return StarletteJSONResponse({"status": "updated"})
        except Exception as e:
            logger.error(f"Failed to update credentials: {e}")
            return StarletteJSONResponse({"error": str(e)}, status_code=500)

    @mcp.custom_route("/health", methods=["GET"])
    async def mcp_health(request: Request) -> StarletteJSONResponse:
        """Health check for MCP server."""
        return StarletteJSONResponse(
            {"status": "healthy", "service": "ai-accountant-mcp", "version": "0.1.0"}
        )

    logger.info("")
    logger.info("✓ MCP server initialized successfully")
    logger.info("✓ Custom routes: /update-credentials (POST), /health (GET)")
    logger.info("")
except Exception as e:
    logger.error("")
    logger.error(f"✗ FATAL: Failed to initialize MCP server")
    logger.error(f"  Error: {e}")
    logger.error("")

    # Create a minimal fallback server that reports the error
    mcp = FastMCP(name="tripletex-mcp", version="0.1.0")

    @mcp.tool()
    def initialization_error() -> str:
        """
        MCP server initialization failed.

        This tool indicates the server did not start correctly.

        Check the logs above for the specific error.

        Common issues:
        - Missing or invalid SESSION_TOKEN in .env
        - Invalid API_URL in .env
        - Network connectivity to Tripletex API
        - Invalid or expired session token
        - Tripletex API is down
        """
        return (
            f"MCP Server failed to initialize: {str(e)}\n\n"
            f"Check the startup logs for details.\n"
            f"Ensure .env file has valid SESSION_TOKEN and API_URL."
        )


# ============================================================================
# Log Server Information
# ============================================================================


def log_startup_info():
    """Log startup information."""
    logger.info("")
    logger.info("╔" + "=" * 78 + "╗")
    logger.info("║ " + "TRIPLETEX MCP SERVER READY".ljust(76) + " ║")
    logger.info("║ " + "=" * 76 + " ║")
    logger.info(f"║ Fallback URL:  {(API_URL or 'not set').ljust(56)} ║")
    logger.info(f"║ Web UI:        {(WEB_UI or 'not set').ljust(56)} ║")
    logger.info(f"║ Tools Ready:   ~800 MCP tools available".ljust(77) + "║")
    logger.info(f"║ Listen On:     http://0.0.0.0:8083/mcp".ljust(77) + "║")
    logger.info("║ " + "=" * 76 + " ║")
    logger.info(
        "║ DYNAMIC AUTH: Reads TRIPLETEX_* env vars on each request".ljust(77) + "║"
    )
    logger.info(
        "║ Set TRIPLETEX_BASE_URL and TRIPLETEX_SESSION_TOKEN dynamically".ljust(77)
        + "║"
    )
    logger.info("╚" + "=" * 78 + "╝")
    logger.info("")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    logger.info("Starting Tripletex MCP Server...")
    logger.info("Press Ctrl+C to stop")

    try:
        # Log startup info
        if mcp and isinstance(mcp, FastMCP):
            log_startup_info()

        # Run MCP server on port 8083
        # HTTP transport for compatibility with coordinator
        mcp.run(transport="http", host="0.0.0.0", port=8083)
    except KeyboardInterrupt:
        logger.info("")
        logger.info("Shutting down MCP server...")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        raise
