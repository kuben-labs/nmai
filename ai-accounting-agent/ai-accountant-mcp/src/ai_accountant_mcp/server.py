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


def get_credentials():
    """
    Get Tripletex credentials, preferring dynamic runtime values.

    This is called on EACH request to allow dynamic credential updates.

    Priority:
    1. TRIPLETEX_* env vars (set dynamically per request)
    2. Static .env values (fallback)
    """
    # Dynamic credentials (set per-request)
    base_url = os.getenv("TRIPLETEX_BASE_URL")
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

    def _get_current_auth(self) -> tuple[str, str]:
        """Get current credentials from environment."""
        base_url, token = get_credentials()
        base_url = base_url or self._fallback_base_url
        token = token or self._fallback_token
        return base_url, token

    async def request(self, method: str, url: Any, **kwargs) -> httpx.Response:
        """Override request to inject dynamic credentials."""
        base_url, token = self._get_current_auth()

        # Convert URL to string
        url_str = str(url)

        # If dynamic credentials differ from fallback, update headers
        if token != self._fallback_token:
            headers = dict(kwargs.get("headers", {}) or {})
            headers["Authorization"] = encode_session_token(token)
            headers["Content-Type"] = "application/json"
            kwargs["headers"] = headers

        # Check if we need to use a different base URL
        if (
            base_url
            and base_url != self._fallback_base_url
            and not url_str.startswith("http")
        ):
            url_str = f"{base_url.rstrip('/')}/{url_str.lstrip('/')}"
            # When using full URL, headers from __init__ aren't applied automatically
            headers = dict(kwargs.get("headers", {}) or {})
            if "Authorization" not in headers:
                headers["Authorization"] = encode_session_token(token)
            headers["Content-Type"] = "application/json"
            kwargs["headers"] = headers

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
    logger.info("")
    logger.info("✓ MCP server initialized successfully")
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
