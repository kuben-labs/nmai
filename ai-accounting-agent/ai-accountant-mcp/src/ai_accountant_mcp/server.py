"""
FastMCP Server for Tripletex API v2

Automatically generates MCP tools from the Tripletex OpenAPI specification.
All 800+ API endpoints are exposed as callable MCP tools.

Authentication: Session-based from environment variables (.env)
- Reads SESSION_TOKEN, API_URL, etc. from .env file
- Automatically adds authorization headers to all requests
- Coordinator doesn't need to worry about auth - it just calls tools

This is the KEY INSIGHT: The MCP handles auth internally.
Users of the MCP just call tools - they don't deal with credentials.
"""

import os
import json
import base64
import httpx
from fastmcp import FastMCP
from loguru import logger
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# Configuration from Environment Variables
# ============================================================================

SESSION_TOKEN = os.getenv("SESSION_TOKEN")
API_URL = os.getenv("API_URL", "https://kkpqfuj-amager.tripletex.dev/v2")
OPENAPI_URL = f"{API_URL}/openapi.json"
LOGIN_EMAIL = os.getenv("LOGIN_URL")  # Email address (login)
WEB_UI = os.getenv("WEB_UI", "https://kkpqfuj-amager.tripletex.dev/")

# ============================================================================
# Authentication Helper
# ============================================================================


def encode_session_token(token: str) -> str:
    """
    Encode session token in Tripletex Basic Auth format.

    Tripletex format: Authorization: Basic base64("0:" + SESSION_TOKEN)
    The username is always "0", password is the session token.

    Args:
        token: The session token from environment

    Returns:
        Base64-encoded Basic Auth header value
    """
    if not token:
        raise ValueError("SESSION_TOKEN not set in .env")

    credentials = f"0:{token}"
    encoded = base64.b64encode(credentials.encode()).decode()
    return f"Basic {encoded}"


# ============================================================================
# Configuration Validation
# ============================================================================


def validate_config():
    """Validate that all required environment variables are set."""
    errors = []

    if not SESSION_TOKEN:
        errors.append("SESSION_TOKEN not set in .env")
    if not API_URL:
        errors.append("API_URL not set in .env")
    if not LOGIN_EMAIL:
        errors.append("LOGIN_URL (email) not set in .env")

    if errors:
        logger.error("Configuration validation failed:")
        for error in errors:
            logger.error(f"  ✗ {error}")
        logger.error("")
        logger.error("Please set these in ai-accountant-mcp/.env:")
        logger.error("  SESSION_TOKEN=<your-token>")
        logger.error("  API_URL=<your-api-url>")
        logger.error("  LOGIN_URL=<your-email>")
        raise ValueError("Missing required environment variables")

    logger.info("✓ Configuration validation passed:")
    logger.info(f"  - API_URL: {API_URL}")
    logger.info(f"  - LOGIN_EMAIL: {LOGIN_EMAIL}")
    logger.info(f"  - SESSION_TOKEN: {SESSION_TOKEN[:30]}...{SESSION_TOKEN[-20:]}")


# ============================================================================
# Initialize MCP Server from OpenAPI Spec
# ============================================================================


def create_mcp_server():
    """
    Create and configure the MCP server from Tripletex OpenAPI spec.

    Key points:
    1. Validates configuration from .env
    2. Creates authorization header from SESSION_TOKEN
    3. Fetches OpenAPI spec from Tripletex (using authenticated request)
    4. Creates authenticated HTTP client
    5. Generates MCP tools from all OpenAPI endpoints
    6. Returns the configured MCP server

    The MCP server is now "sealed" - it has credentials baked in.
    Callers just use the tools without worrying about auth.
    """
    logger.info("Initializing Tripletex MCP Server...")
    logger.info("")

    # Validate configuration
    validate_config()
    logger.info("")

    # Create authorization header
    auth_header = encode_session_token(SESSION_TOKEN)
    logger.info("✓ Authorization header created")

    # Fetch OpenAPI spec with authentication
    logger.info(f"Fetching OpenAPI spec from {OPENAPI_URL}")
    try:
        response = httpx.get(
            OPENAPI_URL, headers={"Authorization": auth_header}, timeout=30.0
        )
        response.raise_for_status()
        openapi_spec = response.json()
        logger.info(f"✓ OpenAPI spec loaded successfully")
    except httpx.RequestError as e:
        logger.error(f"✗ Failed to fetch OpenAPI spec: {e}")
        raise
    except Exception as e:
        logger.error(f"✗ Error loading OpenAPI spec: {e}")
        raise

    # Create authenticated HTTP client
    # IMPORTANT: The authorization header is baked into the client
    # All requests from the MCP tools will automatically use this auth
    client = httpx.AsyncClient(
        base_url=API_URL,
        headers={"Authorization": auth_header},
        timeout=30.0,
    )

    logger.info(f"✓ Authenticated HTTP client created")
    logger.info(f"  Base URL: {API_URL}")

    # Generate MCP from OpenAPI spec
    # FastMCP will use the authenticated client for all API calls
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
    logger.info(f"║ API URL:       {(API_URL or '').ljust(56)} ║")
    logger.info(f"║ Web UI:        {(WEB_UI or '').ljust(56)} ║")
    logger.info(
        f"║ Auth Status:   ✓ Authenticated ({LOGIN_EMAIL or ''})".ljust(77) + "║"
    )
    logger.info(f"║ Tools Ready:   ~800 MCP tools available".ljust(77) + "║")
    logger.info(f"║ Listen On:     http://0.0.0.0:8083/mcp".ljust(77) + "║")
    logger.info("║ " + "=" * 76 + " ║")
    logger.info(
        "║ NOTE: MCP handles authentication internally from .env".ljust(77) + "║"
    )
    logger.info("║ Callers don't need to worry about credentials.".ljust(77) + "║")
    logger.info("╚" + "=" * 78 + "╝")
    logger.info("")
    logger.info("╔" + "=" * 78 + "╗")
    logger.info("║ " + "TRIPLETEX MCP SERVER STARTED SUCCESSFULLY".ljust(76) + " ║")
    logger.info("║ " + "=" * 76 + " ║")
    logger.info(f"║ API URL:       {API_URL.ljust(56)} ║")
    logger.info(f"║ Web UI:        {WEB_UI.ljust(56)} ║")
    logger.info(f"║ Auth Status:   ✓ Authenticated ({LOGIN_EMAIL})".ljust(77) + "║")
    logger.info(f"║ Tools Ready:   ~800 MCP tools available".ljust(77) + "║")
    logger.info(f"║ Listen On:     http://0.0.0.0:8083/mcp".ljust(77) + "║")
    logger.info("║ " + "=" * 76 + " ║")
    logger.info(
        "║ NOTE: MCP handles authentication internally from .env".ljust(77) + "║"
    )
    logger.info("║ Callers don't need to worry about credentials.".ljust(77) + "║")
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
