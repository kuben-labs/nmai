"""
FastMCP Server for Tripletex API v2

This server automatically generates MCP tools from the Tripletex OpenAPI specification.
All 800+ API endpoints are exposed as MCP tools that LLMs can call.

Authentication: Uses session-based authentication (Basic Auth with username=0)
- Session token provided per-request in Authorization header
- No env vars needed - token passed dynamically by coordinator

Usage:
    python server.py

    Starts MCP server on http://0.0.0.0:8083/mcp
"""

import json
import base64
import httpx
from fastmcp import FastMCP
from loguru import logger

# ============================================================================
# Configuration
# ============================================================================

TRIPLETEX_BASE_URL = "https://kkpqfuj-amager.tripletex.dev/v2"
TRIPLETEX_OPENAPI_URL = "https://kkpqfuj-amager.tripletex.dev/v2/openapi.json"

# ============================================================================
# Initialize MCP from OpenAPI Spec
# ============================================================================


def create_mcp_server():
    """
    Create and configure the MCP server from Tripletex OpenAPI spec.

    This function:
    1. Fetches the OpenAPI spec from Tripletex
    2. Creates an HTTP client (unauthenticated initially)
    3. Generates MCP tools from all OpenAPI endpoints
    4. Returns the configured MCP server

    Returns:
        FastMCP server with all Tripletex tools configured
    """
    logger.info("Initializing Tripletex MCP Server...")

    # Step 1: Fetch OpenAPI spec
    logger.info(f"Fetching OpenAPI spec from {TRIPLETEX_OPENAPI_URL}")
    try:
        response = httpx.get(TRIPLETEX_OPENAPI_URL, timeout=30.0)
        response.raise_for_status()
        openapi_spec = response.json()
        logger.info(f"✓ OpenAPI spec loaded successfully")
    except httpx.RequestError as e:
        logger.error(f"Failed to fetch OpenAPI spec: {e}")
        raise

    # Step 2: Create HTTP client
    # Note: Authentication is handled dynamically by the coordinator
    # Each request will include the session token in the Authorization header
    client = httpx.AsyncClient(
        base_url=TRIPLETEX_BASE_URL,
        timeout=30.0,
    )

    logger.info(f"Base URL: {TRIPLETEX_BASE_URL}")

    # Step 3: Create MCP from OpenAPI spec
    # This automatically converts ~800 API endpoints into MCP tools
    logger.info("Generating MCP tools from OpenAPI spec...")
    mcp = FastMCP.from_openapi(
        openapi_spec=openapi_spec,
        client=client,
        name="tripletex-mcp",
        tags={"tripletex", "accounting", "api-v2"},
    )

    endpoint_count = len(openapi_spec.get("paths", {}))
    logger.info(f"✓ MCP server created with {endpoint_count} endpoints")

    return mcp


# ============================================================================
# Create and Configure Server
# ============================================================================

try:
    mcp = create_mcp_server()
except Exception as e:
    logger.error(f"Failed to initialize MCP server: {e}")
    # Create a minimal server that reports the error
    mcp = FastMCP(name="tripletex-mcp", version="0.1.0")

    @mcp.tool()
    def initialization_error() -> str:
        """
        MCP server initialization failed.

        Check the logs above for details.
        Common issues:
        - Network connectivity to Tripletex API
        - Invalid OpenAPI spec format
        - Timeout fetching spec
        """
        return f"Initialization error: {str(e)}"


# ============================================================================
# Authentication Helper (Reference)
# ============================================================================


def encode_session_token(session_token: str) -> str:
    """
    Encode session token in Tripletex Basic Auth format.

    Tripletex uses: Authorization: Basic base64(0:SESSION_TOKEN)
    Username is always "0", password is the session token.

    Args:
        session_token: The session token from Tripletex

    Returns:
        Base64-encoded Basic Auth credentials
    """
    credentials = f"0:{session_token}"
    encoded = base64.b64encode(credentials.encode()).decode()
    return f"Basic {encoded}"


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    logger.info("Starting Tripletex MCP Server on http://0.0.0.0:8083")
    logger.info("Press Ctrl+C to stop")

    # Run MCP server on port 8083
    mcp.run(transport="http", host="0.0.0.0", port=8083)
