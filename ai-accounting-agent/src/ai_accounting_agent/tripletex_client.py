"""Direct Tripletex API client with per-request credentials.

This client allows the agent to make Tripletex API calls using credentials
provided in each request, rather than relying on static .env credentials.

The MCP server is still used for tool definitions, but actual API calls
go through this client to ensure proper authentication.
"""

import base64
import httpx
from typing import Dict, Any, Optional, List
from loguru import logger


class TripletexClient:
    """
    Async HTTP client for Tripletex API with dynamic credentials.

    Usage:
        client = TripletexClient(base_url, session_token)
        result = await client.get("/employee", params={"fields": "id,firstName"})
        result = await client.post("/customer", json={"name": "Test"})
    """

    def __init__(self, base_url: str, session_token: str, timeout: float = 30.0):
        """
        Initialize the Tripletex client.

        Args:
            base_url: Tripletex API base URL (e.g., https://tx-proxy.ainm.no/v2)
            session_token: Session token for authentication
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.session_token = session_token
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    def _encode_auth(self) -> str:
        """Encode session token in Tripletex Basic Auth format."""
        credentials = f"0:{self.session_token}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return f"Basic {encoded}"

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": self._encode_auth(),
                    "Content-Type": "application/json",
                },
                timeout=self.timeout,
            )
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def get(
        self, path: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make a GET request to the Tripletex API.

        Args:
            path: API endpoint path (e.g., "/employee")
            params: Query parameters

        Returns:
            JSON response as dictionary
        """
        client = await self._get_client()
        try:
            response = await client.get(path, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(
                f"GET {path} failed: {e.response.status_code} - {e.response.text}"
            )
            return {
                "error": True,
                "status_code": e.response.status_code,
                "message": e.response.text,
            }
        except Exception as e:
            logger.error(f"GET {path} error: {e}")
            return {"error": True, "message": str(e)}

    async def post(
        self,
        path: str,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make a POST request to the Tripletex API.

        Args:
            path: API endpoint path (e.g., "/customer")
            json: JSON body
            params: Query parameters

        Returns:
            JSON response as dictionary
        """
        client = await self._get_client()
        try:
            response = await client.post(path, json=json, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(
                f"POST {path} failed: {e.response.status_code} - {e.response.text}"
            )
            return {
                "error": True,
                "status_code": e.response.status_code,
                "message": e.response.text,
            }
        except Exception as e:
            logger.error(f"POST {path} error: {e}")
            return {"error": True, "message": str(e)}

    async def put(
        self,
        path: str,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make a PUT request to the Tripletex API.

        Args:
            path: API endpoint path (e.g., "/employee/123")
            json: JSON body
            params: Query parameters

        Returns:
            JSON response as dictionary
        """
        client = await self._get_client()
        try:
            response = await client.put(path, json=json, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(
                f"PUT {path} failed: {e.response.status_code} - {e.response.text}"
            )
            return {
                "error": True,
                "status_code": e.response.status_code,
                "message": e.response.text,
            }
        except Exception as e:
            logger.error(f"PUT {path} error: {e}")
            return {"error": True, "message": str(e)}

    async def delete(
        self, path: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make a DELETE request to the Tripletex API.

        Args:
            path: API endpoint path (e.g., "/travelExpense/123")
            params: Query parameters

        Returns:
            JSON response as dictionary (or empty dict on success)
        """
        client = await self._get_client()
        try:
            response = await client.delete(path, params=params)
            response.raise_for_status()
            # DELETE often returns empty or 204
            if response.status_code == 204 or not response.content:
                return {"success": True}
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(
                f"DELETE {path} failed: {e.response.status_code} - {e.response.text}"
            )
            return {
                "error": True,
                "status_code": e.response.status_code,
                "message": e.response.text,
            }
        except Exception as e:
            logger.error(f"DELETE {path} error: {e}")
            return {"error": True, "message": str(e)}
