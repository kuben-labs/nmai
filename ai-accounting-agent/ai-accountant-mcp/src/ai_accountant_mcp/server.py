"""AI Accountant MCP Server using Tripletex API v2

This server provides tools to interact with Tripletex API v2 including:
- Creating invoices
- Searching invoices
"""

import os
from fastmcp import FastMCP
from dotenv import load_dotenv
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP(
    name="ai-accountant-mcp",
    version="0.1.0"
)

class TripletexAPI:
    """Simple Tripletex API client for demonstration purposes"""
    def __init__(self, client_id, client_secret, access_token, access_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = access_token
        self.access_secret = access_secret

    def create_invoice(self, invoice_data):
        """Create a new invoice in Tripletex (placeholder implementation)"""
        # Here you would implement the actual API call to Tripletex
        # For demonstration, we return a mock response
        return {"id": "mock-invoice-id", "status": "created"}
    
# Initialize Tripletex API client with OAuth 2.0
def get_tripletex_client():
    """Initialize and return Tripletex API client with OAuth 2.0 credentials"""
    client_id = os.getenv("CLIENT_ID")
    client_secret = os.getenv("CLIENT_SECRET")
    access_token = os.getenv("ACCESS_TOKEN")
    access_secret = os.getenv("ACCESS_TOKEN_SECRET")
    # Debug logging (remove in production)
    print(f"DEBUG - CLIENT_ID present: {bool(client_id)}")
    print(f"DEBUG - CLIENT_SECRET present: {bool(client_secret)}")
    print(f"DEBUG - ACCESS_TOKEN present: {bool(access_token)}")
    print(f"DEBUG - ACCESS_TOKEN_SECRET present: {bool(access_secret)}")
    
    if not all([client_id, client_secret, access_token, access_secret]):
        raise ValueError(
            "Missing required environment variables. Please set: "
            "CLIENT_ID, CLIENT_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET"
        )
    
    return TripletexAPI(client_id, client_secret, access_token, access_secret)


@mcp.tool()
def post_invoice(invoice_data: dict) -> str:
    """Create a new invoice in Tripletex

    Args:
        invoice_data: A dictionary containing the invoice details

    Returns:
        Success message with invoice ID or error message
    """
    try:
        api = get_tripletex_client()
        
        # Validate invoice data
        if not invoice_data:
            return "Error: Invoice data is required"

        
        # Create invoice
        response = api.create_invoice(invoice_data)
        invoice_id = response.get('id')
        if invoice_id:
            return f"Invoice created successfully with ID: {invoice_id}"
        else:
            return "Failed to create invoice. No ID returned."

    except ValueError as e:
        return f"Configuration error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


if __name__ == "__main__":
    # Run as HTTP server for MCP
    mcp.run(transport="http", host="0.0.0.0", port=8083)
