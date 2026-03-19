# ai-accountant-mcp MCP Server

A Model Context Protocol (MCP) server for Tripletex API integration.

## Features example tasks

- � Create employees
- 📄 Create customers
- � Create products
- � Create invoices
- � Manage orders
- 🧾 Manage vouchers

## Prerequisites

1. Tripletex sandbox account with API access
2. Tripletex App with API access enabled
3. API credentials (API Key, API Secret)

### Setting up Tripletex Sandbox Account

1. [Tripletex API](https://kkpqfuj-amager.tripletex.dev/v2-docs/)
2. go to [Sandbox Account](../docs/Sandbox%20Account.md)


## Installation
```bash
cd ai-accountant-mcp
uv sync
```

## Configuration
Create a `.env` file in the project root:

```env
API_KEY=your_api_key_here
API_SECRET_KEY=your_api_secret_key_here
ACCESS_TOKEN=your_access_token_here
ACCESS_TOKEN_SECRET=your_access_token_secret_here
```

## Running the Server

```bash
# With uv
uv run python src/twitter_mcp/server.py

# Or with python directly
python src/twitter_mcp/server.py
```

The server will start on `http://0.0.0.0:8083/mcp`
