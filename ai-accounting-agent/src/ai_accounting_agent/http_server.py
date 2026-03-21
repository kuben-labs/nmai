"""FastAPI HTTP server for Tripletex accounting agent."""

import json
import os
import asyncio
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger
from dotenv import load_dotenv

from .coordinator import run_accounting_task
from .file_processor import FileProcessor
from .tripletex_client import set_tripletex_client, cleanup_tripletex_client

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="AI Accounting Agent",
    description="Multi-agent accounting task solver for Tripletex",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Pydantic Models for Request/Response
# ============================================================================


class FileInput(BaseModel):
    """Attached file in the request."""

    filename: str
    content_base64: str
    mime_type: str


class TripletexCredentials(BaseModel):
    """Tripletex API credentials."""

    base_url: str
    session_token: str


class SolveRequest(BaseModel):
    """Request to solve an accounting task."""

    prompt: str
    files: Optional[List[FileInput]] = None
    tripletex_credentials: TripletexCredentials


class SolveResponse(BaseModel):
    """Response from solving an accounting task."""

    status: str
    message: Optional[str] = None


# ============================================================================
# API Endpoints
# ============================================================================


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "ai-accounting-agent", "version": "0.1.0"}


@app.post("/solve")
async def solve(request: Request):
    """
    Main endpoint for solving accounting tasks.

    Receives a task prompt, optional files, and Tripletex credentials.
    Orchestrates the multi-agent workflow and returns {"status": "completed"}.

    Request format (from Tripletex):
    {
        "prompt": "Task description in any language",
        "files": [
            {
                "filename": "invoice.pdf",
                "content_base64": "JVBERi0x...",
                "mime_type": "application/pdf"
            }
        ],
        "tripletex_credentials": {
            "base_url": "https://tx-proxy.ainm.no/v2",
            "session_token": "token123..."
        }
    }

    Returns:
    {"status": "completed"}
    """
    request_id = request.headers.get("x-request-id", "unknown")
    logger.info(f"\n{'=' * 80}")
    logger.info(f"[{request_id}] 📤 RECEIVED REQUEST FROM COMPETITION")
    logger.info(f"{'=' * 80}")

    try:
        # Parse request
        body = await request.json()
        logger.debug(f"[{request_id}] Request body parsed successfully")

        # Validate required fields
        prompt = body.get("prompt")
        if not prompt:
            logger.error(f"[{request_id}] ❌ Missing required field: prompt")
            logger.info(f"[{request_id}] Returning HTTP 200 (spec compliance)")
            return JSONResponse(
                status_code=200,
                content={"status": "completed"},
            )

        # Extract optional fields
        files = body.get("files", [])
        tripletex_credentials = body.get("tripletex_credentials", {})

        # Log incoming request details
        logger.info(f"[{request_id}] ✓ Prompt received: {prompt[:80]}...")
        logger.info(f"[{request_id}] ✓ Files attached: {len(files)}")
        if files:
            for i, f in enumerate(files, 1):
                logger.info(
                    f"[{request_id}]   - File {i}: {f.get('filename')} ({f.get('mime_type')})"
                )

        base_url = tripletex_credentials.get("base_url", "")
        session_token = tripletex_credentials.get("session_token", "")
        logger.info(f"[{request_id}] ✓ Credentials received:")
        logger.info(f"[{request_id}]   - base_url: {base_url}")
        logger.info(
            f"[{request_id}]   - session_token: {session_token[:20]}..."
            if session_token
            else f"[{request_id}]   - session_token: [MISSING]"
        )

        # Process attached files if any
        file_data = None
        extracted_file_content = ""
        if files:
            logger.info(
                f"[{request_id}] 📦 Processing {len(files)} attached file(s)..."
            )
            file_data = FileProcessor.process_files(files)
            logger.debug(
                f"[{request_id}] File processing result: {file_data.get('success')}"
            )

            # Extract text content from processed files for the agent
            if file_data.get("success") and file_data.get("processed_files"):
                file_contents = []
                for pf in file_data["processed_files"]:
                    filename = pf.get("filename", "unknown")
                    content = pf.get("extracted_text") or pf.get("content", "")
                    if content:
                        file_contents.append(
                            f"--- FILE: {filename} ---\n{content}\n--- END FILE ---"
                        )
                if file_contents:
                    extracted_file_content = "\n\n".join(file_contents)
                    logger.info(
                        f"[{request_id}] ✓ Extracted {len(file_contents)} file(s) content ({len(extracted_file_content)} chars)"
                    )
            else:
                logger.warning(
                    f"[{request_id}] ⚠ File processing failed or returned no content"
                )

        # Set up Tripletex client with per-request credentials
        if base_url and session_token:
            # Set dynamic credentials for this request
            set_tripletex_client(base_url, session_token)
            # Also set environment variables for MCP server if it checks them
            os.environ["TRIPLETEX_BASE_URL"] = base_url
            os.environ["TRIPLETEX_SESSION_TOKEN"] = session_token
            logger.info(
                f"[{request_id}] ✓ Configured Tripletex client with provided credentials"
            )
        else:
            logger.warning(
                f"[{request_id}] ⚠ Missing Tripletex credentials in request!"
            )

        # Run the accounting task workflow with 280s timeout (20s buffer before 300s hard limit)
        logger.info(f"[{request_id}] ⏳ Starting accounting task execution...")
        logger.info(f"[{request_id}] Timeout: 280s (300s spec limit - 20s buffer)")
        task_start = asyncio.get_event_loop().time()

        try:
            result = await asyncio.wait_for(
                run_accounting_task(
                    prompt=prompt,
                    file_content=extracted_file_content,
                    tripletex_credentials=tripletex_credentials,
                ),
                timeout=280.0,  # 280s timeout, 20s buffer before 300s hard limit
            )
        except asyncio.TimeoutError:
            task_elapsed = asyncio.get_event_loop().time() - task_start
            logger.error(
                f"[{request_id}] ⏱ TIMEOUT after {task_elapsed:.2f}s - task exceeded 280s limit"
            )
            await cleanup_tripletex_client()
            logger.info(
                f"[{request_id}] ✓ Returning HTTP 200 with spec-compliant response"
            )
            return JSONResponse(
                status_code=200,
                content={"status": "completed"},  # Return completed to avoid penalty
            )

        task_elapsed = asyncio.get_event_loop().time() - task_start
        logger.info(f"[{request_id}] ✓ Task execution completed in {task_elapsed:.2f}s")

        # Clean up client
        await cleanup_tripletex_client()

        # Always return completed per spec
        logger.info(f"[{request_id}] 📥 RESPONSE TO COMPETITION:")
        logger.info(f"[{request_id}] Status Code: HTTP 200")
        logger.info(f"[{request_id}] Body: {json.dumps({'status': 'completed'})}")
        logger.info(f"[{request_id}] ✓ SPEC COMPLIANT")
        logger.info(f"{'=' * 80}\n")
        return JSONResponse(status_code=200, content={"status": "completed"})

    except ValueError as e:
        logger.error(f"[{request_id}] ❌ Validation error: {e}")
        logger.info(f"[{request_id}] 📥 RESPONSE TO COMPETITION (Error case):")
        logger.info(f"[{request_id}] Status Code: HTTP 200")
        logger.info(f"[{request_id}] Body: {json.dumps({'status': 'completed'})}")
        logger.info(f"[{request_id}] ✓ SPEC COMPLIANT (errors return 200)")
        logger.info(f"{'=' * 80}\n")
        return JSONResponse(
            status_code=200,
            content={"status": "completed"},
        )

    except Exception as e:
        logger.error(f"[{request_id}] ❌ Unexpected error: {e}", exc_info=True)
        logger.info(f"[{request_id}] 📥 RESPONSE TO COMPETITION (Exception case):")
        logger.info(f"[{request_id}] Status Code: HTTP 200")
        logger.info(f"[{request_id}] Body: {json.dumps({'status': 'completed'})}")
        logger.info(f"[{request_id}] ✓ SPEC COMPLIANT (exceptions return 200)")
        logger.info(f"{'=' * 80}\n")
        return JSONResponse(
            status_code=200,
            content={"status": "completed"},
        )


# ============================================================================
# Startup/Shutdown Events
# ============================================================================


@app.on_event("startup")
def startup_event():
    """Run on application startup."""
    logger.info("AI Accounting Agent starting up...")
    logger.info(f"LLM Model: {os.getenv('LLM_MODEL')}")
    logger.info(f"MCP Config: mcp_accountant.json")
    logger.info("Ready to accept /solve requests")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    logger.info("AI Accounting Agent shutting down...")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8080))

    logger.info(f"Starting server on {host}:{port}")

    uvicorn.run(app, host=host, port=port, log_level="info")
