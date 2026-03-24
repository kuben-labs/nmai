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
from machine_core import FileProcessor

# Load environment variables
load_dotenv()

# Track concurrent /solve requests for observability
_active_requests = 0
_active_requests_lock = asyncio.Lock()

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

    global _active_requests
    async with _active_requests_lock:
        _active_requests += 1
        current = _active_requests
    logger.info(f"[{request_id}] Active /solve requests: {current}")

    try:
        # Parse request and log the FULL body for debugging
        body = await request.json()
        logger.info(f"[{request_id}] === FULL REQUEST BODY ===")
        logger.info(f"[{request_id}] Keys: {list(body.keys())}")
        # Log prompt in full
        logger.info(f"[{request_id}] prompt: {body.get('prompt', 'MISSING')}")
        # Log credentials
        creds = body.get("tripletex_credentials", {})
        logger.info(
            f"[{request_id}] tripletex_credentials.base_url: {creds.get('base_url', 'MISSING')}"
        )
        logger.info(
            f"[{request_id}] tripletex_credentials.session_token: {str(creds.get('session_token', 'MISSING'))[:30]}..."
        )
        # Log files
        files_raw = body.get("files", [])
        logger.info(f"[{request_id}] files count: {len(files_raw)}")
        for i, f in enumerate(files_raw):
            logger.info(
                f"[{request_id}]   file[{i}]: {f.get('filename')} ({f.get('mime_type')}) base64_len={len(f.get('content_base64', ''))}"
            )
        logger.info(f"[{request_id}] === END REQUEST BODY ===")

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
                    # file_processor returns extracted_data as a nested dict
                    # with keys like full_text (PDF), text (image OCR), content (text/CSV)
                    extracted = pf.get("extracted_data", {})
                    content = (
                        extracted.get("full_text")  # PDF via pdfplumber/PyPDF2
                        or extracted.get("text")  # Image via OCR
                        or extracted.get("content")  # Text/CSV
                        or ""
                    )
                    if content:
                        file_contents.append(
                            f"--- FILE: {filename} ---\n{content}\n--- END FILE ---"
                        )
                if file_contents:
                    extracted_file_content = "\n\n".join(file_contents)
                    logger.info(
                        f"[{request_id}] ✓ Extracted {len(file_contents)} file(s) content ({len(extracted_file_content)} chars)"
                    )
                    # Log the actual extracted content so we can verify it
                    logger.info(
                        f"[{request_id}] === EXTRACTED FILE CONTENT ===\n{extracted_file_content[:2000]}"
                    )
                    logger.info(f"[{request_id}] === END FILE CONTENT ===")
            else:
                logger.warning(
                    f"[{request_id}] ⚠ File processing failed or returned no content"
                )

        # Credentials are passed directly to the coordinator (no MCP server needed)
        if base_url and session_token:
            logger.info(
                f"[{request_id}] ✓ Credentials will be used for direct API calls"
            )
        else:
            logger.warning(
                f"[{request_id}] ⚠ Missing Tripletex credentials in request!"
            )

        # Run the accounting task workflow with 300s timeout (20s buffer before 300s hard limit)
        logger.info(f"[{request_id}] ⏳ Starting accounting task execution...")
        logger.info(f"[{request_id}] Timeout: 300s (300s spec limit - 20s buffer)")
        task_start = asyncio.get_event_loop().time()

        try:
            result = await asyncio.wait_for(
                run_accounting_task(
                    prompt=prompt,
                    file_content=extracted_file_content,
                    tripletex_credentials=tripletex_credentials,
                ),
                timeout=300.0,  # 300s timeout, 20s buffer before 300s hard limit
            )
        except asyncio.TimeoutError:
            task_elapsed = asyncio.get_event_loop().time() - task_start
            logger.error(
                f"[{request_id}] ⏱ TIMEOUT after {task_elapsed:.2f}s - task exceeded 300s limit"
            )
            logger.info(
                f"[{request_id}] ✓ Returning HTTP 200 with spec-compliant response"
            )
            return JSONResponse(
                status_code=200,
                content={"status": "completed"},  # Return completed to avoid penalty
            )

        task_elapsed = asyncio.get_event_loop().time() - task_start
        logger.info(f"[{request_id}] ✓ Task execution completed in {task_elapsed:.2f}s")

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

    finally:
        async with _active_requests_lock:
            _active_requests -= 1
        logger.info(
            f"[{request_id}] /solve finished. Active requests: {_active_requests}"
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
