"""FastAPI HTTP server for Tripletex accounting agent."""

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
    logger.info(f"[{request_id}] Received /solve request")

    try:
        # Parse request
        body = await request.json()
        logger.debug(f"[{request_id}] Request body parsed")

        # Validate required fields
        prompt = body.get("prompt")
        if not prompt:
            logger.error(f"[{request_id}] Missing required field: prompt")
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "Missing required field: prompt",
                },
            )

        # Extract optional fields
        files = body.get("files", [])
        tripletex_credentials = body.get("tripletex_credentials", {})

        logger.info(f"[{request_id}] Task prompt: {prompt[:100]}...")
        logger.info(f"[{request_id}] Files: {len(files)}")

        # Process attached files if any
        file_data = None
        if files:
            logger.info(f"[{request_id}] Processing {len(files)} attached files...")
            file_data = FileProcessor.process_files(files)
            logger.debug(
                f"[{request_id}] File processing result: {file_data.get('success')}"
            )

        # Run the accounting task workflow
        logger.info(f"[{request_id}] Starting accounting task workflow...")
        task_start = asyncio.get_event_loop().time()

        result = await run_accounting_task(
            prompt=prompt, files=files, tripletex_credentials=tripletex_credentials
        )

        task_elapsed = asyncio.get_event_loop().time() - task_start
        logger.info(f"[{request_id}] Task workflow completed in {task_elapsed:.2f}s")

        # Check if successful
        if result.get("success"):
            logger.info(f"[{request_id}] Task completed successfully")
            return JSONResponse(status_code=200, content={"status": "completed"})
        else:
            logger.error(f"[{request_id}] Task failed: {result.get('error')}")
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": result.get("error", "Task execution failed"),
                },
            )

    except ValueError as e:
        logger.error(f"[{request_id}] Validation error: {e}")
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": f"Invalid request: {str(e)}"},
        )

    except asyncio.TimeoutError:
        logger.error(f"[{request_id}] Request timeout (>5 minutes)")
        return JSONResponse(
            status_code=504,
            content={
                "status": "error",
                "message": "Request timeout - task took too long",
            },
        )

    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Internal server error: {str(e)}"},
        )


@app.post("/debug/plan")
async def debug_plan(request: Request):
    """
    Debug endpoint to test just the planning phase.

    Useful for debugging task analysis without full execution.

    Request body:
    {
        "prompt": "Task description"
    }
    """
    try:
        # Get request body safely
        body = await request.json()
        if not body:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Empty request body"},
            )

        prompt = body.get("prompt")
        if not prompt:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "Missing 'prompt' field in request body",
                },
            )

        logger.info(f"Debug plan request: {prompt[:100]}...")

        from .planner import AccountingPlanner

        planner = AccountingPlanner()
        plan = await planner.plan_task(prompt)

        return JSONResponse(
            status_code=200, content={"status": "success", "plan": plan}
        )

    except ValueError as e:
        logger.error(f"Debug plan JSON error: {e}")
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": f"Invalid JSON: {str(e)}. Expected: {{'prompt': 'task description'}}",
            },
        )
    except Exception as e:
        logger.error(f"Debug plan error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500, content={"status": "error", "message": str(e)}
        )


@app.post("/debug/split")
async def debug_split(request: Request):
    """
    Debug endpoint to test the task splitting phase.

    Request body:
    {
        "planning_context": "The plan output from /debug/plan"
    }
    """
    try:
        # Get request body safely
        body = await request.json()
        if not body:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Empty request body"},
            )

        planning_context = body.get("planning_context")
        if not planning_context:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "Missing 'planning_context' field in request body",
                },
            )

        logger.info(f"Debug split request: {planning_context[:100]}...")

        from .task_splitter import TaskSplitter

        splitter = TaskSplitter()
        subtasks = await splitter.split_into_subtasks(planning_context)

        return JSONResponse(
            status_code=200, content={"status": "success", "subtasks": subtasks}
        )

    except ValueError as e:
        logger.error(f"Debug split JSON error: {e}")
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": f"Invalid JSON: {str(e)}. Expected: {{'planning_context': 'plan text'}}",
            },
        )
    except Exception as e:
        logger.error(f"Debug split error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500, content={"status": "error", "message": str(e)}
        )


# ============================================================================
# Startup/Shutdown Events
# ============================================================================


@app.on_event("startup")
async def startup_event():
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
    port = int(os.getenv("PORT", 8000))

    logger.info(f"Starting server on {host}:{port}")

    uvicorn.run(app, host=host, port=port, log_level="info")
