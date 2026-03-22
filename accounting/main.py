"""FastAPI server for the Tripletex AI Accounting Agent."""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

AGENT_API_KEY = os.getenv("AGENT_API_KEY", "")


@asynccontextmanager
async def lifespan(app: FastAPI):
    project = os.getenv("GCP_PROJECT_ID", "ai-nm26osl-1759")
    model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
    logger.info(f"Accounting agent ready (project={project}, model={model})")
    yield


app = FastAPI(title="Tripletex AI Accounting Agent", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/")
@app.post("/solve")
async def solve(request: Request):
    if AGENT_API_KEY:
        auth_header = request.headers.get("authorization", "")
        if auth_header != f"Bearer {AGENT_API_KEY}":
            raise HTTPException(status_code=401, detail="Invalid API key")

    body = await request.json()
    prompt = body.get("prompt", "")
    files = body.get("files", [])
    creds = body.get("tripletex_credentials", {})

    base_url = creds.get("base_url", "")
    session_token = creds.get("session_token", "")

    if not prompt or not base_url or not session_token:
        raise HTTPException(status_code=400, detail="Missing required fields")

    logger.info(f"Received task: {prompt[:200]}...")
    logger.info(f"Files: {len(files)}, Base URL: {base_url}")

    # Structured log for task tracking — full prompt so we can correlate with scoring results
    import json as _json
    task_input_log = _json.dumps({
        "tag": "TASK_INPUT",
        "prompt": prompt[:3000],
        "file_count": len(files),
        "file_names": [f.get("filename", "") for f in files],
    }, ensure_ascii=False)
    logger.info(task_input_log)

    import asyncio
    from agent import run_agent

    result = await asyncio.to_thread(
        run_agent, prompt, files, base_url, session_token
    )

    return JSONResponse(result)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
