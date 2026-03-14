# main.py
import asyncio
import logging
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from database import get_page
from agents.orchestrator import run_orchestrator
from boilerplate import INITIAL_BOILERPLATE

# ---------------------------------------------------------------------------
# Logging — structured, visible in Railway
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
# Quiet noisy libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("supabase").setLevel(logging.WARNING)
logging.getLogger("postgrest").setLevel(logging.WARNING)
logging.getLogger("realtime").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

app = FastAPI(title="Hyphertext Agent Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AgentRunRequest(BaseModel):
    message_id: str
    page_id: str
    content: str
    model_id: Optional[str] = None
    inference_mode: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Hyphertext Agent Backend starting up")


@app.get("/")
def health():
    return {"status": "ok", "service": "hyphertext-agent"}


@app.post("/agent/run")
async def agent_run(req: AgentRunRequest):
    logger.info(
        "[API] /agent/run received — page=%s message=%s content_len=%d",
        req.page_id, req.message_id, len(req.content),
    )
    try:
        page = await get_page(req.page_id)
        if not page:
            logger.warning("[API] Page not found: %s", req.page_id)
            raise HTTPException(status_code=404, detail="Page not found")

        owner_id = page.get("owner_id")

        inference_mode = req.inference_mode
        if inference_mode not in ("economy", "speed"):
            inference_mode = "economy"

        logger.info(
            "[API] Dispatching agent task — page=%s owner=%s",
            req.page_id, owner_id,
        )

        asyncio.create_task(
            run_orchestrator(
                page_id=req.page_id,
                message_id=req.message_id,
                user_prompt=req.content,
                owner_id=owner_id,
                requested_inference_mode=inference_mode,
            )
        )

        return {"status": "accepted"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("[API] /agent/run unhandled error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))