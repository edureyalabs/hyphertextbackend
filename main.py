# main.py
import asyncio
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from database import get_page
from agents.orchestrator import run_orchestrator
from boilerplate import INITIAL_BOILERPLATE

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
    # Accepted for backward compatibility — model selection is automatic.
    model_id: Optional[str] = None
    inference_mode: Optional[str] = None


@app.get("/")
def health():
    return {"status": "ok", "service": "hyphertext-agent"}


@app.post("/agent/run")
async def agent_run(req: AgentRunRequest):
    try:
        page = await get_page(req.page_id)
        if not page:
            raise HTTPException(status_code=404, detail="Page not found")

        owner_id = page.get("owner_id")

        inference_mode = req.inference_mode
        if inference_mode not in ("economy", "speed"):
            inference_mode = "economy"

        # Fire-and-forget: the task runs inside run_orchestrator's own
        # semaphore + timeout guards. Any exception is caught there and
        # written to the DB — it never propagates back here.
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
        logger.error("[main] /agent/run error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))