# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import asyncio
from database import get_page
from agents.orchestrator import run_orchestrator
from boilerplate import INITIAL_BOILERPLATE

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
    # model_id is accepted for backward compatibility but intentionally ignored —
    # model selection is now fully automatic via coding_router.
    model_id: Optional[str] = None
    # inference_mode is only honoured on the very first message for a page.
    # "economy" → Together AI (default)
    # "speed"   → Cerebras (~1000 TPS)
    # On subsequent messages the persisted pages.inference_mode takes precedence.
    inference_mode: Optional[str] = None


@app.get("/")
def health():
    return {"status": "ok", "service": "hyphertext-agent"}


@app.post("/agent/run")
async def agent_run(req: AgentRunRequest):
    try:
        page = get_page(req.page_id)
        if not page:
            raise HTTPException(status_code=404, detail="Page not found")

        owner_id = page.get("owner_id")

        # Validate inference_mode value — default to "economy" for unknown values
        inference_mode = req.inference_mode
        if inference_mode not in ("economy", "speed"):
            inference_mode = "economy"

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

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))