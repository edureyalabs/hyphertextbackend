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
    # model_id is accepted for backward compatibility with older frontend versions
    # but is intentionally ignored — model selection is now fully automatic.
    model_id: Optional[str] = None


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

        asyncio.create_task(
            run_orchestrator(
                page_id=req.page_id,
                message_id=req.message_id,
                user_prompt=req.content,
                owner_id=owner_id,
            )
        )

        return {"status": "accepted"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))