# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from database import get_page
from agents.orchestrator import run_orchestrator
from boilerplate import INITIAL_BOILERPLATE
from config import DEFAULT_MODEL, ALL_MODELS

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
    model_id: str = DEFAULT_MODEL


@app.get("/")
def health():
    return {"status": "ok", "service": "hyphertext-agent"}


@app.get("/models")
def list_models():
    return {"models": ALL_MODELS}


@app.post("/agent/run")
async def agent_run(req: AgentRunRequest):
    try:
        page = get_page(req.page_id)
        if not page:
            raise HTTPException(status_code=404, detail="Page not found")

        model_id = req.model_id if req.model_id in ALL_MODELS else DEFAULT_MODEL

        # owner_id is needed by the asset pipeline to upload extracted images
        owner_id = page.get("owner_id")

        asyncio.create_task(
            run_orchestrator(
                page_id=req.page_id,
                message_id=req.message_id,
                user_prompt=req.content,
                model_id=model_id,
                owner_id=owner_id,
            )
        )

        return {"status": "accepted", "model": model_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))