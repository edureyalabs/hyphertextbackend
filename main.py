# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from db import get_page
from agents import run_create_agent, run_edit_agent
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
    content: str  # the user's prompt


@app.get("/")
def health():
    return {"status": "ok", "service": "hyphertext-agent"}


@app.post("/agent/run")
async def agent_run(req: AgentRunRequest):
    """
    Entry point called by the Supabase Edge Function.
    Decides which agent to use based on whether this is the first message.
    """
    try:
        page = get_page(req.page_id)
        if not page:
            raise HTTPException(status_code=404, detail="Page not found")

        current_html = page.get("html_content", "")
        is_first_prompt = (
            current_html == "" or
            current_html == INITIAL_BOILERPLATE or
            "describe what you want to build" in current_html
        )

        if is_first_prompt:
            # Run create agent — writes full file
            asyncio.create_task(
                run_create_agent(req.page_id, req.message_id, req.content)
            )
        else:
            # Run edit agent — surgical str_replace
            asyncio.create_task(
                run_edit_agent(req.page_id, req.message_id, req.content)
            )

        return {"status": "accepted", "agent": "create" if is_first_prompt else "edit"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))