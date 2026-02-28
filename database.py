from supabase import create_client, Client
from config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def get_page(page_id: str) -> dict:
    res = supabase.table("pages").select("*").eq("id", page_id).single().execute()
    return res.data


def update_page_html(page_id: str, html: str):
    supabase.table("pages").update({
        "html_content": html,
        "updated_at": "now()"
    }).eq("id", page_id).execute()


def update_page_summary_and_map(page_id: str, html_summary: str, component_map: list):
    supabase.table("pages").update({
        "html_summary": html_summary,
        "component_map": component_map,
        "updated_at": "now()"
    }).eq("id", page_id).execute()


def get_chat_history(page_id: str, limit: int = 10) -> list:
    res = (
        supabase.table("chat_messages")
        .select("role, content, message_type, meta")
        .eq("page_id", page_id)
        .eq("status", "completed")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return list(reversed(res.data))


def update_message_status(message_id: str, status: str):
    supabase.table("chat_messages").update({
        "status": status
    }).eq("id", message_id).execute()


def insert_assistant_message(page_id: str, content: str, message_type: str = "chat", meta: dict = None):
    supabase.table("chat_messages").insert({
        "page_id": page_id,
        "role": "assistant",
        "content": content,
        "status": "completed",
        "message_type": message_type,
        "meta": meta or {}
    }).execute()


def insert_thinking_message(page_id: str, plan: dict) -> str:
    res = supabase.table("chat_messages").insert({
        "page_id": page_id,
        "role": "assistant",
        "content": "thinking",
        "status": "completed",
        "message_type": "thinking",
        "meta": {"plan": plan}
    }).execute()
    return res.data[0]["id"] if res.data else None


def snapshot_version(page_id: str, html: str):
    res = (
        supabase.table("page_versions")
        .select("version_num")
        .eq("page_id", page_id)
        .order("version_num", desc=True)
        .limit(1)
        .execute()
    )
    next_version = (res.data[0]["version_num"] + 1) if res.data else 1

    supabase.table("page_versions").insert({
        "page_id": page_id,
        "html_snapshot": html,
        "version_num": next_version,
        "trigger_type": "agent_complete"
    }).execute()


def get_edit_history(page_id: str, limit: int = 5) -> list:
    res = (
        supabase.table("edit_history")
        .select("*")
        .eq("page_id", page_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return list(reversed(res.data))


def insert_edit_history(
    page_id: str,
    message_id: str,
    complexity: str,
    decision: str,
    plan_json: dict,
    changes_json: list,
    clarification_asked: bool,
    web_searches_used: list,
    model_used: str,
    tokens_used: int,
    success: bool
):
    supabase.table("edit_history").insert({
        "page_id": page_id,
        "message_id": message_id,
        "complexity": complexity,
        "decision": decision,
        "plan_json": plan_json,
        "changes_json": changes_json,
        "clarification_asked": clarification_asked,
        "web_searches_used": web_searches_used,
        "model_used": model_used,
        "tokens_used": tokens_used,
        "success": success
    }).execute()


def insert_clarification(page_id: str, message_id: str, question: str) -> str:
    res = supabase.table("clarification_threads").insert({
        "page_id": page_id,
        "message_id": message_id,
        "question": question,
        "resolved": False
    }).execute()
    return res.data[0]["id"] if res.data else None


def resolve_clarification(clarification_id: str, answer: str):
    supabase.table("clarification_threads").update({
        "answer": answer,
        "resolved": True,
        "resolved_at": "now()"
    }).eq("id", clarification_id).execute()


def get_pending_clarification(page_id: str) -> dict:
    res = (
        supabase.table("clarification_threads")
        .select("*")
        .eq("page_id", page_id)
        .eq("resolved", False)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    return res.data[0] if res.data else None