# database.py
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


def get_chat_history(page_id: str, limit: int = 10) -> list:
    res = (
        supabase.table("chat_messages")
        .select("role, content")
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


def insert_assistant_message(page_id: str, content: str):
    supabase.table("chat_messages").insert({
        "page_id": page_id,
        "role": "assistant",
        "content": content,
        "status": "completed"
    }).execute()


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