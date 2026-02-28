# database.py
from supabase import create_client, Client
from config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
from typing import Optional

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def get_supabase_client() -> Client:
    """Expose the shared client for use in processors."""
    return supabase


# ─── Pages ───────────────────────────────────────────────────────────────────

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


# ─── Chat ────────────────────────────────────────────────────────────────────

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


# ─── Assets ──────────────────────────────────────────────────────────────────

def get_pending_assets_for_page(page_id: str) -> list:
    """Return all assets in 'pending' status for a page (top-level only)."""
    res = (
        supabase.table("page_assets")
        .select("*")
        .eq("page_id", page_id)
        .eq("processing_status", "pending")
        .is_("parent_asset_id", "null")
        .execute()
    )
    return res.data or []


def get_page_assets_ready(page_id: str) -> list:
    """Return all 'ready' assets for a page (including extracted children)."""
    res = (
        supabase.table("page_assets")
        .select("*")
        .eq("page_id", page_id)
        .eq("processing_status", "ready")
        .order("created_at", ascending=True)
        .execute()
    )
    return res.data or []


def update_asset_processing_started(asset_id: str):
    supabase.table("page_assets").update({
        "processing_status": "processing",
        "updated_at": "now()"
    }).eq("id", asset_id).execute()


def update_asset_image_result(
    asset_id: str,
    vision_description: str,
    vision_tags: list,
    vision_suggested_use: str,
    vision_alt_text: str,
    vision_contains_text: bool,
    vision_extracted_text: str,
    dominant_colors: list,
):
    supabase.table("page_assets").update({
        "processing_status": "ready",
        "vision_description": vision_description,
        "vision_tags": vision_tags,
        "vision_suggested_use": vision_suggested_use,
        "vision_alt_text": vision_alt_text,
        "vision_contains_text": vision_contains_text,
        "vision_extracted_text": vision_extracted_text,
        "dominant_colors": dominant_colors,
        "updated_at": "now()"
    }).eq("id", asset_id).execute()


def update_asset_document_result(
    asset_id: str,
    extracted_text: str,
    extracted_summary: str,
):
    supabase.table("page_assets").update({
        "processing_status": "ready",
        "extracted_text": extracted_text,
        "extracted_summary": extracted_summary,
        "updated_at": "now()"
    }).eq("id", asset_id).execute()


def mark_asset_failed(asset_id: str, error: str):
    supabase.table("page_assets").update({
        "processing_status": "failed",
        "processing_error": error,
        "updated_at": "now()"
    }).eq("id", asset_id).execute()


def insert_extracted_image_asset(
    page_id: str,
    owner_id: str,
    parent_asset_id: str,
    file_name: str,
    original_file_name: str,
    file_type: str,
    storage_path: str,
    public_url: str,
    width: int,
    height: int,
    file_size_bytes: int,
) -> Optional[str]:
    """Insert a child image asset extracted from a document. Returns new asset id."""
    res = supabase.table("page_assets").insert({
        "page_id": page_id,
        "owner_id": owner_id,
        "parent_asset_id": parent_asset_id,
        "file_name": file_name,
        "original_file_name": original_file_name,
        "file_type": file_type,
        "asset_type": "extracted_image",
        "storage_path": storage_path,
        "public_url": public_url,
        "width": width,
        "height": height,
        "file_size_bytes": file_size_bytes,
        "processing_status": "processing",  # will be updated by vision call right after
    }).execute()
    return res.data[0]["id"] if res.data else None