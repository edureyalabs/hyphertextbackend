# database.py — fully async using supabase AsyncClient
# All public functions are async and await the Supabase client.
# The async client is initialised once per process via get_db().

import asyncio
import logging
from typing import Optional

from supabase import acreate_client, AsyncClient
from config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Single async client per process — initialised lazily on first use.
# ---------------------------------------------------------------------------

_client: Optional[AsyncClient] = None
_client_lock = asyncio.Lock()


async def get_db() -> AsyncClient:
    """Return (or create) the process-level async Supabase client."""
    global _client
    if _client is not None:
        return _client
    async with _client_lock:
        if _client is None:
            _client = await acreate_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    return _client


# Kept for the handful of legacy sync call-sites that haven't been converted yet.
# Do NOT use in new code.
def get_supabase_client():
    from supabase import create_client
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


# ── Pages ────────────────────────────────────────────────────────────────────

async def get_page(page_id: str) -> dict:
    db = await get_db()
    res = await db.table("pages").select("*").eq("id", page_id).single().execute()
    return res.data


async def update_page_html(page_id: str, html: str):
    db = await get_db()
    await db.table("pages").update({
        "html_content": html,
        "updated_at": "now()"
    }).eq("id", page_id).execute()


async def update_page_summary_and_map(page_id: str, html_summary: str, component_map: list):
    db = await get_db()
    await db.table("pages").update({
        "html_summary": html_summary,
        "component_map": component_map,
        "updated_at": "now()"
    }).eq("id", page_id).execute()


async def update_page_coding_model(page_id: str, coding_model_id: Optional[str]):
    try:
        db = await get_db()
        await db.table("pages").update({
            "coding_model_id": coding_model_id,
            "updated_at": "now()"
        }).eq("id", page_id).execute()
    except Exception as e:
        logger.warning("[DB] update_page_coding_model failed: %s", e)


async def update_page_inference_mode(page_id: str, mode: str):
    try:
        db = await get_db()
        await db.table("pages").update({
            "inference_mode": mode,
            "updated_at": "now()"
        }).eq("id", page_id).execute()
    except Exception as e:
        logger.warning("[DB] update_page_inference_mode failed: %s", e)


# ── Chat ─────────────────────────────────────────────────────────────────────

async def get_chat_history(page_id: str, limit: int = 10) -> list:
    db = await get_db()
    res = (
        await db.table("chat_messages")
        .select("role, content, message_type, meta, status")
        .eq("page_id", page_id)
        .in_("message_type", ["chat", "clarification"])
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return list(reversed(res.data))


async def get_consecutive_clarification_count(page_id: str) -> int:
    db = await get_db()
    res = (
        await db.table("clarification_threads")
        .select("id, resolved")
        .eq("page_id", page_id)
        .order("created_at", desc=True)
        .limit(5)
        .execute()
    )
    count = 0
    for row in res.data:
        if not row["resolved"]:
            count += 1
        else:
            break
    return count


async def update_message_status(message_id: str, status: str):
    db = await get_db()
    await db.table("chat_messages").update({
        "status": status
    }).eq("id", message_id).execute()


async def insert_assistant_message(
    page_id: str,
    content: str,
    message_type: str = "chat",
    meta: dict = None,
):
    db = await get_db()
    await db.table("chat_messages").insert({
        "page_id": page_id,
        "role": "assistant",
        "content": content,
        "status": "completed",
        "message_type": message_type,
        "meta": meta or {}
    }).execute()


async def insert_thinking_message(page_id: str, plan: dict) -> Optional[str]:
    db = await get_db()
    res = await db.table("chat_messages").insert({
        "page_id": page_id,
        "role": "assistant",
        "content": "thinking",
        "status": "completed",
        "message_type": "thinking",
        "meta": {"plan": plan}
    }).execute()
    return res.data[0]["id"] if res.data else None


async def snapshot_version(page_id: str, html: str, trigger_type: str = "agent_complete"):
    db = await get_db()
    res = (
        await db.table("page_versions")
        .select("version_num")
        .eq("page_id", page_id)
        .order("version_num", desc=True)
        .limit(1)
        .execute()
    )
    next_version = (res.data[0]["version_num"] + 1) if res.data else 1
    await db.table("page_versions").insert({
        "page_id": page_id,
        "html_snapshot": html,
        "version_num": next_version,
        "trigger_type": trigger_type
    }).execute()


async def get_page_versions(page_id: str, limit: int = 10) -> list:
    db = await get_db()
    res = (
        await db.table("page_versions")
        .select("id, version_num, trigger_type, created_at")
        .eq("page_id", page_id)
        .order("version_num", desc=True)
        .limit(limit)
        .execute()
    )
    return res.data or []


async def get_version_html(version_id: str) -> Optional[str]:
    db = await get_db()
    res = (
        await db.table("page_versions")
        .select("html_snapshot")
        .eq("id", version_id)
        .single()
        .execute()
    )
    return res.data["html_snapshot"] if res.data else None


async def get_edit_history(page_id: str, limit: int = 5) -> list:
    db = await get_db()
    res = (
        await db.table("edit_history")
        .select("*")
        .eq("page_id", page_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return list(reversed(res.data))


async def insert_edit_history(
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
    success: bool,
    owner_id: str = None,
):
    db = await get_db()
    await db.table("edit_history").insert({
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
        "success": success,
        "owner_id": owner_id
    }).execute()


async def insert_clarification(page_id: str, message_id: str, question: str) -> Optional[str]:
    db = await get_db()
    res = await db.table("clarification_threads").insert({
        "page_id": page_id,
        "message_id": message_id,
        "question": question,
        "resolved": False
    }).execute()
    return res.data[0]["id"] if res.data else None


async def resolve_clarification(clarification_id: str, answer: str):
    db = await get_db()
    await db.table("clarification_threads").update({
        "answer": answer,
        "resolved": True,
        "resolved_at": "now()"
    }).eq("id", clarification_id).execute()


async def get_pending_clarification(page_id: str) -> Optional[dict]:
    db = await get_db()
    res = (
        await db.table("clarification_threads")
        .select("*")
        .eq("page_id", page_id)
        .eq("resolved", False)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    return res.data[0] if res.data else None


# ── Assets ───────────────────────────────────────────────────────────────────

async def get_pending_assets_for_page(page_id: str) -> list:
    db = await get_db()
    res = (
        await db.table("page_assets")
        .select("*")
        .eq("page_id", page_id)
        .eq("processing_status", "pending")
        .is_("parent_asset_id", "null")
        .execute()
    )
    return res.data or []


async def get_page_assets_ready(page_id: str) -> list:
    db = await get_db()
    res = (
        await db.table("page_assets")
        .select("*")
        .eq("page_id", page_id)
        .eq("processing_status", "ready")
        .order("created_at", desc=False)
        .execute()
    )
    return res.data or []


async def update_asset_processing_started(asset_id: str):
    db = await get_db()
    await db.table("page_assets").update({
        "processing_status": "processing",
        "updated_at": "now()"
    }).eq("id", asset_id).execute()


async def update_asset_image_result(
    asset_id: str,
    vision_description: str,
    vision_tags: list,
    vision_suggested_use: str,
    vision_alt_text: str,
    vision_contains_text: bool,
    vision_extracted_text: str,
    dominant_colors: list,
):
    db = await get_db()
    await db.table("page_assets").update({
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


async def update_asset_document_result(
    asset_id: str,
    extracted_text: str,
    extracted_summary: str,
):
    db = await get_db()
    await db.table("page_assets").update({
        "processing_status": "ready",
        "extracted_text": extracted_text,
        "extracted_summary": extracted_summary,
        "updated_at": "now()"
    }).eq("id", asset_id).execute()


async def mark_asset_failed(asset_id: str, error: str):
    db = await get_db()
    await db.table("page_assets").update({
        "processing_status": "failed",
        "processing_error": error,
        "updated_at": "now()"
    }).eq("id", asset_id).execute()


async def insert_extracted_image_asset(
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
    db = await get_db()
    res = await db.table("page_assets").insert({
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
        "processing_status": "processing",
    }).execute()
    return res.data[0]["id"] if res.data else None


# ── Billing ───────────────────────────────────────────────────────────────────

async def deduct_dollar_credits(
    user_id: str,
    input_tokens: int,
    output_tokens: int,
    model_id: str,
    description: str,
    reference_id: str = None,
) -> dict:
    try:
        db = await get_db()
        res = await db.rpc("deduct_dollar_credits", {
            "p_user_id": user_id,
            "p_input_tokens": input_tokens,
            "p_output_tokens": output_tokens,
            "p_model_id": model_id,
            "p_description": description,
            "p_reference_id": reference_id,
        }).execute()
        return res.data or {"success": False, "error": "RPC returned no data"}
    except Exception as e:
        logger.warning("[DB] deduct_dollar_credits error: %s", e)
        return {"success": False, "error": str(e)}


async def check_token_balance(user_id: str) -> dict:
    try:
        db = await get_db()
        res = await db.rpc("check_token_balance", {"p_user_id": user_id}).execute()
        return res.data or {"has_balance": False, "balance": 0, "dollar_balance": 0.0}
    except Exception as e:
        logger.warning("[DB] check_token_balance error: %s", e)
        return {"has_balance": False, "balance": 0, "dollar_balance": 0.0}


async def deduct_tokens(
    user_id: str,
    amount: int,
    description: str,
    reference_id: str = None,
) -> dict:
    """Legacy flat-token deduction — kept for backward compat."""
    try:
        db = await get_db()
        res = await db.rpc("deduct_tokens", {
            "p_user_id": user_id,
            "p_amount": amount,
            "p_description": description,
            "p_reference_id": reference_id
        }).execute()
        return res.data or {"success": False, "error": "RPC failed"}
    except Exception as e:
        logger.warning("[DB] deduct_tokens error: %s", e)
        return {"success": False, "error": str(e)}


# ── Subscriptions ────────────────────────────────────────────────────────────

async def get_user_subscription(user_id: str) -> dict:
    db = await get_db()
    res = await db.rpc("get_user_subscription", {"p_user_id": user_id}).execute()
    return res.data or {}


async def upgrade_subscription(
    user_id: str,
    tier: str,
    razorpay_order_id: str,
    razorpay_payment_id: str,
    amount_usd: float,
) -> dict:
    db = await get_db()
    res = await db.rpc("upgrade_subscription", {
        "p_user_id": user_id,
        "p_tier": tier,
        "p_razorpay_order_id": razorpay_order_id,
        "p_razorpay_payment_id": razorpay_payment_id,
        "p_amount_usd": amount_usd
    }).execute()
    return res.data or {"success": False}


async def check_can_publish(user_id: str, page_id: str) -> dict:
    db = await get_db()
    res = await db.rpc("check_can_publish", {
        "p_user_id": user_id,
        "p_page_id": page_id
    }).execute()
    return res.data or {"allowed": False, "reason": "unknown"}


async def check_can_create_page(user_id: str) -> dict:
    db = await get_db()
    res = await db.rpc("check_can_create_page", {"p_user_id": user_id}).execute()
    return res.data or {"allowed": False, "reason": "unknown"}