from supabase import create_client, Client
from config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
from typing import Optional

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def get_supabase_client() -> Client:
    return supabase


# ── Pages ────────────────────────────────────────────────────────────────────

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


def update_page_coding_model(page_id: str, coding_model_id: str):
    try:
        supabase.table("pages").update({
            "coding_model_id": coding_model_id,
            "updated_at": "now()"
        }).eq("id", page_id).execute()
    except Exception as e:
        print(f"[DB] update_page_coding_model failed (column may not exist yet): {e}")


# ── Chat ─────────────────────────────────────────────────────────────────────

def get_chat_history(page_id: str, limit: int = 10) -> list:
    res = (
        supabase.table("chat_messages")
        .select("role, content, message_type, meta, status")
        .eq("page_id", page_id)
        .in_("message_type", ["chat", "clarification"])
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return list(reversed(res.data))


def get_consecutive_clarification_count(page_id: str) -> int:
    res = (
        supabase.table("clarification_threads")
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


def snapshot_version(page_id: str, html: str, trigger_type: str = "agent_complete"):
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
        "trigger_type": trigger_type
    }).execute()


def get_page_versions(page_id: str, limit: int = 10) -> list:
    res = (
        supabase.table("page_versions")
        .select("id, version_num, trigger_type, created_at")
        .eq("page_id", page_id)
        .order("version_num", desc=True)
        .limit(limit)
        .execute()
    )
    return res.data or []


def get_version_html(version_id: str) -> Optional[str]:
    res = (
        supabase.table("page_versions")
        .select("html_snapshot")
        .eq("id", version_id)
        .single()
        .execute()
    )
    return res.data["html_snapshot"] if res.data else None


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
    success: bool,
    owner_id: str = None
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
        "success": success,
        "owner_id": owner_id
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


# ── Assets ───────────────────────────────────────────────────────────────────

def get_pending_assets_for_page(page_id: str) -> list:
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
    res = (
        supabase.table("page_assets")
        .select("*")
        .eq("page_id", page_id)
        .eq("processing_status", "ready")
        .order("created_at", desc=False)
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
        "processing_status": "processing",
    }).execute()
    return res.data[0]["id"] if res.data else None


# ── Billing: Dollar-credit system ────────────────────────────────────────────
#
# All AI usage is now billed in dollars based on per-model token pricing.
# The model_pricing table stores input/output price per 1M tokens for each model.
# deduct_dollar_credits RPC calculates cost, deducts from dollar_balance, and
# records the transaction with full token breakdown.
#
# Legacy deduct_tokens is kept only for any remaining callers during transition.

def deduct_dollar_credits(
    user_id: str,
    input_tokens: int,
    output_tokens: int,
    model_id: str,
    description: str,
    reference_id: str = None,
) -> dict:
    """
    Deduct AI usage cost from the user's dollar_balance.

    Pricing is looked up from the model_pricing table in Supabase.
    If the model is not found (e.g. free/unknown), cost defaults to $0.

    Returns the RPC result dict with keys:
        success, dollar_cost, new_balance, input_tokens, output_tokens
    """
    try:
        res = supabase.rpc("deduct_dollar_credits", {
            "p_user_id": user_id,
            "p_input_tokens": input_tokens,
            "p_output_tokens": output_tokens,
            "p_model_id": model_id,
            "p_description": description,
            "p_reference_id": reference_id,
        }).execute()
        return res.data or {"success": False, "error": "RPC returned no data"}
    except Exception as e:
        print(f"[DB] deduct_dollar_credits error: {e}")
        return {"success": False, "error": str(e)}


def check_token_balance(user_id: str) -> dict:
    """
    Returns has_balance (bool), balance (legacy tokens), dollar_balance (float).
    has_balance is True when dollar_balance >= $0.001.
    """
    try:
        res = supabase.rpc("check_token_balance", {"p_user_id": user_id}).execute()
        return res.data or {"has_balance": False, "balance": 0, "dollar_balance": 0.0}
    except Exception as e:
        print(f"[DB] check_token_balance error: {e}")
        return {"has_balance": False, "balance": 0, "dollar_balance": 0.0}


# ── Legacy token deduction — kept for backward compat, prefer deduct_dollar_credits ──

def deduct_tokens(user_id: str, amount: int, description: str, reference_id: str = None) -> dict:
    """
    Legacy flat-token deduction. Still functional but no longer called by the
    orchestrator. Kept so any in-flight code or webhooks don't break.
    """
    try:
        res = supabase.rpc("deduct_tokens", {
            "p_user_id": user_id,
            "p_amount": amount,
            "p_description": description,
            "p_reference_id": reference_id
        }).execute()
        return res.data or {"success": False, "error": "RPC failed"}
    except Exception as e:
        print(f"[DB] deduct_tokens error: {e}")
        return {"success": False, "error": str(e)}


# ── Subscriptions ────────────────────────────────────────────────────────────

def get_user_subscription(user_id: str) -> dict:
    res = supabase.rpc("get_user_subscription", {"p_user_id": user_id}).execute()
    return res.data or {}


def upgrade_subscription(user_id: str, tier: str, razorpay_order_id: str, razorpay_payment_id: str, amount_usd: float) -> dict:
    res = supabase.rpc("upgrade_subscription", {
        "p_user_id": user_id,
        "p_tier": tier,
        "p_razorpay_order_id": razorpay_order_id,
        "p_razorpay_payment_id": razorpay_payment_id,
        "p_amount_usd": amount_usd
    }).execute()
    return res.data or {"success": False}


def check_can_publish(user_id: str, page_id: str) -> dict:
    res = supabase.rpc("check_can_publish", {
        "p_user_id": user_id,
        "p_page_id": page_id
    }).execute()
    return res.data or {"allowed": False, "reason": "unknown"}


def check_can_create_page(user_id: str) -> dict:
    res = supabase.rpc("check_can_create_page", {"p_user_id": user_id}).execute()
    return res.data or {"allowed": False, "reason": "unknown"}