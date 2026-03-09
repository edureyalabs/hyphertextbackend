# agents/orchestrator.py
"""
Main agent orchestrator for Hyphertext — AI-powered single-file HTML page builder.

Inference modes
────────────────────────────────────────────────────────────────────────
Economy (default)  → Together AI (GLM-5 for complex, GLM-4.7-Flash for simple)
Speed              → Cerebras (GLM-4.7 for ALL coding tasks, ~1000 TPS)

IMPORTANT: NO SILENT FALLBACKS.
If the selected mode's model fails, the error is surfaced directly to the user
with a clear message telling them to switch modes. We never silently fall back
to the other provider — that would be dishonest about which model is running
and could produce inconsistent results on pages built with a different model.

Mode is now MUTABLE — users can change it any time. When they do, the frontend
calls PATCH /api/pages/:id with { inference_mode, reset_model: true } which
clears coding_model_id so the next run re-routes from scratch.

Billing model (dollar-credit system)
────────────────────────────────────────────────────────────────────────
All AI usage is billed in dollars. Each model call tracks input_tokens and
output_tokens separately. At the end of a run, deduct_dollar_credits() is
called once with the total token counts and the primary model used.
"""

import json
import logging
import traceback
from agents.models import router as model_router
from agents.models.coding_router import select_coding_model
from agents.tools.html_tools import TOOL_DEFINITIONS, execute_str_replace
from agents.tools.search_tools import brave_search, format_search_results
from agents.knowledge.prompts import (
    build_orchestrator_system_prompt,
    build_planning_prompt,
    build_summary_generation_prompt,
    build_intent_classification_prompt,
    build_conversational_reply_prompt,
)
from agents.processors.asset_pipeline import process_pending_assets
from agents.processors.asset_context import build_asset_context
from database import (
    get_page,
    update_page_html,
    update_page_summary_and_map,
    update_page_coding_model,
    update_page_inference_mode,
    get_chat_history,
    get_edit_history,
    update_message_status,
    insert_assistant_message,
    insert_thinking_message,
    snapshot_version,
    insert_edit_history,
    insert_clarification,
    get_pending_clarification,
    resolve_clarification,
    get_consecutive_clarification_count,
    get_page_versions,
    get_version_html,
    deduct_dollar_credits,
    check_token_balance,
)
from boilerplate import INITIAL_BOILERPLATE
from config import PLANNING_MODEL, CONVERSATION_MODEL

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model error types — used to give the user actionable messages
# ---------------------------------------------------------------------------

class ModelProviderError(Exception):
    """
    Raised when a specific model provider fails (e.g. Cerebras 503, Together 429).
    Carries the inference_mode so the UI can show the right switch suggestion.
    """
    def __init__(self, message: str, inference_mode: str, provider: str):
        super().__init__(message)
        self.inference_mode = inference_mode
        self.provider = provider

    def user_facing_message(self) -> str:
        if self.inference_mode == "speed":
            return (
                "⚡ Speed mode (Cerebras) is currently unavailable. "
                "Please switch to Economy mode and try again."
            )
        else:
            return (
                "Economy mode (Together AI) is currently unavailable. "
                "Please switch to Speed mode and try again."
            )


def _wrap_model_call(fn, inference_mode: str, provider: str):
    """
    Wraps a model router call. If it raises, converts to ModelProviderError
    so the orchestrator can surface a clean message to the user.
    """
    try:
        return fn()
    except Exception as e:
        err_str = str(e).lower()
        # Detect provider-level failures (rate limits, outages, auth, bad params)
        is_provider_error = any(x in err_str for x in [
            "503", "502", "500", "429", "rate limit", "overloaded",
            "connection", "timeout", "unavailable", "badrequest",
            "400", "401", "403", "quota", "capacity"
        ])
        if is_provider_error:
            raise ModelProviderError(str(e), inference_mode, provider) from e
        raise  # re-raise unknown errors as-is


def _provider_for_mode(inference_mode: str) -> str:
    return "Cerebras" if inference_mode == "speed" else "Together AI"


# ---------------------------------------------------------------------------
# Token accounting helpers
# ---------------------------------------------------------------------------

class TokenLedger:
    def __init__(self):
        self._usage: dict[str, dict] = {}

    def add(self, model_id: str, input_tokens: int, output_tokens: int):
        if model_id not in self._usage:
            self._usage[model_id] = {"input": 0, "output": 0}
        self._usage[model_id]["input"]  += input_tokens
        self._usage[model_id]["output"] += output_tokens

    def total_tokens(self) -> int:
        return sum(v["input"] + v["output"] for v in self._usage.values())

    def flush(self, user_id: str, description: str, reference_id: str = None):
        if not user_id:
            return
        for model_id, usage in self._usage.items():
            if usage["input"] == 0 and usage["output"] == 0:
                continue
            deduct_dollar_credits(
                user_id=user_id,
                input_tokens=usage["input"],
                output_tokens=usage["output"],
                model_id=model_id,
                description=description,
                reference_id=reference_id,
            )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_boilerplate(html: str) -> bool:
    if not html:
        return True
    if html.strip() == INITIAL_BOILERPLATE.strip():
        return True
    if "describe what you want to build" in html:
        return True
    return False


def _parse_plan(raw: str) -> dict:
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = [l for l in cleaned.split("\n") if not l.startswith("```")]
            cleaned = "\n".join(lines)
        return json.loads(cleaned)
    except Exception:
        return {
            "decision": "surgical_edit",
            "complexity": "simple",
            "confidence": 0.5,
            "needs_clarification": False,
            "clarification_question": None,
            "description": "apply requested changes",
            "changes": [],
            "needs_web_search": False,
            "search_query": None,
        }


def _classify_intent(user_prompt: str, chat_history: list) -> str:
    history_context = ""
    if chat_history:
        lines = []
        for msg in chat_history[-6:]:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if len(content) > 150:
                content = content[:150] + "..."
            lines.append(f"{role.upper()}: {content}")
        history_context = "\n".join(lines)

    messages = [
        {
            "role": "system",
            "content": build_intent_classification_prompt(),
        },
    ]

    if history_context:
        messages.append({
            "role": "user",
            "content": (
                f"RECENT CONVERSATION CONTEXT (for your reference only):\n"
                f"{history_context}\n\n"
                f"NOW CLASSIFY THIS NEW MESSAGE:\n{user_prompt}"
            ),
        })
    else:
        messages.append({"role": "user", "content": user_prompt})

    try:
        response = model_router.chat(
            model_id=PLANNING_MODEL,
            messages=messages,
            max_tokens=10,
            temperature=0.0,
        )
        result = response["content"].strip().lower()
        if "revert" in result:
            return "revert"
        if "conversational" in result:
            return "conversational"
        return "code_change"
    except Exception:
        return "code_change"


# ---------------------------------------------------------------------------
# Intent handlers
# ---------------------------------------------------------------------------

async def _handle_conversational(
    page_id: str,
    message_id: str,
    user_prompt: str,
    owner_id: str,
    ledger: TokenLedger,
    chat_history: list,
    page_title: str = "",
):
    system_content = build_conversational_reply_prompt(
        user_prompt=user_prompt,
        chat_history=chat_history,
        page_title=page_title,
    )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_prompt},
    ]

    response = model_router.chat(
        model_id=CONVERSATION_MODEL,
        messages=messages,
        max_tokens=300,
        temperature=0.6,
    )
    ledger.add(CONVERSATION_MODEL, response["input_tokens"], response["output_tokens"])

    reply = response["content"] or "Happy to help! What would you like to build or change?"
    update_message_status(message_id, "completed")
    insert_assistant_message(page_id, reply)
    ledger.flush(owner_id, "Conversation reply", message_id)
    insert_edit_history(
        page_id=page_id,
        message_id=message_id,
        complexity="simple",
        decision="conversational",
        plan_json={},
        changes_json=[],
        clarification_asked=False,
        web_searches_used=[],
        model_used=CONVERSATION_MODEL,
        tokens_used=ledger.total_tokens(),
        success=True,
        owner_id=owner_id,
    )


async def _handle_revert(
    page_id: str,
    message_id: str,
    user_prompt: str,
    owner_id: str,
):
    versions = get_page_versions(page_id, limit=5)
    if len(versions) < 2:
        update_message_status(message_id, "completed")
        insert_assistant_message(page_id, "There are no previous versions to revert to yet.")
        return

    previous = versions[1]
    html = get_version_html(previous["id"])
    if not html:
        update_message_status(message_id, "completed")
        insert_assistant_message(page_id, "Could not retrieve the previous version. Please try again.")
        return

    update_page_html(page_id, html)
    snapshot_version(page_id, html, trigger_type="revert")
    update_message_status(message_id, "completed")
    insert_assistant_message(page_id, "Done. I've reverted to the previous version of your page.")
    insert_edit_history(
        page_id=page_id,
        message_id=message_id,
        complexity="simple",
        decision="revert",
        plan_json={"reverted_to_version": previous["version_num"]},
        changes_json=[],
        clarification_asked=False,
        web_searches_used=[],
        model_used=CONVERSATION_MODEL,
        tokens_used=0,
        success=True,
        owner_id=owner_id,
    )


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------

async def _generate_summary_if_needed(
    page_id: str,
    current_html: str,
    html_summary: str,
    ledger: TokenLedger,
) -> str:
    if html_summary or _is_boilerplate(current_html):
        return html_summary

    messages = [
        {
            "role": "system",
            "content": "You analyze HTML pages and produce structured summaries for an AI coding agent.",
        },
        {"role": "user", "content": build_summary_generation_prompt(current_html)},
    ]
    try:
        response = model_router.chat(
            model_id=PLANNING_MODEL,
            messages=messages,
            max_tokens=800,
            temperature=0.1,
        )
        ledger.add(PLANNING_MODEL, response["input_tokens"], response["output_tokens"])
        raw = response["content"] or ""
        try:
            parsed = json.loads(raw.strip().lstrip("```json").rstrip("```").strip())
            summary = parsed.get("html_summary", raw)
            component_map = parsed.get("component_map", [])
            update_page_summary_and_map(page_id, summary, component_map)
            return summary
        except Exception:
            update_page_summary_and_map(page_id, raw, [])
            return raw
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run_orchestrator(
    page_id: str,
    message_id: str,
    user_prompt: str,
    owner_id: str = None,
    requested_inference_mode: str = None,
):
    """
    Run the full agent loop for a single user message.

    Args:
        page_id:                  The page being edited.
        message_id:               The user's chat message ID.
        user_prompt:              The user's text input.
        owner_id:                 User ID for billing.
        requested_inference_mode: "economy" or "speed" — honoured on every
                                  message now (mode is mutable). If the page
                                  has a persisted mode AND the request doesn't
                                  supply one, the persisted mode wins.
    """
    ledger = TokenLedger()
    web_searches_used = []
    changes_log = []
    plan = {}
    clarification_asked = False
    coding_model = None
    inference_mode = "economy"

    try:
        update_message_status(message_id, "processing")

        # ── dollar balance check ──────────────────────────────────────────────
        if owner_id:
            balance_check = check_token_balance(owner_id)
            if not balance_check.get("has_balance", True):
                dollar_balance = balance_check.get("dollar_balance", 0.0)
                update_message_status(message_id, "error")
                insert_assistant_message(
                    page_id,
                    "You have run out of credits. Please purchase more credits to continue using the AI agent.",
                    meta={
                        "insufficient_tokens": True,
                        "dollar_balance": float(dollar_balance),
                        "balance": balance_check.get("balance", 0),
                    },
                )
                return

        # ── load page + history ───────────────────────────────────────────────
        page = get_page(page_id)
        current_html      = page.get("html_content", "")
        html_summary      = page.get("html_summary", "")
        component_map     = page.get("component_map", [])
        persisted_model   = page.get("coding_model_id")
        persisted_mode    = page.get("inference_mode")
        page_title        = page.get("title", "")

        edit_history = get_edit_history(page_id, limit=5)
        chat_history = get_chat_history(page_id, limit=10)

        # ── resolve inference mode ────────────────────────────────────────────
        # Priority: request value > persisted DB value > default "economy"
        # The request value always wins now — mode is mutable.
        # When the user switches modes, the frontend sends the new mode on the
        # next message and we update the DB here.
        if requested_inference_mode in ("economy", "speed"):
            inference_mode = requested_inference_mode
        elif persisted_mode in ("economy", "speed"):
            inference_mode = persisted_mode
        else:
            inference_mode = "economy"

        # ── persist mode (update if changed) ─────────────────────────────────
        if inference_mode != persisted_mode:
            update_page_inference_mode(page_id, inference_mode)
            # When mode changes, clear the persisted coding_model_id so the
            # router picks the right model for the new provider on this run.
            if persisted_mode is not None:
                update_page_coding_model(page_id, None)
                persisted_model = None
            logger.info(
                "[orchestrator] page=%s — inference_mode changed from '%s' to '%s'",
                page_id, persisted_mode, inference_mode
            )

        # ── intent classification ─────────────────────────────────────────────
        intent = _classify_intent(user_prompt, chat_history)

        if intent == "conversational":
            await _handle_conversational(
                page_id=page_id,
                message_id=message_id,
                user_prompt=user_prompt,
                owner_id=owner_id,
                ledger=ledger,
                chat_history=chat_history,
                page_title=page_title,
            )
            return

        if intent == "revert":
            await _handle_revert(page_id, message_id, user_prompt, owner_id)
            return

        ledger.add(PLANNING_MODEL, 20, 3)

        # ── process pending uploads ───────────────────────────────────────────
        if owner_id:
            await process_pending_assets(page_id, owner_id)

        # ── build asset context ───────────────────────────────────────────────
        asset_context = build_asset_context(page_id)

        # ── lazily generate summary ───────────────────────────────────────────
        is_new_page = _is_boilerplate(current_html)
        is_imported = page.get("page_source") == "import" and not html_summary

        if not html_summary and not is_new_page:
            html_summary = await _generate_summary_if_needed(
                page_id, current_html, html_summary, ledger
            )

        # ── resolve pending clarification ─────────────────────────────────────
        pending_clarification = get_pending_clarification(page_id)
        if pending_clarification:
            resolve_clarification(pending_clarification["id"], user_prompt)
            user_prompt = (
                f"Earlier you asked: {pending_clarification['question']}\n"
                f"User answered: {user_prompt}\n"
                f"Now proceed with the original task using this information."
            )

        # ── clarification guard ───────────────────────────────────────────────
        consecutive_clarifications = get_consecutive_clarification_count(page_id)
        clarification_blocked = consecutive_clarifications >= 2

        # ── planning ──────────────────────────────────────────────────────────
        planning_messages = [
            {
                "role": "system",
                "content": (
                    "You are a planning assistant for Hyphertext — an AI-powered HTML page builder. "
                    "ALL user requests are requests to build or modify an HTML page. "
                    "Analyse the user request and return a structured JSON plan."
                ),
            },
            {"role": "user", "content": build_planning_prompt(user_prompt)},
        ]
        plan_response = model_router.chat(
            model_id=PLANNING_MODEL,
            messages=planning_messages,
            max_tokens=1000,
            temperature=0.1,
        )
        ledger.add(PLANNING_MODEL, plan_response["input_tokens"], plan_response["output_tokens"])
        plan = _parse_plan(plan_response["content"])

        if is_new_page:
            plan["decision"] = "full_rewrite"
            plan["needs_clarification"] = False

        if is_imported and not is_new_page and plan.get("decision") != "full_rewrite":
            plan["decision"] = "surgical_edit"

        if clarification_blocked and plan.get("needs_clarification"):
            plan["needs_clarification"] = False
            plan["forced_proceed"] = True

        # ── handle clarification ──────────────────────────────────────────────
        if (
            plan.get("needs_clarification")
            and not is_new_page
            and plan.get("confidence", 1.0) < 0.6
        ):
            question = plan.get("clarification_question", "Could you clarify what you would like?")
            insert_clarification(page_id, message_id, question)
            clarification_asked = True
            update_message_status(message_id, "completed")
            insert_assistant_message(
                page_id,
                question,
                message_type="clarification",
                meta={"awaiting_clarification": True, "reason": plan.get("description", "")},
            )
            insert_edit_history(
                page_id=page_id,
                message_id=message_id,
                complexity=plan.get("complexity", "simple"),
                decision="clarification",
                plan_json=plan,
                changes_json=[],
                clarification_asked=True,
                web_searches_used=[],
                model_used=PLANNING_MODEL,
                tokens_used=ledger.total_tokens(),
                success=True,
                owner_id=owner_id,
            )
            ledger.flush(owner_id, "Planning (clarification)", message_id)
            return

        # ── select coding model ───────────────────────────────────────────────
        coding_model = select_coding_model(
            plan=plan,
            is_new_page=is_new_page,
            is_imported=is_imported,
            inference_mode=inference_mode,
            override_model_id=persisted_model,
        )

        logger.info(
            "[orchestrator] page=%s — routing to model='%s' (mode='%s')",
            page_id, coding_model, inference_mode
        )

        if not persisted_model:
            update_page_coding_model(page_id, coding_model)

        insert_thinking_message(
            page_id,
            {**plan, "_coding_model": coding_model, "_inference_mode": inference_mode}
        )

        # ── optional web search ───────────────────────────────────────────────
        if plan.get("needs_web_search") and plan.get("search_query"):
            search_results = await brave_search(plan["search_query"])
            web_searches_used.append(
                {"query": plan["search_query"], "results": search_results}
            )
            search_context = (
                f"\nWEB SEARCH RESULTS for '{plan['search_query']}':\n"
                f"{format_search_results(search_results)}\n"
            )
        else:
            search_context = ""

        # ── build system prompt ───────────────────────────────────────────────
        system_prompt = build_orchestrator_system_prompt(
            current_html=current_html,
            html_summary=html_summary,
            component_map=component_map,
            edit_history=edit_history,
            chat_history=chat_history,
        )

        if asset_context:
            system_prompt += f"\n\n{asset_context}"
        if search_context:
            system_prompt += f"\n\n{search_context}"
        if is_imported and not is_new_page:
            system_prompt += (
                "\n\n## IMPORTED PAGE NOTE\n"
                "This page was imported by the user from existing code. "
                "Preserve the existing layout, structure, and design unless explicitly told to change it. "
                "Make only the specific requested changes surgically."
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Build or edit the HTML page for this request. "
                    f"Call the appropriate tool immediately — do not write any prose.\n\n"
                    f"REQUEST: {user_prompt}"
                ),
            },
        ]

        # ── agentic tool loop ─────────────────────────────────────────────────
        provider = _provider_for_mode(inference_mode)
        max_iterations = 15
        iteration = 0
        final_summary = "Done."

        while iteration < max_iterations:
            iteration += 1

            # ── HARD FAIL: no silent fallback ─────────────────────────────────
            response = _wrap_model_call(
                lambda: model_router.chat(
                    model_id=coding_model,
                    messages=messages,
                    tools=TOOL_DEFINITIONS,
                    tool_choice="auto",
                    max_tokens=8000,
                    temperature=0.3,
                ),
                inference_mode=inference_mode,
                provider=provider,
            )
            ledger.add(coding_model, response["input_tokens"], response["output_tokens"])

            if not response["tool_calls"]:
                content = (response["content"] or "").strip()
                if changes_log:
                    final_summary = content if content else "Edits complete."
                    snapshot_version(page_id, current_html)
                    update_message_status(message_id, "completed")
                    insert_assistant_message(page_id, final_summary)
                    insert_edit_history(
                        page_id=page_id,
                        message_id=message_id,
                        complexity=plan.get("complexity", "simple"),
                        decision="surgical_edit",
                        plan_json=plan,
                        changes_json=changes_log,
                        clarification_asked=clarification_asked,
                        web_searches_used=web_searches_used,
                        model_used=coding_model,
                        tokens_used=ledger.total_tokens(),
                        success=True,
                        owner_id=owner_id,
                    )
                    ledger.flush(owner_id, f"AI edit: {final_summary[:80]}", message_id)
                    return

                if iteration < max_iterations:
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": (
                            "You must call a tool. Do not write prose responses. "
                            "This is an HTML page builder — call write_full_file to build the page now. "
                            "Do not explain anything. Just call the tool."
                        ),
                    })
                    continue
                else:
                    final_summary = "I wasn't able to complete that. Please try rephrasing your request."
                    break

            tool_results_for_messages = []

            for tool_call in response["tool_calls"]:
                fn_name = tool_call["name"]
                args    = tool_call["arguments"]
                tc_id   = tool_call["id"]

                if fn_name == "write_full_file":
                    html              = args.get("html", "")
                    summary           = args.get("summary", "Page created.")
                    new_html_summary  = args.get("html_summary", "")
                    new_component_map = args.get("component_map", [])

                    if not html:
                        tool_results_for_messages.append({
                            "tool_call_id": tc_id,
                            "result": (
                                "ERROR: html field is empty. You MUST provide the complete HTML content "
                                "in the html parameter. Call write_full_file again with the full HTML."
                            ),
                        })
                        continue

                    if "<!DOCTYPE" not in html and "<html" not in html:
                        tool_results_for_messages.append({
                            "tool_call_id": tc_id,
                            "result": (
                                "ERROR: html field does not contain a valid HTML document. "
                                "It must start with <!DOCTYPE html> and include a full <html> structure. "
                                "Call write_full_file again with the complete valid HTML document."
                            ),
                        })
                        continue

                    update_page_html(page_id, html)
                    current_html = html

                    if new_html_summary:
                        update_page_summary_and_map(page_id, new_html_summary, new_component_map)

                    changes_log.append({"tool": "write_full_file", "summary": summary, "success": True})
                    final_summary = summary
                    snapshot_version(page_id, html)
                    update_message_status(message_id, "completed")
                    insert_assistant_message(page_id, summary)
                    insert_edit_history(
                        page_id=page_id,
                        message_id=message_id,
                        complexity=plan.get("complexity", "moderate"),
                        decision="full_rewrite",
                        plan_json=plan,
                        changes_json=changes_log,
                        clarification_asked=clarification_asked,
                        web_searches_used=web_searches_used,
                        model_used=coding_model,
                        tokens_used=ledger.total_tokens(),
                        success=True,
                        owner_id=owner_id,
                    )
                    ledger.flush(owner_id, f"AI page build: {summary[:80]}", message_id)
                    return

                elif fn_name == "str_replace":
                    old_str = args.get("old_str", "")
                    new_str = args.get("new_str", "")
                    updated_html, success = execute_str_replace(current_html, old_str, new_str)

                    if success:
                        current_html = updated_html
                        update_page_html(page_id, current_html)
                        changes_log.append({
                            "tool": "str_replace",
                            "old_str_preview": old_str[:80],
                            "success": True,
                        })
                        tool_results_for_messages.append({
                            "tool_call_id": tc_id,
                            "result": "Replaced successfully. Continue with the next change or call finish if done.",
                        })
                    else:
                        changes_log.append({
                            "tool": "str_replace",
                            "old_str_preview": old_str[:80],
                            "success": False,
                        })
                        tool_results_for_messages.append({
                            "tool_call_id": tc_id,
                            "result": (
                                "ERROR: old_str not found in the file. "
                                "The string must match EXACTLY including whitespace and indentation. "
                                "Try a shorter, more unique substring. "
                                "Check the current HTML in your context window carefully."
                            ),
                        })

                elif fn_name == "ask_clarification":
                    question = args.get("question", "Could you clarify?")
                    insert_clarification(page_id, message_id, question)
                    clarification_asked = True
                    update_message_status(message_id, "completed")
                    insert_assistant_message(
                        page_id,
                        question,
                        message_type="clarification",
                        meta={"awaiting_clarification": True},
                    )
                    insert_edit_history(
                        page_id=page_id,
                        message_id=message_id,
                        complexity=plan.get("complexity", "simple"),
                        decision="clarification",
                        plan_json=plan,
                        changes_json=[],
                        clarification_asked=True,
                        web_searches_used=web_searches_used,
                        model_used=coding_model,
                        tokens_used=ledger.total_tokens(),
                        success=True,
                        owner_id=owner_id,
                    )
                    ledger.flush(owner_id, "Planning (clarification)", message_id)
                    return

                elif fn_name == "web_search":
                    query = args.get("query", "")
                    search_results = await brave_search(query)
                    web_searches_used.append({"query": query, "results": search_results})
                    formatted = format_search_results(search_results)
                    tool_results_for_messages.append({
                        "tool_call_id": tc_id,
                        "result": formatted,
                    })

                elif fn_name == "finish":
                    final_summary = args.get("summary", "Edits complete.")
                    snapshot_version(page_id, current_html)
                    update_message_status(message_id, "completed")
                    insert_assistant_message(page_id, final_summary)
                    insert_edit_history(
                        page_id=page_id,
                        message_id=message_id,
                        complexity=plan.get("complexity", "simple"),
                        decision="surgical_edit",
                        plan_json=plan,
                        changes_json=changes_log,
                        clarification_asked=clarification_asked,
                        web_searches_used=web_searches_used,
                        model_used=coding_model,
                        tokens_used=ledger.total_tokens(),
                        success=True,
                        owner_id=owner_id,
                    )
                    ledger.flush(owner_id, f"AI edit: {final_summary[:80]}", message_id)
                    return

            assistant_msg = {
                "role": "assistant",
                "content": response["content"] or "",
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["arguments"]),
                        },
                    }
                    for tc in response["tool_calls"]
                ],
            }
            messages.append(assistant_msg)

            for result in tool_results_for_messages:
                messages.append({
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "content": result["result"],
                })

        # ── max iterations reached ────────────────────────────────────────────
        snapshot_version(page_id, current_html)
        update_message_status(message_id, "completed")
        insert_assistant_message(page_id, final_summary)
        insert_edit_history(
            page_id=page_id,
            message_id=message_id,
            complexity=plan.get("complexity", "simple"),
            decision=plan.get("decision", "surgical_edit"),
            plan_json=plan,
            changes_json=changes_log,
            clarification_asked=clarification_asked,
            web_searches_used=web_searches_used,
            model_used=coding_model or PLANNING_MODEL,
            tokens_used=ledger.total_tokens(),
            success=True,
            owner_id=owner_id,
        )
        ledger.flush(owner_id, f"AI edit: {final_summary[:80]}", message_id)

    except ModelProviderError as e:
        # ── Hard fail: surface the provider error directly to the user ────────
        logger.error(
            "[orchestrator] ModelProviderError page=%s message=%s mode=%s provider=%s: %s",
            page_id, message_id, e.inference_mode, e.provider, str(e)
        )
        update_message_status(message_id, "error")
        insert_assistant_message(
            page_id,
            e.user_facing_message(),
            meta={
                "model_provider_error": True,
                "inference_mode": e.inference_mode,
                "provider": e.provider,
            },
        )
        insert_edit_history(
            page_id=page_id,
            message_id=message_id,
            complexity="unknown",
            decision="unknown",
            plan_json=plan,
            changes_json=changes_log,
            clarification_asked=clarification_asked,
            web_searches_used=web_searches_used,
            model_used=coding_model or PLANNING_MODEL,
            tokens_used=ledger.total_tokens(),
            success=False,
            owner_id=owner_id,
        )
        # Don't bill for a failed provider call
        # ledger.flush is intentionally skipped here for the coding model cost

    except Exception as e:
        logger.error(
            "[orchestrator] UNHANDLED ERROR page=%s message=%s\n%s",
            page_id, message_id, traceback.format_exc()
        )
        update_message_status(message_id, "error")
        insert_assistant_message(
            page_id,
            f"Something went wrong: {str(e)[:120]}. Please try again.",
        )
        insert_edit_history(
            page_id=page_id,
            message_id=message_id,
            complexity="unknown",
            decision="unknown",
            plan_json=plan,
            changes_json=changes_log,
            clarification_asked=clarification_asked,
            web_searches_used=web_searches_used,
            model_used=coding_model or PLANNING_MODEL,
            tokens_used=ledger.total_tokens(),
            success=False,
            owner_id=owner_id,
        )
        ledger.flush(owner_id, "AI edit (error)", message_id)