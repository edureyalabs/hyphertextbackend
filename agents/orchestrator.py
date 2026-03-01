import json
from agents.models import router as model_router
from agents.tools.html_tools import TOOL_DEFINITIONS, execute_str_replace
from agents.tools.search_tools import brave_search, format_search_results
from agents.knowledge.prompts import build_orchestrator_system_prompt, build_planning_prompt, build_summary_generation_prompt
from agents.processors.asset_pipeline import process_pending_assets
from agents.processors.asset_context import build_asset_context
from database import (
    get_page,
    update_page_html,
    update_page_summary_and_map,
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
    deduct_tokens,
    check_token_balance,
)
from boilerplate import INITIAL_BOILERPLATE

CONVERSATIONAL_MODEL = "groq/llama-3.1-8b"
CLASSIFICATION_MODEL = "groq/llama-3.1-8b"


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
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
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
            "search_query": None
        }


def _classify_intent(user_prompt: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an intent classifier for an HTML page builder. "
                "Classify the user message into exactly one of these categories:\n"
                "- conversational: greetings, thanks, feedback, questions about the platform, general chat, expressions of satisfaction\n"
                "- revert: user wants to undo, go back, revert, restore a previous version\n"
                "- code_change: any request to modify, build, create, edit, fix, update, add, remove, change the page\n"
                "Reply with only one word: conversational, revert, or code_change"
            )
        },
        {"role": "user", "content": user_prompt}
    ]
    try:
        response = model_router.chat(
            model_id=CLASSIFICATION_MODEL,
            messages=messages,
            max_tokens=10,
            temperature=0.0
        )
        result = response["content"].strip().lower()
        if result in ("conversational", "revert", "code_change"):
            return result
        return "code_change"
    except Exception:
        return "code_change"


async def _handle_conversational(page_id: str, message_id: str, user_prompt: str, model_id: str, owner_id: str, tokens_used: int):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant for Hyphertext, an AI-powered HTML page builder and hosting platform. "
                "You help users build, edit, and host single-file HTML pages. "
                "Be friendly, concise, and encouraging. "
                "If they ask about capabilities, tell them they can describe any web page and you will build it, "
                "they can upload images and documents to include in their pages, "
                "they can publish pages to get a live URL instantly. "
                "Do not mention code or technical details unless asked."
            )
        },
        {"role": "user", "content": user_prompt}
    ]
    response = model_router.chat(
        model_id=model_id,
        messages=messages,
        max_tokens=400,
        temperature=0.7
    )
    tokens_used += response["input_tokens"] + response["output_tokens"]
    reply = response["content"] or "Happy to help! What would you like to build or change?"
    update_message_status(message_id, "completed")
    insert_assistant_message(page_id, reply)
    if owner_id:
        deduct_tokens(owner_id, tokens_used, "Conversation reply", message_id)
    insert_edit_history(
        page_id=page_id,
        message_id=message_id,
        complexity="simple",
        decision="conversational",
        plan_json={},
        changes_json=[],
        clarification_asked=False,
        web_searches_used=[],
        model_used=model_id,
        tokens_used=tokens_used,
        success=True,
        owner_id=owner_id
    )


async def _handle_revert(page_id: str, message_id: str, user_prompt: str, model_id: str, owner_id: str):
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
    insert_assistant_message(page_id, "Done. I have reverted to the previous version of your page.")
    insert_edit_history(
        page_id=page_id,
        message_id=message_id,
        complexity="simple",
        decision="revert",
        plan_json={"reverted_to_version": previous["version_num"]},
        changes_json=[],
        clarification_asked=False,
        web_searches_used=[],
        model_used=model_id,
        tokens_used=0,
        success=True,
        owner_id=owner_id
    )


async def _generate_summary_if_needed(page_id: str, current_html: str, html_summary: str, model_id: str) -> tuple[str, int]:
    if html_summary or _is_boilerplate(current_html):
        return html_summary, 0

    messages = [
        {"role": "system", "content": "You analyze HTML pages and produce structured summaries for an AI coding agent."},
        {"role": "user", "content": build_summary_generation_prompt(current_html)}
    ]
    try:
        response = model_router.chat(
            model_id=model_id,
            messages=messages,
            max_tokens=800,
            temperature=0.1
        )
        tokens_used = response["input_tokens"] + response["output_tokens"]
        raw = response["content"] or ""
        try:
            parsed = json.loads(raw.strip().lstrip("```json").rstrip("```").strip())
            summary = parsed.get("html_summary", raw)
            component_map = parsed.get("component_map", [])
            update_page_summary_and_map(page_id, summary, component_map)
            return summary, tokens_used
        except Exception:
            update_page_summary_and_map(page_id, raw, [])
            return raw, tokens_used
    except Exception:
        return "", 0


async def run_orchestrator(page_id: str, message_id: str, user_prompt: str, model_id: str, owner_id: str = None):
    tokens_used = 0
    web_searches_used = []
    changes_log = []
    plan = {}
    clarification_asked = False

    try:
        update_message_status(message_id, "processing")

        # token balance check
        if owner_id:
            balance_check = check_token_balance(owner_id)
            if not balance_check.get("has_balance", True):
                update_message_status(message_id, "error")
                insert_assistant_message(
                    page_id,
                    "You have run out of tokens. Please purchase more tokens to continue using the AI agent.",
                    meta={"insufficient_tokens": True, "balance": balance_check.get("balance", 0)}
                )
                return

        # intent classification
        intent = _classify_intent(user_prompt)
        tokens_used += 15

        if intent == "conversational":
            await _handle_conversational(page_id, message_id, user_prompt, model_id, owner_id, tokens_used)
            return

        if intent == "revert":
            await _handle_revert(page_id, message_id, user_prompt, model_id, owner_id)
            return

        # process pending assets
        if owner_id:
            await process_pending_assets(page_id, owner_id)

        # build asset context
        asset_context = build_asset_context(page_id)

        # load page and history
        page = get_page(page_id)
        current_html = page.get("html_content", "")
        html_summary = page.get("html_summary", "")
        component_map = page.get("component_map", [])

        edit_history = get_edit_history(page_id, limit=5)
        chat_history = get_chat_history(page_id, limit=10)

        # lazily generate summary for imported pages
        if not html_summary and not _is_boilerplate(current_html):
            html_summary, summary_tokens = await _generate_summary_if_needed(page_id, current_html, html_summary, model_id)
            tokens_used += summary_tokens

        # handle pending clarification
        pending_clarification = get_pending_clarification(page_id)
        if pending_clarification:
            resolve_clarification(pending_clarification["id"], user_prompt)
            user_prompt = (
                f"Earlier you asked: {pending_clarification['question']}\n"
                f"User answered: {user_prompt}\n"
                f"Now proceed with the original task using this information."
            )

        is_new_page = _is_boilerplate(current_html)
        is_imported = page.get("page_source") == "import" and not html_summary

        # check consecutive clarification limit
        consecutive_clarifications = get_consecutive_clarification_count(page_id)
        clarification_blocked = consecutive_clarifications >= 2

        # planning
        planning_messages = [
            {
                "role": "system",
                "content": (
                    "You are a planning assistant for an HTML coding agent. "
                    "Analyze the user request and return a structured JSON plan."
                )
            },
            {
                "role": "user",
                "content": build_planning_prompt(user_prompt)
            }
        ]

        plan_response = model_router.chat(
            model_id=model_id,
            messages=planning_messages,
            max_tokens=1000,
            temperature=0.1
        )
        tokens_used += plan_response["input_tokens"] + plan_response["output_tokens"]
        plan = _parse_plan(plan_response["content"])

        if is_new_page:
            plan["decision"] = "full_rewrite"
            plan["needs_clarification"] = False

        # for imported pages with clear existing structure, nudge toward surgical
        if is_imported and not is_new_page and plan.get("decision") != "full_rewrite":
            plan["decision"] = "surgical_edit"

        # block clarification if already asked twice consecutively
        if clarification_blocked and plan.get("needs_clarification"):
            plan["needs_clarification"] = False
            plan["forced_proceed"] = True

        if plan.get("needs_clarification") and not is_new_page and plan.get("confidence", 1.0) < 0.6:
            question = plan.get("clarification_question", "Could you clarify what you would like?")
            insert_clarification(page_id, message_id, question)
            clarification_asked = True
            update_message_status(message_id, "completed")
            insert_assistant_message(
                page_id,
                question,
                message_type="clarification",
                meta={"awaiting_clarification": True, "reason": plan.get("description", "")}
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
                model_used=model_id,
                tokens_used=tokens_used,
                success=True,
                owner_id=owner_id
            )
            if owner_id:
                deduct_tokens(owner_id, tokens_used, "Planning (clarification)", message_id)
            return

        insert_thinking_message(page_id, plan)

        # optional web search
        if plan.get("needs_web_search") and plan.get("search_query"):
            search_results = await brave_search(plan["search_query"])
            web_searches_used.append({
                "query": plan["search_query"],
                "results": search_results
            })
            search_context = f"\nWEB SEARCH RESULTS for '{plan['search_query']}':\n{format_search_results(search_results)}\n"
        else:
            search_context = ""

        # build system prompt
        system_prompt = build_orchestrator_system_prompt(
            current_html=current_html,
            html_summary=html_summary,
            component_map=component_map,
            edit_history=edit_history,
            chat_history=chat_history
        )

        if asset_context:
            system_prompt = system_prompt + f"\n\n{asset_context}"

        if search_context:
            system_prompt = system_prompt + f"\n\n{search_context}"

        if is_imported and not is_new_page:
            system_prompt = system_prompt + (
                "\n\nIMPORTED PAGE NOTE: This page was imported by the user from existing code. "
                "Preserve the existing layout, structure, and design unless explicitly told to change it. "
                "Make only the specific requested changes surgically."
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # agentic tool loop
        max_iterations = 15
        iteration = 0
        final_summary = "Done."

        while iteration < max_iterations:
            iteration += 1

            response = model_router.chat(
                model_id=model_id,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                max_tokens=8000,
                temperature=0.3
            )
            tokens_used += response["input_tokens"] + response["output_tokens"]

            if not response["tool_calls"]:
                final_summary = response["content"] or "Changes applied."
                break

            tool_results_for_messages = []

            for tool_call in response["tool_calls"]:
                fn_name = tool_call["name"]
                args    = tool_call["arguments"]
                tc_id   = tool_call["id"]

                if fn_name == "write_full_file":
                    html = args.get("html", "")
                    summary = args.get("summary", "Page created.")
                    new_html_summary  = args.get("html_summary", "")
                    new_component_map = args.get("component_map", [])

                    if not html:
                        tool_results_for_messages.append({
                            "tool_call_id": tc_id,
                            "result": "ERROR: html field is empty. You must provide complete HTML content."
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
                        model_used=model_id,
                        tokens_used=tokens_used,
                        success=True,
                        owner_id=owner_id
                    )
                    if owner_id:
                        deduct_tokens(owner_id, tokens_used, f"AI page build: {summary[:80]}", message_id)
                    return

                elif fn_name == "str_replace":
                    old_str = args.get("old_str", "")
                    new_str = args.get("new_str", "")
                    updated_html, success = execute_str_replace(current_html, old_str, new_str)

                    if success:
                        current_html = updated_html
                        update_page_html(page_id, current_html)
                        changes_log.append({"tool": "str_replace", "old_str_preview": old_str[:80], "success": True})
                        tool_results_for_messages.append({"tool_call_id": tc_id, "result": "replaced successfully"})
                    else:
                        changes_log.append({"tool": "str_replace", "old_str_preview": old_str[:80], "success": False})
                        tool_results_for_messages.append({
                            "tool_call_id": tc_id,
                            "result": (
                                "ERROR: old_str not found in the file. "
                                "Check for exact whitespace and indentation match. "
                                "Try a shorter, more unique substring."
                            )
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
                        meta={"awaiting_clarification": True}
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
                        model_used=model_id,
                        tokens_used=tokens_used,
                        success=True,
                        owner_id=owner_id
                    )
                    if owner_id:
                        deduct_tokens(owner_id, tokens_used, "Planning (clarification)", message_id)
                    return

                elif fn_name == "web_search":
                    query = args.get("query", "")
                    search_results = await brave_search(query)
                    web_searches_used.append({"query": query, "results": search_results})
                    formatted = format_search_results(search_results)
                    tool_results_for_messages.append({"tool_call_id": tc_id, "result": formatted})

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
                        model_used=model_id,
                        tokens_used=tokens_used,
                        success=True,
                        owner_id=owner_id
                    )
                    if owner_id:
                        deduct_tokens(owner_id, tokens_used, f"AI edit: {final_summary[:80]}", message_id)
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
                            "arguments": json.dumps(tc["arguments"])
                        }
                    }
                    for tc in response["tool_calls"]
                ]
            }
            messages.append(assistant_msg)

            for result in tool_results_for_messages:
                messages.append({
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "content": result["result"]
                })

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
            model_used=model_id,
            tokens_used=tokens_used,
            success=True,
            owner_id=owner_id
        )
        if owner_id:
            deduct_tokens(owner_id, tokens_used, f"AI edit: {final_summary[:80]}", message_id)

    except Exception as e:
        print(f"[ORCHESTRATOR ERROR] {e}")
        update_message_status(message_id, "error")
        insert_assistant_message(page_id, "Something went wrong. Please try again.")
        insert_edit_history(
            page_id=page_id,
            message_id=message_id,
            complexity="unknown",
            decision="unknown",
            plan_json=plan,
            changes_json=changes_log,
            clarification_asked=clarification_asked,
            web_searches_used=web_searches_used,
            model_used=model_id,
            tokens_used=tokens_used,
            success=False,
            owner_id=owner_id
        )
        if owner_id and tokens_used > 0:
            deduct_tokens(owner_id, tokens_used, "AI edit (error)", message_id)