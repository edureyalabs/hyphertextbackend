# agents.py
import json
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL
from database import (
    get_page, update_page_html, get_chat_history,
    update_message_status, insert_assistant_message, snapshot_version
)

client = Groq(api_key=GROQ_API_KEY)

# ============================================================
# TOOLS DEFINITIONS
# ============================================================

CREATE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write_full_file",
            "description": (
                "Write the complete HTML file from scratch. "
                "Called once to produce the full page. "
                "Must be a complete, valid, self-contained HTML document."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "html": {
                        "type": "string",
                        "description": "The complete HTML file content including doctype, head, and body."
                    },
                    "summary": {
                        "type": "string",
                        "description": "A short 1-2 sentence description of what was built, shown to the user."
                    }
                },
                "required": ["html", "summary"]
            }
        }
    }
]

EDIT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "str_replace",
            "description": (
                "Replace an exact string in the current HTML with new content. "
                "The old_str must match EXACTLY as it appears in the file — "
                "same whitespace, same indentation. "
                "Call this multiple times for multiple independent changes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "old_str": {
                        "type": "string",
                        "description": "The exact string to find and replace. Must be unique in the file."
                    },
                    "new_str": {
                        "type": "string",
                        "description": "The replacement string."
                    }
                },
                "required": ["old_str", "new_str"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Signal that all edits are complete. Must be called once at the end.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "A short 1-2 sentence human-readable summary of what was changed."
                    }
                },
                "required": ["summary"]
            }
        }
    }
]

# ============================================================
# TOOL EXECUTORS
# ============================================================

def execute_write_full_file(page_id: str, html: str) -> str:
    update_page_html(page_id, html)
    return "full_file_written"


def execute_str_replace(current_html: str, old_str: str, new_str: str) -> tuple[str, bool]:
    """Returns (updated_html, success)"""
    if old_str not in current_html:
        return current_html, False
    updated = current_html.replace(old_str, new_str, 1)  # replace only first occurrence
    return updated, True

# ============================================================
# SYSTEM PROMPTS
# ============================================================

CREATE_SYSTEM_PROMPT = """You are an elite HTML/CSS/JS developer building single-page web experiences.

Your job: take the user's description and produce a stunning, complete, self-contained HTML page.

RULES:
- Call write_full_file ONCE with the complete HTML
- The HTML must be a full document: <!DOCTYPE html>, <head>, <body>
- Make it beautiful — use Google Fonts, good typography, proper spacing
- All CSS must be inline in a <style> tag in <head>
- All JS must be inline in a <script> tag before </body>
- No external dependencies except Google Fonts CDN
- Do NOT use placeholder lorem ipsum text — write real, contextual content
- After write_full_file, you are done. Do not call any other tools."""

EDIT_SYSTEM_PROMPT = """You are an elite HTML/CSS/JS developer performing surgical edits to an existing page.

CURRENT HTML FILE:
{current_html}

RULES:
- Use str_replace for every change — find the exact string, replace it
- old_str must match EXACTLY (whitespace, indentation, everything)
- Make multiple str_replace calls for multiple independent changes
- Do NOT rewrite large blocks unnecessarily — only change what was asked
- After all edits are done, call finish with a summary
- Do NOT call write_full_file"""

# ============================================================
# AGENT 1: CREATE AGENT
# ============================================================

async def run_create_agent(page_id: str, message_id: str, user_prompt: str):
    """First-prompt agent — writes the full HTML file."""
    try:
        update_message_status(message_id, "processing")

        messages = [
            {"role": "system", "content": CREATE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            tools=CREATE_TOOLS,
            tool_choice={"type": "function", "function": {"name": "write_full_file"}},
            max_tokens=8000,
            temperature=0.7
        )

        message = response.choices[0].message

        if not message.tool_calls:
            raise ValueError("Agent did not call write_full_file")

        tool_call = message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)

        html = args.get("html", "")
        summary = args.get("summary", "Page created.")

        if not html:
            raise ValueError("Agent returned empty HTML")

        # Write to DB — Realtime fires here, frontend preview updates
        execute_write_full_file(page_id, html)

        # Snapshot this version
        snapshot_version(page_id, html)

        # Mark user message as completed
        update_message_status(message_id, "completed")

        # Insert assistant reply
        insert_assistant_message(page_id, summary)

    except Exception as e:
        print(f"[CREATE AGENT ERROR] {e}")
        update_message_status(message_id, "error")
        insert_assistant_message(page_id, f"Something went wrong while generating the page. Please try again.")

# ============================================================
# AGENT 2: EDIT AGENT
# ============================================================

async def run_edit_agent(page_id: str, message_id: str, user_prompt: str):
    """Edit agent — uses str_replace for surgical changes."""
    try:
        update_message_status(message_id, "processing")

        # Fetch current state of the HTML
        page = get_page(page_id)
        current_html = page["html_content"]

        # Fetch recent conversation history for context
        history = get_chat_history(page_id, limit=10)

        system_prompt = EDIT_SYSTEM_PROMPT.format(current_html=current_html)

        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        # Add current user message
        messages.append({"role": "user", "content": user_prompt})

        # Agent loop — runs until finish is called or max iterations hit
        max_iterations = 10
        iteration = 0
        final_summary = "Done."

        while iteration < max_iterations:
            iteration += 1

            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                tools=EDIT_TOOLS,
                tool_choice="auto",
                max_tokens=4000,
                temperature=0.3  # lower temp for precise edits
            )

            message = response.choices[0].message

            if not message.tool_calls:
                # No tool call — agent responded in text, treat as done
                final_summary = message.content or "Edits complete."
                break

            # Process each tool call in this response
            tool_results = []

            for tool_call in message.tool_calls:
                fn_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                if fn_name == "str_replace":
                    old_str = args.get("old_str", "")
                    new_str = args.get("new_str", "")

                    updated_html, success = execute_str_replace(current_html, old_str, new_str)

                    if success:
                        current_html = updated_html
                        # Push update to DB immediately — Realtime fires, preview refreshes
                        update_page_html(page_id, current_html)
                        tool_results.append({
                            "tool_call_id": tool_call.id,
                            "result": "replaced successfully"
                        })
                    else:
                        tool_results.append({
                            "tool_call_id": tool_call.id,
                            "result": f"ERROR: old_str not found in file. Make sure it matches exactly."
                        })

                elif fn_name == "finish":
                    final_summary = args.get("summary", "Edits complete.")
                    # Snapshot on finish
                    snapshot_version(page_id, current_html)
                    update_message_status(message_id, "completed")
                    insert_assistant_message(page_id, final_summary)
                    return  # cleanly exit the loop

            # Add assistant message and tool results to conversation for next iteration
            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ]
            })

            for result in tool_results:
                messages.append({
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "content": result["result"]
                })

        # If we exit the loop without finish being called
        snapshot_version(page_id, current_html)
        update_message_status(message_id, "completed")
        insert_assistant_message(page_id, final_summary)

    except Exception as e:
        print(f"[EDIT AGENT ERROR] {e}")
        update_message_status(message_id, "error")
        insert_assistant_message(page_id, "Something went wrong while editing. Please try again.")