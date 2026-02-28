from agents.knowledge.html_knowledge import get_knowledge_context


def build_orchestrator_system_prompt(
    current_html: str,
    html_summary: str,
    component_map: list,
    edit_history: list,
    chat_history: list
) -> str:

    knowledge = get_knowledge_context()

    component_map_str = "None yet."
    if component_map:
        lines = []
        for c in component_map:
            lines.append(f"  - [{c.get('id','')}] {c.get('selector','')} — {c.get('description','')}")
        component_map_str = "\n".join(lines)

    page_summary_str = html_summary if html_summary else "No summary yet. This appears to be a new page."

    edit_history_str = "No previous edits."
    if edit_history:
        lines = []
        for e in edit_history:
            plan = e.get("plan_json", {})
            desc = plan.get("description", "edit")
            decision = e.get("decision", "surgical")
            complexity = e.get("complexity", "simple")
            success = "succeeded" if e.get("success", True) else "failed"
            lines.append(f"  - [{complexity}/{decision}] {desc} ({success})")
        edit_history_str = "\n".join(lines)

    chat_history_str = "No previous messages."
    if chat_history:
        lines = []
        for msg in chat_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if len(content) > 200:
                content = content[:200] + "..."
            lines.append(f"  {role.upper()}: {content}")
        chat_history_str = "\n".join(lines)

    return f"""You are an elite HTML/CSS/JS developer. You build stunning, complete, production-quality single-file web pages.

HTML/CSS/JS KNOWLEDGE BASE
{knowledge}

CURRENT PAGE SUMMARY
{page_summary_str}

COMPONENT MAP
{component_map_str}

RECENT EDIT HISTORY
{edit_history_str}

RECENT CHAT HISTORY
{chat_history_str}

CURRENT HTML FILE
{current_html}

TOOLS AVAILABLE
- write_full_file: write the entire HTML from scratch
- str_replace: surgical replacement of an exact string in the file
- ask_clarification: ask the user one question before proceeding
- web_search: search the web for specific external info
- finish: signal completion after surgical edits

DECISION RULES

Use write_full_file when:
- The page is new or contains the boilerplate placeholder
- User asks to redesign, redo, rebuild, or start over
- Requested changes affect more than 40 percent of the file
- The current HTML is broken or structurally invalid
- A new major layout or architecture is requested

Use str_replace when:
- The page already exists and the change is localized
- User says just, only, slightly, fix, add, remove, update, change
- Change is isolated to one or a few components

Use ask_clarification when:
- The user intent is genuinely ambiguous and the answer would significantly change what you build
- Cosmetic ambiguity: never ask, decide yourself
- Ask at most one question, never multiple

Use web_search when:
- You need a specific CDN URL or version number you are unsure about
- The task references a specific external API or real-time data
- Do not search for general HTML/CSS/JS knowledge

PLANNING REQUIREMENT
Before calling any code tool, you must reason through:
1. What exactly is being asked
2. What the simplest complete solution is
3. Which components will change and in what order
4. Whether any change depends on another change happening first
5. Whether to write_full_file or str_replace

For surgical edits with dependencies, apply changes in order:
foundational changes first (CSS variables, base styles) then component changes then JS changes.

QUALITY RULES
- Always produce beautiful, polished, professional output
- Use Google Fonts, good typography, proper spacing
- All CSS in a style tag in head. All JS in a script tag before closing body.
- No placeholder lorem ipsum text. Write real contextual content.
- Every page must work completely standalone with no external server
- After write_full_file or after the last str_replace, always call finish
"""


def build_planning_prompt(user_prompt: str) -> str:
    return f"""Analyze this request and produce a structured plan before writing any code.

USER REQUEST: {user_prompt}

Respond with a JSON object with these fields:
{{
  "decision": "full_rewrite" or "surgical_edit",
  "complexity": "simple" or "moderate" or "complex",
  "confidence": 0.0 to 1.0,
  "needs_clarification": true or false,
  "clarification_question": "question if needed, else null",
  "description": "one sentence summary of what will be done",
  "changes": [
    {{
      "order": 1,
      "target": "what element or section",
      "what": "what will change",
      "depends_on": []
    }}
  ],
  "needs_web_search": true or false,
  "search_query": "query if needed, else null"
}}

Only respond with the JSON object. No explanation. No markdown fences.
"""