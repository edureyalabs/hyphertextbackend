TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "write_full_file",
            "description": (
                "Write a complete HTML file from scratch. "
                "Use this when building a new page, doing a full redesign, "
                "or when changes affect more than 40 percent of the file. "
                "Must produce a complete valid self-contained HTML document."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "html": {
                        "type": "string",
                        "description": "The complete HTML file including doctype, head, and body."
                    },
                    "summary": {
                        "type": "string",
                        "description": "1-2 sentence description of what was built."
                    },
                    "html_summary": {
                        "type": "string",
                        "description": "A 300-500 word plain text description of what this page is, its sections, its state shape, and the key JS logic. Used as context for future edits."
                    },
                    "component_map": {
                        "type": "array",
                        "description": "Array of key components/sections in the page.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "selector": {"type": "string"},
                                "type": {"type": "string"},
                                "description": {"type": "string"}
                            }
                        }
                    }
                },
                "required": ["html", "summary", "html_summary", "component_map"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "str_replace",
            "description": (
                "Replace an exact string in the current HTML with new content. "
                "old_str must match EXACTLY as it appears in the file including whitespace and indentation. "
                "Call multiple times for multiple independent changes. "
                "Never use this to replace more than 60 percent of the file."
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
            "name": "ask_clarification",
            "description": (
                "Ask the user one precise clarifying question before proceeding. "
                "Only call this when the user intent is genuinely ambiguous and the answer would significantly change what you build. "
                "Do not ask about cosmetic choices. Do not ask more than one question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The single clarifying question to ask the user."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why this information is necessary to proceed correctly."
                    }
                },
                "required": ["question", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current information. "
                "Use only when the task requires knowing a specific CDN URL, library API, real-time data, or something not in your training. "
                "Do not use for general HTML/CSS/JS knowledge."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why this search is needed."
                    }
                },
                "required": ["query", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Signal that all edits are complete. Must be called once at the end of every surgical edit session.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "1-2 sentence human readable summary of what was changed."
                    },
                    "updated_component_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of component ids from the component_map that were modified."
                    }
                },
                "required": ["summary", "updated_component_ids"]
            }
        }
    }
]


def execute_str_replace(current_html: str, old_str: str, new_str: str) -> tuple[str, bool]:
    if old_str not in current_html:
        return current_html, False
    updated = current_html.replace(old_str, new_str, 1)
    return updated, True