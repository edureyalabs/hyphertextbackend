# agents/models/deepinfra_client.py
"""
DeepInfra client using the OpenAI-compatible API.
DeepInfra exposes all models at https://api.deepinfra.com/v1/openai
with standard OpenAI message/tool schemas — no conversion needed.
"""

import json
from openai import OpenAI
from config import DEEPINFRA_API_KEY, DEEPINFRA_BASE_URL, DEEPINFRA_MODELS

_client = OpenAI(
    api_key=DEEPINFRA_API_KEY,
    base_url=DEEPINFRA_BASE_URL,
)


def chat(
    model_id: str,
    messages: list,
    tools: list = None,
    tool_choice=None,
    max_tokens: int = 8000,
    temperature: float = 0.3,
) -> dict:
    """
    Send a chat request to DeepInfra.

    Args:
        model_id: Internal alias like "deepinfra/glm-5"
        messages:  OpenAI-format message list (system/user/assistant/tool)
        tools:     OpenAI-format tool definitions (function schema)
        tool_choice: "auto" | {"type": "function", "function": {"name": ...}}
        max_tokens: Maximum output tokens
        temperature: Sampling temperature

    Returns:
        dict with keys: content, tool_calls, input_tokens, output_tokens
    """
    model_name = DEEPINFRA_MODELS.get(model_id)
    if not model_name:
        raise ValueError(f"Unknown DeepInfra model: {model_id}")

    kwargs = {
        "model":       model_name,
        "messages":    messages,
        "max_tokens":  max_tokens,
        "temperature": temperature,
    }

    if tools:
        kwargs["tools"] = tools

    # Resolve tool_choice into OpenAI format
    if tool_choice == "auto":
        kwargs["tool_choice"] = "auto"
    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        kwargs["tool_choice"] = tool_choice  # pass through as-is

    response = _client.chat.completions.create(**kwargs)
    msg = response.choices[0].message

    content_text = msg.content or ""
    parsed_tool_calls = []

    if msg.tool_calls:
        for tc in msg.tool_calls:
            try:
                arguments = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            parsed_tool_calls.append({
                "id":        tc.id,
                "name":      tc.function.name,
                "arguments": arguments,
            })

    return {
        "content":       content_text,
        "tool_calls":    parsed_tool_calls if parsed_tool_calls else None,
        "input_tokens":  response.usage.prompt_tokens     if response.usage else 0,
        "output_tokens": response.usage.completion_tokens if response.usage else 0,
    }