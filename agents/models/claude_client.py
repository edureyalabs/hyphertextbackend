import anthropic
import json
from config import ANTHROPIC_API_KEY, CLAUDE_MODELS

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def _convert_tools_to_anthropic(tools: list) -> list:
    converted = []
    for t in tools:
        fn = t["function"]
        converted.append({
            "name": fn["name"],
            "description": fn["description"],
            "input_schema": fn["parameters"]
        })
    return converted


def _convert_messages_for_anthropic(messages: list) -> tuple[str, list]:
    system_prompt = ""
    converted = []

    for msg in messages:
        role = msg["role"]

        if role == "system":
            system_prompt = msg["content"]
            continue

        if role == "tool":
            converted.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": msg["tool_call_id"],
                        "content": msg["content"]
                    }
                ]
            })
            continue

        if role == "assistant" and msg.get("tool_calls"):
            content_blocks = []
            if msg.get("content"):
                content_blocks.append({"type": "text", "text": msg["content"]})
            for tc in msg["tool_calls"]:
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "input": json.loads(tc["function"]["arguments"])
                })
            converted.append({"role": "assistant", "content": content_blocks})
            continue

        converted.append({"role": role, "content": msg["content"]})

    return system_prompt, converted


def chat(model_id: str, messages: list, tools: list = None, tool_choice=None, max_tokens: int = 8000, temperature: float = 0.3) -> dict:
    model_name = CLAUDE_MODELS.get(model_id)
    if not model_name:
        raise ValueError(f"Unknown Claude model: {model_id}")

    system_prompt, converted_messages = _convert_messages_for_anthropic(messages)

    kwargs = {
        "model": model_name,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": converted_messages,
    }
    if system_prompt:
        kwargs["system"] = system_prompt
    if tools:
        kwargs["tools"] = _convert_tools_to_anthropic(tools)
    if tool_choice and isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        kwargs["tool_choice"] = {"type": "tool", "name": tool_choice["function"]["name"]}
    elif tool_choice == "auto":
        kwargs["tool_choice"] = {"type": "auto"}

    response = _client.messages.create(**kwargs)

    content_text = ""
    tool_calls = None
    parsed_tool_calls = []

    for block in response.content:
        if block.type == "text":
            content_text = block.text
        elif block.type == "tool_use":
            parsed_tool_calls.append({
                "id": block.id,
                "name": block.name,
                "arguments": block.input
            })

    if parsed_tool_calls:
        tool_calls = parsed_tool_calls

    return {
        "content": content_text,
        "tool_calls": tool_calls,
        "input_tokens": response.usage.input_tokens if response.usage else 0,
        "output_tokens": response.usage.output_tokens if response.usage else 0,
    }