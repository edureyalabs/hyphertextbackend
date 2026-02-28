import json
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODELS

_client = Groq(api_key=GROQ_API_KEY)


def chat(model_id: str, messages: list, tools: list = None, tool_choice=None, max_tokens: int = 8000, temperature: float = 0.3) -> dict:
    model_name = GROQ_MODELS.get(model_id)
    if not model_name:
        raise ValueError(f"Unknown Groq model: {model_id}")

    kwargs = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if tools:
        kwargs["tools"] = tools
    if tool_choice:
        kwargs["tool_choice"] = tool_choice

    response = _client.chat.completions.create(**kwargs)
    msg = response.choices[0].message

    tool_calls = None
    if msg.tool_calls:
        tool_calls = [
            {
                "id": tc.id,
                "name": tc.function.name,
                "arguments": json.loads(tc.function.arguments)
            }
            for tc in msg.tool_calls
        ]

    return {
        "content": msg.content or "",
        "tool_calls": tool_calls,
        "input_tokens": response.usage.prompt_tokens if response.usage else 0,
        "output_tokens": response.usage.completion_tokens if response.usage else 0,
    }