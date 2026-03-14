# agents/models/cerebras_client.py
"""
Cerebras Inference client — fully async using AsyncOpenAI.

Key behaviours:
1. Uses max_completion_tokens (not max_tokens) — Cerebras spec.
2. disable_reasoning passed as extra_body to suppress <think> blocks.
   Some org-tier keys reject it — we catch BadRequestError and retry once without it.
3. All network calls are non-blocking (await).
"""

import json
import logging
import traceback
from openai import AsyncOpenAI, BadRequestError
from config import CEREBRAS_API_KEY, CEREBRAS_BASE_URL, CEREBRAS_MODELS

logger = logging.getLogger(__name__)

_client = AsyncOpenAI(
    api_key=CEREBRAS_API_KEY,
    base_url=CEREBRAS_BASE_URL,
)


async def _do_request(model_name: str, kwargs: dict) -> dict:
    """Execute the API call and normalise response into our standard dict."""
    response = await _client.chat.completions.create(**kwargs)
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


async def chat(
    model_id: str,
    messages: list,
    tools: list = None,
    tool_choice=None,
    max_tokens: int = 8000,
    temperature: float = 0.3,
) -> dict:
    model_name = CEREBRAS_MODELS.get(model_id)
    if not model_name:
        raise ValueError(f"Unknown Cerebras model: {model_id}")

    kwargs = {
        "model":                 model_name,
        "messages":              messages,
        "max_completion_tokens": max_tokens,
        "temperature":           temperature,
        "extra_body": {
            "disable_reasoning": False,
        },
    }

    if tools:
        kwargs["tools"] = tools

    if tool_choice == "auto":
        kwargs["tool_choice"] = "auto"
    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        kwargs["tool_choice"] = tool_choice

    # Attempt 1: with disable_reasoning
    try:
        return await _do_request(model_name, kwargs)

    except BadRequestError as e:
        err_body = str(e)
        if "disable_reasoning" in err_body or "extra_body" in err_body or "unknown" in err_body.lower():
            logger.warning(
                "[cerebras_client] disable_reasoning rejected (%s). Retrying without it.",
                err_body[:200],
            )
            kwargs_retry = {k: v for k, v in kwargs.items() if k != "extra_body"}
            return await _do_request(model_name, kwargs_retry)
        raise

    except Exception:
        logger.error("[cerebras_client] Unexpected error:\n%s", traceback.format_exc())
        raise