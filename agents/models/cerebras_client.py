# agents/models/cerebras_client.py
"""
Cerebras Inference client using the OpenAI-compatible API.
Cerebras exposes models at https://api.cerebras.ai/v1
with standard OpenAI message/tool schemas.

Key differences vs Together/DeepInfra:
1. Uses max_completion_tokens (not max_tokens) — Cerebras spec.
2. disable_reasoning is passed as a top-level body param via extra_body.
   IMPORTANT: some org-tier keys reject disable_reasoning entirely and return
   a 400. We catch that and retry once without it, so the call always lands.
3. Tool-call parsing is identical to other OpenAI-compat clients.
"""

import json
import logging
import traceback
from openai import OpenAI, BadRequestError
from config import CEREBRAS_API_KEY, CEREBRAS_BASE_URL, CEREBRAS_MODELS

logger = logging.getLogger(__name__)

_client = OpenAI(
    api_key=CEREBRAS_API_KEY,
    base_url=CEREBRAS_BASE_URL,
)


def _do_request(model_name: str, kwargs: dict) -> dict:
    """Execute the API call and normalise the response into our standard dict."""
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


def chat(
    model_id: str,
    messages: list,
    tools: list = None,
    tool_choice=None,
    max_tokens: int = 8000,
    temperature: float = 0.3,
) -> dict:
    """
    Send a chat request to Cerebras Inference.

    Args:
        model_id:    Internal alias like "cerebras/glm-4.7"
        messages:    OpenAI-format message list (system/user/assistant/tool)
        tools:       OpenAI-format tool definitions (function schema)
        tool_choice: "auto" | {"type": "function", "function": {"name": ...}}
        max_tokens:  Maximum output tokens
        temperature: Sampling temperature

    Returns:
        dict with keys: content, tool_calls, input_tokens, output_tokens

    Raises:
        ValueError: if model_id is unknown
        Exception:  propagated from the underlying HTTP call (no silent swallow)
    """
    model_name = CEREBRAS_MODELS.get(model_id)
    if not model_name:
        raise ValueError(f"Unknown Cerebras model: {model_id}")

    # Cerebras uses max_completion_tokens, not max_tokens.
    # Passing max_tokens causes a 400 on some org-tier accounts.
    kwargs = {
        "model":                model_name,
        "messages":             messages,
        "max_completion_tokens": max_tokens,
        "temperature":          temperature,
        # disable_reasoning suppresses <think>...</think> blocks that inflate
        # token cost without improving HTML output quality.
        # Passed as extra_body so the openai SDK includes it at the top-level
        # of the JSON body (Cerebras non-standard field).
        "extra_body": {
            "disable_reasoning": True,
        },
    }

    if tools:
        kwargs["tools"] = tools

    if tool_choice == "auto":
        kwargs["tool_choice"] = "auto"
    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        kwargs["tool_choice"] = tool_choice

    # ── Attempt 1: with disable_reasoning ────────────────────────────────────
    try:
        return _do_request(model_name, kwargs)

    except BadRequestError as e:
        # Some Cerebras org-tier keys reject disable_reasoning.
        # Retry once without it rather than failing the whole request.
        err_body = str(e)
        if "disable_reasoning" in err_body or "extra_body" in err_body or "unknown" in err_body.lower():
            logger.warning(
                "[cerebras_client] disable_reasoning rejected by API (%s). "
                "Retrying without it.", err_body[:200]
            )
            kwargs_retry = {k: v for k, v in kwargs.items() if k != "extra_body"}
            return _do_request(model_name, kwargs_retry)
        # Any other 400 — re-raise so orchestrator sees it
        raise

    except Exception:
        # Log the full traceback so it's visible in server logs, then re-raise.
        # The orchestrator's outer try/except will handle it.
        logger.error(
            "[cerebras_client] Unexpected error:\n%s", traceback.format_exc()
        )
        raise