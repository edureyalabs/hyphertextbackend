# agents/models/router.py
"""
Model router — dispatches to the correct provider client.

Providers:
  groq/*          → agents/models/groq_client
  deepinfra/*     → agents/models/deepinfra_client

The orchestrator selects logical model aliases (CODING_MODEL_COMPLEX,
CODING_MODEL_SIMPLE, PLANNING_MODEL, CONVERSATION_MODEL) from config.py.
This router simply resolves those aliases to the right provider.
"""

from config import GROQ_MODELS, TOGETHER_MODELS
from agents.models import groq_client, together_client


def chat(
    model_id: str,
    messages: list,
    tools: list = None,
    tool_choice=None,
    max_tokens: int = 8000,
    temperature: float = 0.3,
) -> dict:
    """
    Route a chat request to the appropriate provider.

    Returns:
        dict: { content, tool_calls, input_tokens, output_tokens }
    """
    if model_id in GROQ_MODELS:
        return groq_client.chat(
            model_id, messages, tools, tool_choice, max_tokens, temperature
        )
    elif model_id in TOGETHER_MODELS:
        return together_client.chat(
            model_id, messages, tools, tool_choice, max_tokens, temperature
        )
    else:
        raise ValueError(f"Unknown model: {model_id!r}. "
                         f"Valid options: {list(GROQ_MODELS) + list(TOGETHER_MODELS)}")