# agents/models/router.py
"""
Model router — dispatches to the correct provider client.

Providers currently in use:
  groq/*          → agents/models/groq_client
  cerebras/*      → agents/models/cerebras_client

Together AI client is kept but not used in the active coding path.
"""

from config import GROQ_MODELS, TOGETHER_MODELS, CEREBRAS_MODELS
from agents.models import groq_client, together_client, cerebras_client


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
    elif model_id in CEREBRAS_MODELS:
        return cerebras_client.chat(
            model_id, messages, tools, tool_choice, max_tokens, temperature
        )
    elif model_id in TOGETHER_MODELS:
        # Kept for future use — not in active coding path
        return together_client.chat(
            model_id, messages, tools, tool_choice, max_tokens, temperature
        )
    else:
        raise ValueError(
            f"Unknown model: {model_id!r}. "
            f"Valid options: {list(GROQ_MODELS) + list(CEREBRAS_MODELS) + list(TOGETHER_MODELS)}"
        )