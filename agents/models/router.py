# agents/models/router.py
"""
Model router — async dispatch to the correct provider client.
All provider clients are now fully async — every call must be awaited.
"""

from config import GROQ_MODELS, TOGETHER_MODELS, CEREBRAS_MODELS
from agents.models import groq_client, together_client, cerebras_client


async def chat(
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
        return await groq_client.chat(
            model_id, messages, tools, tool_choice, max_tokens, temperature
        )
    elif model_id in CEREBRAS_MODELS:
        return await cerebras_client.chat(
            model_id, messages, tools, tool_choice, max_tokens, temperature
        )
    elif model_id in TOGETHER_MODELS:
        return await together_client.chat(
            model_id, messages, tools, tool_choice, max_tokens, temperature
        )
    else:
        raise ValueError(
            f"Unknown model: {model_id!r}. "
            f"Valid options: {list(GROQ_MODELS) + list(CEREBRAS_MODELS) + list(TOGETHER_MODELS)}"
        )