from config import GROQ_MODELS, CLAUDE_MODELS
from agents.models import groq_client, claude_client


def chat(model_id: str, messages: list, tools: list = None, tool_choice=None, max_tokens: int = 8000, temperature: float = 0.3) -> dict:
    if model_id in GROQ_MODELS:
        return groq_client.chat(model_id, messages, tools, tool_choice, max_tokens, temperature)
    elif model_id in CLAUDE_MODELS:
        return claude_client.chat(model_id, messages, tools, tool_choice, max_tokens, temperature)
    else:
        raise ValueError(f"Unknown model: {model_id}")