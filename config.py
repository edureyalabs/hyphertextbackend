import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

GROQ_MODELS = {
    "groq/llama-3.3-70b": "llama-3.3-70b-versatile",
    "groq/llama-3.1-8b": "llama-3.1-8b-instant",
}

CLAUDE_MODELS = {
    "claude/claude-sonnet-4-5": "claude-sonnet-4-5",
    "claude/claude-haiku-4-5": "claude-haiku-4-5",
}

ALL_MODELS = list(GROQ_MODELS.keys()) + list(CLAUDE_MODELS.keys())

DEFAULT_MODEL = "groq/llama-3.3-70b"

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"