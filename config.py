import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY       = os.getenv("GROQ_API_KEY")
ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY")   # kept for image vision (claude-haiku)
TOGETHER_API_KEY   = os.getenv("TOGETHER_API_KEY")
BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY")

SUPABASE_URL              = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# ── Groq: intent classification, planning, conversation ───────────────────────
GROQ_MODELS = {
    "groq/llama-3.3-70b": "llama-3.3-70b-versatile",
    "groq/llama-3.1-8b":  "llama-3.1-8b-instant",
}

# ── Together AI: all HTML coding tasks ───────────────────────────────────────
# GLM-5         → complex / full rewrites / new page builds / imported pages
# GLM-4.7-Flash → surgical edits, simple fixes, single-component changes
TOGETHER_MODELS = {
    "together/glm-5":         "zai-org/GLM-5",
    "together/glm-4.7-flash": "zai-org/GLM-4.7-Flash",
}

TOGETHER_BASE_URL = "https://api.together.xyz/v1"

# ── Internal role aliases (never exposed to frontend) ────────────────────────
CODING_MODEL_COMPLEX = "together/glm-5"           # full rewrites, new pages, complex edits
CODING_MODEL_SIMPLE  = "together/glm-4.7-flash"   # surgical edits, simple changes
PLANNING_MODEL       = "groq/llama-3.3-70b"        # planning + intent classification
CONVERSATION_MODEL   = "groq/llama-3.1-8b"         # lightweight chat replies

ALL_MODELS = list(GROQ_MODELS.keys()) + list(TOGETHER_MODELS.keys())

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"