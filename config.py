import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY         = os.getenv("GROQ_API_KEY")
ANTHROPIC_API_KEY    = os.getenv("ANTHROPIC_API_KEY")   # kept for image vision (claude-haiku)
TOGETHER_API_KEY     = os.getenv("TOGETHER_API_KEY")
CEREBRAS_API_KEY     = os.getenv("CEREBRAS_API_KEY")
BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY")

SUPABASE_URL              = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# ── Groq: intent classification, planning, conversation ───────────────────────
GROQ_MODELS = {
    "groq/llama-3.3-70b": "llama-3.3-70b-versatile",
    "groq/llama-3.1-8b":  "llama-3.1-8b-instant",
}

# ── Together AI: complex HTML coding tasks (Economy mode) ─────────────────────
# GLM-5         → complex / full rewrites / new page builds / imported pages
# GLM-4.7-Flash → surgical edits, simple fixes, single-component changes (Economy)
TOGETHER_MODELS = {
    "together/glm-5":         "zai-org/GLM-5",
    "together/glm-4.7-flash": "zai-org/GLM-4.7-Flash",
}

TOGETHER_BASE_URL = "https://api.together.xyz/v1"

# ── Cerebras: ultra-fast HTML coding (Speed mode) ─────────────────────────────
# GLM-4.7 on Cerebras wafer-scale hardware — ~1,000-1,700 tokens/sec
# Used for ALL coding tasks (both full rewrites and surgical edits) when
# the user selects Speed mode. disable_reasoning=True keeps output clean.
CEREBRAS_MODELS = {
    "cerebras/glm-4.7": "zai-glm-4.7",
}

CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"

# ── Internal role aliases (never exposed to frontend) ────────────────────────
CODING_MODEL_COMPLEX  = "together/glm-5"           # Economy: full rewrites, new pages, complex edits
CODING_MODEL_SIMPLE   = "together/glm-4.7-flash"   # Economy: surgical edits, simple changes
CODING_MODEL_SPEED    = "cerebras/glm-4.7"         # Speed: ALL coding tasks via Cerebras
PLANNING_MODEL        = "groq/llama-3.3-70b"        # planning + intent classification
CONVERSATION_MODEL    = "groq/llama-3.1-8b"         # lightweight chat replies

ALL_MODELS = (
    list(GROQ_MODELS.keys())
    + list(TOGETHER_MODELS.keys())
    + list(CEREBRAS_MODELS.keys())
)

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"